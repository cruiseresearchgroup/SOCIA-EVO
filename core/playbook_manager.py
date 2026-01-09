"""
PlaybookManager
---------------

A manager for ACE mode that maintains a JSON playbook file with three-level persistence:

1. **Real-time Level (WAL)**: Write-Ahead Logging for every operation.
2. **Checkpoint Level (Snapshot)**: State snapshots at key milestones.
3. **Session Level (Final Archive)**: Final cleanup and archival when task ends.

Directory Structure:
    /playbook_storage
      /current
        playbook.json  (latest version)
        playbook.log   (incremental JSONL log)
      /snapshots
        playbook_{timestamp}_iter_001.json
        playbook_{timestamp}_first_compile_success.json
        playbook_{timestamp}_best_loss_0.15.json

Default playbook keys:
1) "playbook_metadata": {
       "version": "v0.1",
       "project_name": "",
       "last_updated_time": "",
       "last_updated_iteration": ""
   }
2) "strategies": {}
"""

import argparse
import glob
import json
import logging
import os
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

# Lazy loading for sentence-transformers (heavy import)
_sbert_model = None

def _get_sbert_model():
    """Lazy load SBERT model to avoid slow import at startup."""
    global _sbert_model
    if _sbert_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for similarity calculation. "
                "Install with: pip install sentence-transformers"
            )
    return _sbert_model

# Common English stopwords for text normalization
STOPWORDS = {
    'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 
    'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been', 'be', 'have',
    'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may',
    'might', 'must', 'shall', 'can', 'need', 'dare', 'ought', 'used', 'it', 'its',
    'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'we', 'they', 'them',
    'his', 'her', 'our', 'your', 'their', 'what', 'which', 'who', 'whom', 'when',
    'where', 'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more', 'most',
    'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
    'than', 'too', 'very', 'just', 'also', 'now', 'here', 'there', 'then', 'if',
    'else', 'because', 'about', 'into', 'through', 'during', 'before', 'after',
    'above', 'below', 'between', 'under', 'again', 'further', 'once', 'any',
}


class PlaybookManager:
    """Centralized store for ACE playbook knowledge with three-level persistence."""

    # Default storage root (relative to project root)
    DEFAULT_STORAGE_ROOT = "playbook_storage"
    
    # Maximum number of snapshots to keep (excluding milestones)
    MAX_SNAPSHOTS = 10
    
    # Milestone tags that should not be deleted during cleanup
    MILESTONE_TAGS = ["first_compile_success", "best_loss"]

    # Strategy status constants
    STATUS_OPEN = "open"           # New or reactivated strategy, waiting for selection
    STATUS_QUEUED = "queued"       # Not selected due to budget, waiting in queue
    STATUS_IN_PROGRESS = "in_progress"  # Selected and applied to code patch prompt
    STATUS_RESOLVED = "resolved"   # Successfully resolved (metrics improved or issue disappeared)
    
    # Event types for WAL logging
    EVENT_NEW = "E_new"            # New entry added
    EVENT_MERGE = "E_merge"        # Entry merged with existing strategy
    EVENT_SELECTED = "E_selected"  # Strategy selected for prompt
    EVENT_NOT_SELECTED = "not_selected"  # Strategy not selected (budget limit)
    EVENT_RESOLVED = "E_resolved"  # Strategy resolved (metrics improved)
    EVENT_PERSIST = "E_persist"    # Strategy persists (metrics worsened)
    EVENT_UNCERTAIN = "E_uncertain"  # Strategy uncertain (metrics unchanged or mixed)
    
    # Threshold for metric improvement/worsening (3%)
    METRIC_CHANGE_THRESHOLD = 0.03

    def __init__(
        self,
        storage_root: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """
        Initialize PlaybookManager with three-level persistence.
        
        Args:
            storage_root: Root directory for playbook storage. 
                         Defaults to 'playbook_storage' in project root.
            logger: Optional logger instance.
        """
        self.logger = logger or logging.getLogger("SOCIA.PlaybookManager")
        
        # Determine storage root
        if storage_root:
            self.storage_root = storage_root
        else:
            # Default to project root / playbook_storage
            project_root = self._get_project_root()
            self.storage_root = os.path.join(project_root, self.DEFAULT_STORAGE_ROOT)
        
        # Set up directory paths
        self.current_dir = os.path.join(self.storage_root, "current")
        self.snapshots_dir = os.path.join(self.storage_root, "snapshots")
        self.playbook_path = os.path.join(self.current_dir, "playbook.json")
        self.log_path = os.path.join(self.current_dir, "playbook.log")
        
        # Ensure directories exist
        os.makedirs(self.current_dir, exist_ok=True)
        os.makedirs(self.snapshots_dir, exist_ok=True)
        
        # Initialize playbook: load if exists, otherwise create default
        if os.path.exists(self.playbook_path):
            self.playbook = self._load_from_file(self.playbook_path)
            self.logger.info("Playbook loaded from %s", self.playbook_path)
        else:
            self.playbook = self._default_playbook()
            self._save_current()
            self.logger.info("Playbook initialized with default structure at %s", self.playbook_path)
    
    @staticmethod
    def _get_project_root() -> str:
        """Get the project root directory."""
        # Traverse up from this file to find project root
        current = os.path.dirname(os.path.abspath(__file__))
        # Go up from core/ to project root
        return os.path.dirname(current)

    # ------------------------------------------------------------------ #
    # Default structure
    # ------------------------------------------------------------------ #
    @staticmethod
    def _default_playbook() -> Dict[str, Any]:
        return {
            "playbook_metadata": {
                "version": "v0.1",
                "project_name": "",
                "last_updated_time": "",
                "last_updated_iteration": "",
            },
            "strategies": {},
        }

    # ------------------------------------------------------------------ #
    # Strategy management
    # ------------------------------------------------------------------ #
    @staticmethod
    def _count_tokens(content: str) -> int:
        """Count tokens (words) in content string."""
        if not content:
            return 0
        words = [w for w in content.split() if w.strip()]
        return len(words)
    
    def _count_reflection_tokens(self, reflection: Dict[str, Any]) -> int:
        """Count tokens in reflection fields (all text fields combined)."""
        text_fields = [
            "error_identification",
            "root_cause_analysis",
            "correct_approach",
            "key_insight",
        ]
        total_text = " ".join(
            str(reflection.get(field, ""))
            for field in text_fields
        )
        return self._count_tokens(total_text)
    
    def add_strategy(
        self,
        entry_id: str,
        entry_type: str,
        content: str,
        parent_id: Optional[str] = None,
        status: str = "active",
        usage_count: int = 0,
        unusage_count: int = 0,
        success_attribution: int = 0,
        failure_attribution: int = 0,
    ) -> str:
        """
        Add a strategy entry to the playbook (legacy method, kept for backward compatibility).
        
        Note: This method is deprecated. Use add_feedback_entry or add_feedback_entries instead.
        
        Args:
            entry_id: Unique identifier for the strategy entry (e.g., "bp-agent-001")
            entry_type: Type of the strategy (e.g., "agent_definition")
            content: Content of the strategy entry
            parent_id: Optional parent entry ID
            status: Status of the entry ("active", "obsolete", or "rolled_back")
            usage_count: Initial usage count
            unusage_count: Initial unusage count
            success_attribution: Initial success attribution count
            failure_attribution: Initial failure attribution count
        
        Returns:
            The entry_id of the added strategy
        """
        if entry_id in self.playbook.get("strategies", {}):
            self.logger.warning(f"Strategy entry {entry_id} already exists. Use edit_strategy to modify.")
            return entry_id
        
        token_count = self._count_tokens(content)
        
        entry = {
            "meta_info": {
                "token_count": token_count,
                "status": status,
                "usage_count": usage_count,
                "unusage_count": unusage_count,
                "success_attribution": success_attribution,
                "failure_attribution": failure_attribution,
            },
            "reflection": {
                "legacy_content": content,
                "legacy_type": entry_type,
            }
        }
        
        if "strategies" not in self.playbook:
            self.playbook["strategies"] = {}
        
        self.playbook["strategies"][entry_id] = entry
        self._touch_metadata()
        
        # WAL: Log the ADD operation
        self._write_wal({
            "op": "ADD",
            "id": entry_id,
            "parent_id": parent_id,
            "type": entry_type,
        })
        
        # Save current playbook
        self._save_current()
        self.logger.debug("Strategy added: %s", entry_id)
        return entry_id
    
    def add_feedback_entry(
        self,
        entry_id: str,
        reflection: Dict[str, Any],
        status: str = "active",
        usage_count: int = 0,
        unusage_count: int = 0,
        success_attribution: int = 0,
        failure_attribution: int = 0,
    ) -> str:
        """
        Add a feedback entry to the playbook from feedback generation results.
        
        Args:
            entry_id: Unique identifier for the entry (from feedback issue_id)
            reflection: Reflection dictionary containing feedback fields:
                - issue_type: "CODE_BUG|DESIGN_MISMATCH|EVAL_SIGNAL"
                - severity: "blocker|high|medium|low"
                - from_user_feedback: bool
                - blueprint_refs: list
                - code_refs: list
                - evidence: dict
                - error_identification: str
                - root_cause_analysis: str
                - correct_approach: str
                - key_insight: str
            status: Status of the entry ("active", "obsolete", or "rolled_back")
            usage_count: Initial usage count
            unusage_count: Initial unusage count
            success_attribution: Initial success attribution count
            failure_attribution: Initial failure attribution count
        
        Returns:
            The entry_id of the added entry
        """
        if entry_id in self.playbook.get("strategies", {}):
            self.logger.warning(f"Entry {entry_id} already exists. Use edit_strategy to modify.")
            return entry_id
        
        # Calculate token count from reflection text fields
        token_count = self._count_reflection_tokens(reflection)
        
        entry = {
            "meta_info": {
                "token_count": token_count,
                "status": status,
                "usage_count": usage_count,
                "unusage_count": unusage_count,
                "success_attribution": success_attribution,
                "failure_attribution": failure_attribution,
            },
            "reflection": reflection,
        }
        
        if "strategies" not in self.playbook:
            self.playbook["strategies"] = {}
        
        self.playbook["strategies"][entry_id] = entry
        self._touch_metadata()
        
        # WAL: Log the ADD operation
        self._write_wal({
            "op": "ADD",
            "id": entry_id,
        })
        
        # Save current playbook
        self._save_current()
        self.logger.debug("Feedback entry added: %s", entry_id)
        return entry_id
    
    def add_feedback_entries(
        self,
        feedback: Dict[str, Any],
        iteration: Optional[int] = None,
        similarity_threshold: float = 0.70,
        error_similarity_threshold: float = 0.65,
        insight_similarity_threshold: float = 0.70,
    ) -> Dict[str, Any]:
        """
        Add multiple feedback entries to the playbook with similarity-based merging.
        
        Events handled:
        - E_new: New entry added to playbook with status=open
        - E_merge: Entry merged with existing strategy, status set to open, counters preserved
        
        Process:
        1. For each feedback entry, find the most similar existing strategy
        2. If similarity meets merge criteria:
           a. Preserve the matched strategy's counters (usage_count, etc.)
           b. Merge key_insight and metric_links
           c. Delete the matched strategy
           d. Add the new entry with status=open and preserved counters
        3. If similarity does not meet criteria:
           - Add the new entry with status=open and zero counters
        
        Similarity calculation:
        - S_ref: Blueprint refs similarity (Jaccard on tokens + paths)
        - S_error: Error identification similarity (SBERT cosine)
        - S_root_cause: Root cause analysis similarity (SBERT cosine)
        - S_total = 0.2 * S_ref + 0.45 * S_error + 0.35 * S_root_cause
        
        Merge criteria: S_total >= similarity_threshold AND S_error >= error_similarity_threshold
        
        Args:
            feedback: Dictionary of feedback issues (keyed by issue_id)
            iteration: Optional iteration number for metadata update
            similarity_threshold: Threshold for total similarity (default: 0.70)
            error_similarity_threshold: Threshold for error similarity (default: 0.65)
            insight_similarity_threshold: Threshold for key_insight similarity (default: 0.70)
        
        Returns:
            Dictionary containing:
            - added_ids: List of entry_ids that were added
            - merged_ids: List of (entry_id, merged_from_strategy_id) tuples
            - new_ids: List of entry_ids that are completely new
        """
        result = {
            "added_ids": [],
            "merged_ids": [],
            "new_ids": [],
        }
        
        if not feedback or not isinstance(feedback, dict):
            self.logger.warning("Invalid feedback format, skipping playbook update")
            return result
        
        # Get all existing strategy IDs for matching (all statuses can be matched)
        strategies = self.playbook.get("strategies", {})
        candidate_ids = list(strategies.keys())
        self.logger.info(f"Found {len(candidate_ids)} existing strategies for similarity matching")
        
        for issue_id, issue_data in feedback.items():
            if not isinstance(issue_data, dict):
                self.logger.warning(f"Skipping invalid issue entry: {issue_id}")
                continue
            
            # Extract reflection fields from feedback
            reflection = {
                "issue_type": issue_data.get("issue_type", "CODE_BUG"),
                "severity": issue_data.get("severity", "medium"),
                "from_user_feedback": issue_data.get("from_user_feedback", False),
                "blueprint_refs": issue_data.get("blueprint_refs", []),
                "code_refs": issue_data.get("code_refs", []),
                "evidence": issue_data.get("evidence", {}),
                "error_identification": issue_data.get("error_identification", ""),
                "root_cause_analysis": issue_data.get("root_cause_analysis", ""),
                "correct_approach": issue_data.get("correct_approach", ""),
                "key_insight": issue_data.get("key_insight", ""),
                "metric_links": issue_data.get("metric_links", []),
            }
            
            # Find the most similar existing strategy
            best_match_id, S_total, S_error = self._find_best_matching_strategy(
                reflection, candidate_ids
            )
            
            # Determine if this is a merge (E_merge) or new entry (E_new)
            is_merge = (
                best_match_id is not None
                and S_total >= similarity_threshold
                and S_error >= error_similarity_threshold
            )
            
            # Initialize counters (will be overwritten if merge)
            usage_count = 0
            unusage_count = 0
            success_attribution = 0
            failure_attribution = 0
            
            if is_merge:
                # E_merge: Entry merged with existing strategy
                self.logger.info(
                    f"[E_merge] Entry '{issue_id}' merged with strategy '{best_match_id}' "
                    f"(S_total={S_total:.3f}, S_error={S_error:.3f})"
                )
                
                matched_strategy = strategies.get(best_match_id)
                
                if matched_strategy:
                    # Preserve counters from matched strategy
                    matched_meta = matched_strategy.get("meta_info", {})
                    usage_count = matched_meta.get("usage_count", 0)
                    unusage_count = matched_meta.get("unusage_count", 0)
                    success_attribution = matched_meta.get("success_attribution", 0)
                    failure_attribution = matched_meta.get("failure_attribution", 0)
                    
                    # Merge key_insight
                    entry_key_insight = reflection.get("key_insight", "") or ""
                    strategy_reflection = matched_strategy.get("reflection", {})
                    strategy_key_insight = strategy_reflection.get("key_insight", "") or ""
                    
                    key_insight_merged = False
                    key_insight_sim = None
                    
                    if entry_key_insight and strategy_key_insight:
                        key_insight_sim = self._compute_sbert_cosine_similarity(
                            entry_key_insight, strategy_key_insight
                        )
                        
                        if key_insight_sim < insight_similarity_threshold:
                            merged_key_insight = f"{strategy_key_insight} {entry_key_insight}"
                            reflection["key_insight"] = merged_key_insight
                            key_insight_merged = True
                            self.logger.debug(
                                f"Key insights merged (similarity {key_insight_sim:.3f} < {insight_similarity_threshold})"
                            )
                    elif strategy_key_insight and not entry_key_insight:
                        reflection["key_insight"] = strategy_key_insight
                        key_insight_merged = True
                    
                    # Merge metric_links
                    entry_metric_links = reflection.get("metric_links", [])
                    strategy_metric_links = strategy_reflection.get("metric_links", [])
                    
                    if strategy_metric_links and isinstance(strategy_metric_links, list):
                        merged_metric_links = {}
                        for link in strategy_metric_links:
                            if isinstance(link, dict) and "name" in link:
                                merged_metric_links[link["name"]] = link
                        if isinstance(entry_metric_links, list):
                            for link in entry_metric_links:
                                if isinstance(link, dict) and "name" in link:
                                    merged_metric_links[link["name"]] = link
                        if merged_metric_links:
                            reflection["metric_links"] = list(merged_metric_links.values())
                    
                    # Delete the matched strategy
                    del strategies[best_match_id]
                    self._write_wal({
                        "op": self.EVENT_MERGE,
                        "entry_id": issue_id,
                        "strategy_id": best_match_id,
                        "similarity": S_total,
                        "key_insight_similarity": key_insight_sim,
                        "key_insight_merged": key_insight_merged,
                        "preserved_counters": {
                            "usage_count": usage_count,
                            "unusage_count": unusage_count,
                            "success_attribution": success_attribution,
                            "failure_attribution": failure_attribution,
                        },
                    })
                
                # Remove from candidates to prevent matching again
                if best_match_id in candidate_ids:
                    candidate_ids.remove(best_match_id)
                
                result["merged_ids"].append((issue_id, best_match_id))
            else:
                # E_new: New entry added
                self.logger.info(f"[E_new] New entry '{issue_id}' added to playbook")
                self._write_wal({
                    "op": self.EVENT_NEW,
                    "entry_id": issue_id,
                })
                result["new_ids"].append(issue_id)
            
            # Add the entry with status=open
            try:
                token_count = self._count_reflection_tokens(reflection)
                
                entry = {
                    "meta_info": {
                        "token_count": token_count,
                        "status": self.STATUS_OPEN,  # All new entries start as open
                        "usage_count": usage_count,
                        "unusage_count": unusage_count,
                        "success_attribution": success_attribution,
                        "failure_attribution": failure_attribution,
                    },
                    "reflection": reflection,
                }
                
                if "strategies" not in self.playbook:
                    self.playbook["strategies"] = {}
                
                self.playbook["strategies"][issue_id] = entry
                result["added_ids"].append(issue_id)
                
            except Exception as e:
                self.logger.error(f"Error adding feedback entry {issue_id}: {e}")
        
        # Update metadata
        if iteration is not None:
            self._touch_metadata(iteration=iteration)
        else:
            self._touch_metadata()
        
        # Save current playbook to file
        self._save_current()
        
        # Reload playbook from file to keep memory and file in sync
        self.playbook = self._load_from_file(self.playbook_path)
        self.logger.debug("Playbook reloaded from file after saving to ensure synchronization")
        
        self.logger.info(
            f"Feedback entries processed: {len(result['added_ids'])} added "
            f"({len(result['merged_ids'])} merged, {len(result['new_ids'])} new)"
        )
        
        return result
    
    def select_strategies_for_prompt(
        self,
        token_budget: int = 3000,
        iteration: Optional[int] = None,
        lambda_decay: float = 0.05,
        k_queue: int = 10,
    ) -> List[str]:
        """
        Select strategies using Token-Aware Greedy Selection with Marginal Value (SBERT Similarity).
        
        This is a 0-1 knapsack-style algorithm that:
        1. Calculates value for each strategy based on user feedback, severity, urgency, and reliability
        2. Uses SBERT similarity to penalize redundant strategies
        3. Greedily selects strategies with highest marginal density (value/token_cost)
        4. Fills remaining budget with gap-filling pass
        
        Events handled:
        - E_selected: Strategy selected for prompt (open/queued -> in_progress, usage_count += 1)
        - not_selected: Strategy not selected (open -> queued, queued -> queued, unusage_count += 1)
        
        Args:
            token_budget: Maximum tokens for high-attention region (default: 3000)
            iteration: Current iteration number for logging
            lambda_decay: Decay factor for queue urgency (default: 0.05)
            k_queue: Maximum queue count for urgency calculation (default: 10)
        
        Returns:
            List of selected strategy IDs
        """
        strategies = self.playbook.get("strategies", {})
        
        # Step 0: Find all selectable strategies (open/queued)
        candidates = {}  # strategy_id -> candidate info
        for strategy_id, entry in strategies.items():
            meta_info = entry.get("meta_info", {})
            status = meta_info.get("status", self.STATUS_OPEN)
            
            if status not in [self.STATUS_OPEN, self.STATUS_QUEUED]:
                continue
            
            reflection = entry.get("reflection", {})
            
            # Build snippet: error_identification + correct_approach + key_insight
            snippet = self._build_strategy_snippet(reflection)
            
            # Calculate token cost
            token_cost = self._count_tokens(snippet)
            if token_cost == 0:
                token_cost = 1  # Minimum cost to avoid division by zero
            
            candidates[strategy_id] = {
                "entry": entry,
                "status": status,
                "snippet": snippet,
                "token_cost": token_cost,
                "reflection": reflection,
                "meta_info": meta_info,
            }
        
        if not candidates:
            self.logger.info("No selectable strategies (open/queued) found")
            return []
        
        self.logger.info(f"Found {len(candidates)} candidate strategies for selection")
        
        # Step 1: Calculate base value for each candidate
        for strategy_id, cand in candidates.items():
            cand["value"] = self._calculate_strategy_value(
                cand["reflection"], 
                cand["meta_info"], 
                cand["status"],
                lambda_decay=lambda_decay,
                k_queue=k_queue
            )
        
        # Step 2: Compute SBERT embeddings for similarity calculation
        embeddings = self._compute_snippet_embeddings(candidates)
        
        # Step 3: Greedy selection with marginal value
        selected_ids = []
        remaining_budget = token_budget
        
        # Track similarity to selected set for each candidate
        sim_to_selected = {sid: 0.0 for sid in candidates}
        
        while True:
            # Find best candidate that fits in budget
            best_id = None
            best_density = -1.0
            
            for strategy_id, cand in candidates.items():
                if strategy_id in selected_ids:
                    continue
                if cand["token_cost"] > remaining_budget:
                    continue
                
                # Calculate marginal value with similarity penalty
                base_value = cand["value"]
                sim_penalty = 1.0 - sim_to_selected[strategy_id]
                marginal_value = base_value * sim_penalty
                
                # Calculate density (marginal value per token)
                density = marginal_value / cand["token_cost"]
                
                if density > best_density:
                    best_density = density
                    best_id = strategy_id
            
            if best_id is None:
                # No more candidates fit in budget
                break
            
            # Select this candidate
            selected_ids.append(best_id)
            remaining_budget -= candidates[best_id]["token_cost"]
            
            # Update similarity to selected set for all remaining candidates
            best_embedding = embeddings.get(best_id)
            if best_embedding is not None:
                for strategy_id in candidates:
                    if strategy_id in selected_ids:
                        continue
                    other_embedding = embeddings.get(strategy_id)
                    if other_embedding is not None:
                        sim = self._cosine_similarity(best_embedding, other_embedding)
                        sim_to_selected[strategy_id] = max(sim_to_selected[strategy_id], sim)
            
            self.logger.debug(
                f"Selected '{best_id}' (density={best_density:.3f}, "
                f"tokens={candidates[best_id]['token_cost']}, remaining={remaining_budget})"
            )
        
        # Step 4: Gap-filling - try to fill remaining budget with small items
        if remaining_budget > 0:
            self.logger.debug(f"Gap-filling with {remaining_budget} remaining tokens")
            
            # Recalculate marginal values with updated similarities
            gap_candidates = []
            for strategy_id, cand in candidates.items():
                if strategy_id in selected_ids:
                    continue
                if cand["token_cost"] > remaining_budget:
                    continue
                
                base_value = cand["value"]
                sim_penalty = 1.0 - sim_to_selected[strategy_id]
                marginal_value = base_value * sim_penalty
                density = marginal_value / cand["token_cost"]
                
                gap_candidates.append((strategy_id, density, cand["token_cost"]))
            
            # Sort by density and fill greedily
            gap_candidates.sort(key=lambda x: x[1], reverse=True)
            
            for strategy_id, density, token_cost in gap_candidates:
                if token_cost <= remaining_budget:
                    selected_ids.append(strategy_id)
                    remaining_budget -= token_cost
                    self.logger.debug(f"Gap-filled '{strategy_id}' (tokens={token_cost})")
        
        # Step 5: Apply state transitions
        not_selected_ids = []
        
        for strategy_id, cand in candidates.items():
            entry = cand["entry"]
            meta_info = cand["meta_info"]
            old_status = cand["status"]
            
            if strategy_id in selected_ids:
                # E_selected
                meta_info["status"] = self.STATUS_IN_PROGRESS
                meta_info["usage_count"] = meta_info.get("usage_count", 0) + 1
                
                self._write_wal({
                    "op": self.EVENT_SELECTED,
                    "strategy_id": strategy_id,
                    "old_status": old_status,
                    "new_status": self.STATUS_IN_PROGRESS,
                    "iteration": iteration,
                    "token_cost": cand["token_cost"],
                    "value": cand["value"],
                })
                
                self.logger.debug(f"[E_selected] Strategy '{strategy_id}': {old_status} -> in_progress")
            else:
                # not_selected
                if old_status == self.STATUS_OPEN:
                    meta_info["status"] = self.STATUS_QUEUED
                # queued stays queued
                
                meta_info["unusage_count"] = meta_info.get("unusage_count", 0) + 1
                
                self._write_wal({
                    "op": self.EVENT_NOT_SELECTED,
                    "strategy_id": strategy_id,
                    "old_status": old_status,
                    "new_status": meta_info["status"],
                    "iteration": iteration,
                })
                
                not_selected_ids.append(strategy_id)
                self.logger.debug(f"[not_selected] Strategy '{strategy_id}': {old_status} -> {meta_info['status']}")
        
        # Save changes and reload to keep memory and file in sync
        self._save_current()
        self.playbook = self._load_from_file(self.playbook_path)
        self.logger.debug("Playbook reloaded from file after saving to ensure synchronization")
        
        total_tokens_used = token_budget - remaining_budget
        self.logger.info(
            f"Strategy selection complete: {len(selected_ids)} selected "
            f"({total_tokens_used}/{token_budget} tokens used), "
            f"{len(not_selected_ids)} not selected"
        )
        
        return selected_ids
    
    def _build_strategy_snippet(self, reflection: Dict[str, Any]) -> str:
        """
        Build the snippet for a strategy: error_identification + correct_approach + key_insight.
        """
        parts = []
        
        error_id = reflection.get("error_identification", "")
        if error_id:
            parts.append(str(error_id))
        
        correct_approach = reflection.get("correct_approach", "")
        if correct_approach:
            parts.append(str(correct_approach))
        
        key_insight = reflection.get("key_insight", "")
        if key_insight:
            parts.append(str(key_insight))
        
        return " ".join(parts)
    
    def _count_tokens(self, text: str) -> int:
        """
        Count tokens in text. Uses simple whitespace-based estimation.
        For more accurate counting, could use tiktoken or similar.
        """
        if not text:
            return 0
        # Simple estimation: ~4 characters per token on average
        return max(1, len(text) // 4)
    
    def _calculate_strategy_value(
        self,
        reflection: Dict[str, Any],
        meta_info: Dict[str, Any],
        status: str,
        lambda_decay: float = 0.05,
        k_queue: int = 10,
    ) -> float:
        """
        Calculate the value of a strategy for selection.
        
        Value = W_user * W_sev * U(t) * Φ(t)
        
        Where:
        - W_user: 1.0 if from_user_feedback else 0.6
        - W_sev: blocker=1.0, high=0.8, medium=0.6, low=0.4
        - U(t) = U_status * U_queue
          - U_status: open=1.0, queued=0.8
          - U_queue = 1 + λ * min(unusage_count, K_q)
        - Φ(t) = 0.2 + 1.0 * (s+1)/(s+f+2) (Beta-Binomial posterior mean)
        """
        # W_user: user feedback weight
        from_user = reflection.get("from_user_feedback", False)
        w_user = 1.0 if from_user else 0.6
        
        # W_sev: severity weight
        severity = reflection.get("severity", "medium").lower()
        severity_weights = {
            "blocker": 1.0,
            "high": 0.8,
            "medium": 0.6,
            "low": 0.4
        }
        w_sev = severity_weights.get(severity, 0.6)
        
        # U(t): urgency
        # U_status
        u_status = 1.0 if status == self.STATUS_OPEN else 0.8
        
        # U_queue = 1 + λ * min(unusage_count, K_q)
        unusage_count = meta_info.get("unusage_count", 0)
        u_queue = 1.0 + lambda_decay * min(unusage_count, k_queue)
        
        u_total = u_status * u_queue
        
        # Φ(t): reliability (Beta-Binomial posterior mean)
        s = meta_info.get("success_attribution", 0)
        f = meta_info.get("failure_attribution", 0)
        p = (s + 1) / (s + f + 2)  # Beta posterior mean with uniform prior
        phi = 0.2 + 1.0 * p
        
        # Final value
        value = w_user * w_sev * u_total * phi
        
        return value
    
    def _compute_snippet_embeddings(
        self, 
        candidates: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Compute SBERT embeddings for all candidate snippets.
        
        Returns:
            Dictionary mapping strategy_id to embedding vector (numpy array)
        """
        embeddings = {}
        
        try:
            model = _get_sbert_model()
            
            # Collect all snippets
            snippets = []
            ids = []
            for strategy_id, cand in candidates.items():
                snippet = cand.get("snippet", "")
                if snippet:
                    snippets.append(snippet)
                    ids.append(strategy_id)
            
            if snippets:
                # Batch encode all snippets
                vectors = model.encode(snippets, convert_to_numpy=True)
                
                for i, strategy_id in enumerate(ids):
                    embeddings[strategy_id] = vectors[i]
                
                self.logger.debug(f"Computed SBERT embeddings for {len(embeddings)} strategies")
        
        except Exception as e:
            self.logger.warning(f"Failed to compute SBERT embeddings: {e}")
            # Return empty embeddings - similarity will be 0
        
        return embeddings
    
    def _cosine_similarity(self, vec1, vec2) -> float:
        """
        Compute cosine similarity between two vectors.
        """
        try:
            # Use numpy for efficiency
            dot = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return float(dot / (norm1 * norm2))
        except Exception:
            return 0.0
    
    def select_strategies_for_prompt_simple(
        self,
        max_count: Optional[int] = None,
        iteration: Optional[int] = None,
    ) -> List[str]:
        """
        Simple strategy selection: select all open/queued strategies.
        
        This is the original simple selection method, kept for backward compatibility
        and debugging purposes.
        
        Events handled:
        - E_selected: Strategy selected for prompt (open/queued -> in_progress, usage_count += 1)
        - not_selected: Strategy not selected (open -> queued, queued -> queued, unusage_count += 1)
        
        State transitions:
        - open + E_selected -> in_progress
        - open + not_selected -> queued
        - queued + E_selected -> in_progress
        - queued + not_selected -> queued (no change)
        
        Selection strategy:
        - Selects ALL open and queued strategies (no budget limit by default)
        - If max_count is provided, prioritize by severity (blocker > high > medium > low)
        
        Args:
            max_count: Maximum number of strategies to select (None = all)
            iteration: Current iteration number for logging
        
        Returns:
            List of selected strategy IDs
        """
        strategies = self.playbook.get("strategies", {})
        
        # Find all open and queued strategies
        selectable = []
        for strategy_id, entry in strategies.items():
            meta_info = entry.get("meta_info", {})
            status = meta_info.get("status", self.STATUS_OPEN)
            
            if status in [self.STATUS_OPEN, self.STATUS_QUEUED]:
                severity = entry.get("reflection", {}).get("severity", "medium")
                selectable.append((strategy_id, status, severity))
        
        if not selectable:
            self.logger.info("No selectable strategies (open/queued) found")
            return []
        
        # Sort by severity priority (blocker > high > medium > low)
        severity_order = {"blocker": 0, "high": 1, "medium": 2, "low": 3}
        selectable.sort(key=lambda x: severity_order.get(x[2], 2))
        
        # Determine which to select
        if max_count is not None and max_count < len(selectable):
            selected = selectable[:max_count]
            not_selected = selectable[max_count:]
        else:
            selected = selectable
            not_selected = []
        
        selected_ids = []
        
        # Process selected strategies (E_selected)
        for strategy_id, old_status, severity in selected:
            entry = strategies[strategy_id]
            meta_info = entry.get("meta_info", {})
            
            # Transition to in_progress
            meta_info["status"] = self.STATUS_IN_PROGRESS
            meta_info["usage_count"] = meta_info.get("usage_count", 0) + 1
            
            self._write_wal({
                "op": self.EVENT_SELECTED,
                "strategy_id": strategy_id,
                "old_status": old_status,
                "new_status": self.STATUS_IN_PROGRESS,
                "iteration": iteration,
            })
            
            selected_ids.append(strategy_id)
            self.logger.debug(f"[E_selected] Strategy '{strategy_id}': {old_status} -> in_progress")
        
        # Process not selected strategies (not_selected)
        for strategy_id, old_status, severity in not_selected:
            entry = strategies[strategy_id]
            meta_info = entry.get("meta_info", {})
            
            # Open -> queued, queued stays queued
            if old_status == self.STATUS_OPEN:
                meta_info["status"] = self.STATUS_QUEUED
            # queued stays queued (no status change)
            
            meta_info["unusage_count"] = meta_info.get("unusage_count", 0) + 1
            
            self._write_wal({
                "op": self.EVENT_NOT_SELECTED,
                "strategy_id": strategy_id,
                "old_status": old_status,
                "new_status": meta_info["status"],
                "iteration": iteration,
            })
            
            self.logger.debug(f"[not_selected] Strategy '{strategy_id}': {old_status} -> {meta_info['status']}")
        
        # Save changes and reload to keep memory and file in sync
        self._save_current()
        self.playbook = self._load_from_file(self.playbook_path)
        self.logger.debug("Playbook reloaded from file after saving to ensure synchronization")
        
        self.logger.info(
            f"Strategy selection complete: {len(selected_ids)} selected for prompt, "
            f"{len(not_selected)} not selected (queued)"
        )
        
        return selected_ids
    
    def evaluate_in_progress_strategies(
        self,
        current_metrics: Dict[str, float],
        previous_metrics: Optional[Dict[str, float]] = None,
        iteration: Optional[int] = None,
        improvement_threshold: float = 0.03,
    ) -> Dict[str, List[str]]:
        """
        Evaluate in_progress strategies based on metric changes.
        
        This method should be called AFTER simulation execution and BEFORE add_feedback_entries
        to evaluate whether in_progress strategies were successful.
        
        Events handled:
        - E_resolved: Strategy resolved (in_progress -> resolved, success_attribution += 1)
        - E_persist: Strategy persists (in_progress -> open, failure_attribution += 1)
        - E_uncertain: Strategy uncertain (in_progress -> open, no counter change)
        
        Resolution logic for each in_progress strategy:
        1. If strategy has no metric_links: Check if it was merged in current feedback
           - If NOT merged -> E_resolved (issue disappeared)
           - If merged -> handled by add_feedback_entries (E_merge)
        2. If strategy has metric_links: Check metric changes
           - If majority improved by > threshold -> E_resolved
           - If majority worsened by > threshold -> E_persist
           - Otherwise -> E_uncertain
        
        Args:
            current_metrics: Current iteration's metrics (key -> value)
            previous_metrics: Previous iteration's metrics (key -> value), None if first iteration
            iteration: Current iteration number for logging
            improvement_threshold: Threshold for considering a metric improved/worsened (default: 3%)
        
        Returns:
            Dictionary containing:
            - resolved_ids: List of strategy IDs that were resolved
            - persist_ids: List of strategy IDs that persist
            - uncertain_ids: List of strategy IDs that are uncertain
        """
        result = {
            "resolved_ids": [],
            "persist_ids": [],
            "uncertain_ids": [],
        }
        
        if previous_metrics is None:
            self.logger.info("No previous metrics available, skipping in_progress evaluation")
            return result
        
        strategies = self.playbook.get("strategies", {})
        
        # Find all in_progress strategies
        in_progress_ids = []
        for strategy_id, entry in strategies.items():
            meta_info = entry.get("meta_info", {})
            status = meta_info.get("status")
            if status == self.STATUS_IN_PROGRESS:
                in_progress_ids.append(strategy_id)
        
        if not in_progress_ids:
            self.logger.info("No in_progress strategies to evaluate")
            return result
        
        self.logger.info(f"Evaluating {len(in_progress_ids)} in_progress strategies")
        
        for strategy_id in in_progress_ids:
            entry = strategies[strategy_id]
            meta_info = entry.get("meta_info", {})
            reflection = entry.get("reflection", {})
            metric_links = reflection.get("metric_links", [])
            
            if not metric_links:
                # Strategy has no metric links - will be evaluated by merge check later
                # For now, mark as uncertain (will be resolved if not merged)
                self.logger.debug(
                    f"Strategy '{strategy_id}' has no metric_links, "
                    f"will be resolved if not merged in current feedback"
                )
                # We'll handle this after add_feedback_entries
                continue
            
            # Evaluate metric changes
            improved_count = 0
            worsened_count = 0
            uncertain_count = 0
            total_weight = 0.0
            
            for link in metric_links:
                if not isinstance(link, dict):
                    continue
                
                metric_name = link.get("name")
                direction = link.get("direction", "lower_is_better")
                weight = float(link.get("weight", 1.0))
                
                if metric_name not in current_metrics:
                    self.logger.debug(f"Metric '{metric_name}' not found in current metrics")
                    uncertain_count += 1
                    continue
                
                if metric_name not in previous_metrics:
                    self.logger.debug(f"Metric '{metric_name}' not found in previous metrics")
                    uncertain_count += 1
                    continue
                
                current_val = current_metrics[metric_name]
                previous_val = previous_metrics[metric_name]
                
                # Handle zero previous value
                if previous_val == 0:
                    if current_val == 0:
                        uncertain_count += 1
                    elif direction == "lower_is_better":
                        worsened_count += 1 if current_val > 0 else improved_count + 1
                    else:
                        improved_count += 1 if current_val > 0 else worsened_count + 1
                    continue
                
                # Calculate relative change
                relative_change = (current_val - previous_val) / abs(previous_val)
                
                # Determine if improved or worsened based on direction
                if direction == "lower_is_better":
                    # Lower is better: negative change is improvement
                    if relative_change < -improvement_threshold:
                        improved_count += 1
                    elif relative_change > improvement_threshold:
                        worsened_count += 1
                    else:
                        uncertain_count += 1
                else:  # higher_is_better
                    # Higher is better: positive change is improvement
                    if relative_change > improvement_threshold:
                        improved_count += 1
                    elif relative_change < -improvement_threshold:
                        worsened_count += 1
                    else:
                        uncertain_count += 1
                
                total_weight += weight
            
            total_links = improved_count + worsened_count + uncertain_count
            
            # Determine event type based on majority
            if total_links == 0:
                event = self.EVENT_UNCERTAIN
            elif improved_count > worsened_count and improved_count > uncertain_count:
                event = self.EVENT_RESOLVED
            elif worsened_count > improved_count and worsened_count > uncertain_count:
                event = self.EVENT_PERSIST
            else:
                event = self.EVENT_UNCERTAIN
            
            # Apply state transition and update counters
            if event == self.EVENT_RESOLVED:
                meta_info["status"] = self.STATUS_RESOLVED
                meta_info["success_attribution"] = meta_info.get("success_attribution", 0) + 1
                result["resolved_ids"].append(strategy_id)
                self.logger.info(
                    f"[E_resolved] Strategy '{strategy_id}': in_progress -> resolved "
                    f"(improved: {improved_count}, worsened: {worsened_count}, uncertain: {uncertain_count})"
                )
            elif event == self.EVENT_PERSIST:
                meta_info["status"] = self.STATUS_OPEN
                meta_info["failure_attribution"] = meta_info.get("failure_attribution", 0) + 1
                result["persist_ids"].append(strategy_id)
                self.logger.info(
                    f"[E_persist] Strategy '{strategy_id}': in_progress -> open "
                    f"(improved: {improved_count}, worsened: {worsened_count}, uncertain: {uncertain_count})"
                )
            else:  # E_uncertain
                meta_info["status"] = self.STATUS_OPEN
                # No counter change for uncertain
                result["uncertain_ids"].append(strategy_id)
                self.logger.info(
                    f"[E_uncertain] Strategy '{strategy_id}': in_progress -> open "
                    f"(improved: {improved_count}, worsened: {worsened_count}, uncertain: {uncertain_count})"
                )
            
            self._write_wal({
                "op": event,
                "strategy_id": strategy_id,
                "old_status": self.STATUS_IN_PROGRESS,
                "new_status": meta_info["status"],
                "metric_analysis": {
                    "improved": improved_count,
                    "worsened": worsened_count,
                    "uncertain": uncertain_count,
                },
                "iteration": iteration,
            })
        
        # Save changes and reload to keep memory and file in sync
        self._save_current()
        self.playbook = self._load_from_file(self.playbook_path)
        self.logger.debug("Playbook reloaded from file after saving to ensure synchronization")
        
        self.logger.info(
            f"In-progress evaluation complete: {len(result['resolved_ids'])} resolved, "
            f"{len(result['persist_ids'])} persist, {len(result['uncertain_ids'])} uncertain"
        )
        
        return result
    
    def resolve_non_merged_in_progress(
        self,
        merged_strategy_ids: List[str],
        iteration: Optional[int] = None,
    ) -> List[str]:
        """
        Resolve in_progress strategies that were not merged in the current feedback.
        
        This method should be called AFTER add_feedback_entries to handle strategies
        that have no metric_links and were not merged (i.e., the issue disappeared).
        
        For in_progress strategies without metric_links:
        - If NOT merged -> E_resolved (issue no longer appears in feedback)
        - If merged -> already handled by add_feedback_entries
        
        Args:
            merged_strategy_ids: List of strategy IDs that were merged in current feedback
            iteration: Current iteration number for logging
        
        Returns:
            List of strategy IDs that were resolved
        """
        strategies = self.playbook.get("strategies", {})
        resolved_ids = []
        
        for strategy_id, entry in strategies.items():
            meta_info = entry.get("meta_info", {})
            status = meta_info.get("status")
            
            if status != self.STATUS_IN_PROGRESS:
                continue
            
            reflection = entry.get("reflection", {})
            metric_links = reflection.get("metric_links", [])
            
            # Only process strategies without metric_links
            if metric_links:
                continue
            
            # Check if this strategy was merged
            if strategy_id in merged_strategy_ids:
                # Already handled by add_feedback_entries (E_merge)
                continue
            
            # Strategy was in_progress, has no metric_links, and was not merged
            # -> E_resolved (issue disappeared)
            meta_info["status"] = self.STATUS_RESOLVED
            meta_info["success_attribution"] = meta_info.get("success_attribution", 0) + 1
            
            self._write_wal({
                "op": self.EVENT_RESOLVED,
                "strategy_id": strategy_id,
                "old_status": self.STATUS_IN_PROGRESS,
                "new_status": self.STATUS_RESOLVED,
                "reason": "no_metric_links_and_not_merged",
                "iteration": iteration,
            })
            
            resolved_ids.append(strategy_id)
            self.logger.info(
                f"[E_resolved] Strategy '{strategy_id}': in_progress -> resolved "
                f"(no metric_links and not merged in current feedback)"
            )
        
        if resolved_ids:
            # Save changes and reload to keep memory and file in sync
            self._save_current()
            self.playbook = self._load_from_file(self.playbook_path)
            self.logger.debug("Playbook reloaded from file after saving to ensure synchronization")
            self.logger.info(f"Resolved {len(resolved_ids)} in_progress strategies without metric_links")
        
        return resolved_ids
    
    def resolve_non_merged_in_progress_simple(
        self,
        merged_strategy_ids: List[str],
        iteration: Optional[int] = None,
    ) -> List[str]:
        """
        Simple version: Resolve ALL in_progress strategies that were not merged.
        
        This is a simplified version that does NOT check metric_links.
        It resolves any in_progress strategy that was not merged in the current feedback,
        assuming that if the issue didn't reappear in feedback, it was resolved.
        
        This method should be called AFTER add_feedback_entries.
        
        Logic:
        - If NOT merged -> E_resolved (issue no longer appears in feedback)
        - If merged -> already handled by add_feedback_entries (status reset to open)
        
        Args:
            merged_strategy_ids: List of strategy IDs that were merged in current feedback
            iteration: Current iteration number for logging
        
        Returns:
            List of strategy IDs that were resolved
        """
        strategies = self.playbook.get("strategies", {})
        resolved_ids = []
        
        for strategy_id, entry in strategies.items():
            meta_info = entry.get("meta_info", {})
            status = meta_info.get("status")
            
            if status != self.STATUS_IN_PROGRESS:
                continue
            
            # Check if this strategy was merged
            if strategy_id in merged_strategy_ids:
                # Already handled by add_feedback_entries (E_merge -> status reset to open)
                continue
            
            # Strategy was in_progress and was not merged
            # -> E_resolved (issue disappeared from feedback)
            meta_info["status"] = self.STATUS_RESOLVED
            meta_info["success_attribution"] = meta_info.get("success_attribution", 0) + 1
            
            self._write_wal({
                "op": self.EVENT_RESOLVED,
                "strategy_id": strategy_id,
                "old_status": self.STATUS_IN_PROGRESS,
                "new_status": self.STATUS_RESOLVED,
                "reason": "not_merged_in_current_feedback",
                "iteration": iteration,
            })
            
            resolved_ids.append(strategy_id)
            self.logger.info(
                f"[E_resolved] Strategy '{strategy_id}': in_progress -> resolved "
                f"(not merged in current feedback)"
            )
        
        if resolved_ids:
            # Save changes and reload to keep memory and file in sync
            self._save_current()
            self.playbook = self._load_from_file(self.playbook_path)
            self.logger.debug("Playbook reloaded from file after saving to ensure synchronization")
            self.logger.info(f"Resolved {len(resolved_ids)} in_progress strategies (not merged)")
        
        return resolved_ids
    
    def edit_strategy(
        self,
        entry_id: str,
        content: Optional[str] = None,
        increment_usage_count: bool = False,
        increment_unusage_count: bool = False,
        increment_success_attribution: bool = False,
        increment_failure_attribution: bool = False,
    ) -> bool:
        """
        Edit a strategy entry in the playbook.
        
        Args:
            entry_id: Unique identifier for the strategy entry
            content: New content (if provided, will update and recalculate token_count)
            increment_usage_count: If True, increment usage_count by 1
            increment_unusage_count: If True, increment unusage_count by 1
            increment_success_attribution: If True, increment success_attribution by 1
            increment_failure_attribution: If True, increment failure_attribution by 1
        
        Returns:
            True if entry was found and updated, False otherwise
        """
        strategies = self.playbook.get("strategies", {})
        if entry_id not in strategies:
            self.logger.warning(f"Strategy entry {entry_id} not found.")
            return False
        
        entry = strategies[entry_id]
        wal_ops = []
        
        # Ensure entry has the new structure
        if "meta_info" not in entry:
            # Migrate old structure to new structure
            entry = {
                "meta_info": {
                    "token_count": entry.get("token_count", 0),
                    "status": entry.get("status", "active"),
                    "usage_count": entry.get("usage_count", 0),
                    "unusage_count": entry.get("unusage_count", 0),
                    "success_attribution": entry.get("success_attribution", 0),
                    "failure_attribution": entry.get("failure_attribution", 0),
                },
                "reflection": entry.get("reflection", {})
            }
            strategies[entry_id] = entry
        
        meta_info = entry["meta_info"]
        
        # Update content if provided (legacy support)
        if content is not None:
            if "legacy_content" not in entry.get("reflection", {}):
                if "reflection" not in entry:
                    entry["reflection"] = {}
                entry["reflection"]["legacy_content"] = content
            meta_info["token_count"] = self._count_tokens(content)
            wal_ops.append({
                "op": "EDIT_CONTENT",
                "id": entry_id,
            })
        
        # Increment counters if requested
        if increment_usage_count:
            meta_info["usage_count"] = meta_info.get("usage_count", 0) + 1
            wal_ops.append({
                "op": "EDIT_USAGE_COUNT",
                "id": entry_id,
                "delta_usage_count": 1,
            })
        if increment_unusage_count:
            meta_info["unusage_count"] = meta_info.get("unusage_count", 0) + 1
            wal_ops.append({
                "op": "EDIT_UNUSAGE_COUNT",
                "id": entry_id,
                "delta_unusage_count": 1,
            })
        if increment_success_attribution:
            meta_info["success_attribution"] = meta_info.get("success_attribution", 0) + 1
            wal_ops.append({
                "op": "EDIT_SUCCESS_ATTRIBUTION",
                "id": entry_id,
                "delta_success_attribution": 1,
            })
        if increment_failure_attribution:
            meta_info["failure_attribution"] = meta_info.get("failure_attribution", 0) + 1
            wal_ops.append({
                "op": "EDIT_FAILURE_ATTRIBUTION",
                "id": entry_id,
                "delta_failure_attribution": 1,
            })
        
        self._touch_metadata()
        
        # WAL: Log all operations
        for wal_op in wal_ops:
            self._write_wal(wal_op)
        
        # Save current playbook
        self._save_current()
        self.logger.debug("Strategy edited: %s", entry_id)
        return True
    
    def delete_strategy(self, entry_id: str) -> bool:
        """
        Delete a strategy entry by setting its status to "obsolete".
        
        Args:
            entry_id: Unique identifier for the strategy entry
        
        Returns:
            True if entry was found and updated, False otherwise
        """
        strategies = self.playbook.get("strategies", {})
        if entry_id not in strategies:
            self.logger.warning(f"Strategy entry {entry_id} not found.")
            return False
        
        entry = strategies[entry_id]
        
        # Ensure entry has the new structure
        if "meta_info" not in entry:
            # Migrate old structure to new structure
            entry = {
                "meta_info": {
                    "token_count": entry.get("token_count", 0),
                    "status": entry.get("status", "active"),
                    "usage_count": entry.get("usage_count", 0),
                    "unusage_count": entry.get("unusage_count", 0),
                    "success_attribution": entry.get("success_attribution", 0),
                    "failure_attribution": entry.get("failure_attribution", 0),
                },
                "reflection": entry.get("reflection", {})
            }
            strategies[entry_id] = entry
        
        entry["meta_info"]["status"] = "obsolete"
        self._touch_metadata()
        
        # WAL: Log the DELETE operation
        self._write_wal({
            "op": "DELETE",
            "id": entry_id,
        })
        
        self._save_current()
        self.logger.debug("Strategy deleted (status set to obsolete): %s", entry_id)
        return True
    
    def rollback_strategy(self, entry_id: str) -> bool:
        """
        Rollback a strategy entry by setting its status to "rolled_back".
        
        Args:
            entry_id: Unique identifier for the strategy entry
        
        Returns:
            True if entry was found and updated, False otherwise
        """
        strategies = self.playbook.get("strategies", {})
        if entry_id not in strategies:
            self.logger.warning(f"Strategy entry {entry_id} not found.")
            return False
        
        entry = strategies[entry_id]
        
        # Ensure entry has the new structure
        if "meta_info" not in entry:
            # Migrate old structure to new structure
            entry = {
                "meta_info": {
                    "token_count": entry.get("token_count", 0),
                    "status": entry.get("status", "active"),
                    "usage_count": entry.get("usage_count", 0),
                    "unusage_count": entry.get("unusage_count", 0),
                    "success_attribution": entry.get("success_attribution", 0),
                    "failure_attribution": entry.get("failure_attribution", 0),
                },
                "reflection": entry.get("reflection", {})
            }
            strategies[entry_id] = entry
        
        entry["meta_info"]["status"] = "rolled_back"
        self._touch_metadata()
        
        # WAL: Log the ROLLBACK operation
        self._write_wal({
            "op": "ROLLBACK",
            "id": entry_id,
        })
        
        self._save_current()
        self.logger.debug("Strategy rolled back: %s", entry_id)
        return True
    
    def get_strategy(self, entry_id: str) -> Optional[Dict[str, Any]]:
        """Get a strategy entry by ID."""
        strategies = self.playbook.get("strategies", {})
        return strategies.get(entry_id)
    
    def list_strategies(self, status_filter: Optional[str] = None) -> List[str]:
        """List all strategy entry IDs, optionally filtered by status."""
        strategies = self.playbook.get("strategies", {})
        if status_filter:
            return [
                entry_id
                for entry_id, entry in strategies.items()
                if entry.get("meta_info", {}).get("status") == status_filter
                or entry.get("status") == status_filter  # Backward compatibility
            ]
        return list(strategies.keys())

    # ------------------------------------------------------------------ #
    # Similarity Calculation Methods
    # ------------------------------------------------------------------ #
    @staticmethod
    def _tokenize_blueprint_refs(blueprint_refs: List[str]) -> Set[str]:
        """
        Tokenize blueprint_refs for similarity calculation.
        
        Process:
        1. Convert to lowercase
        2. Remove quotes and punctuation
        3. Split by . _ - , :
        
        Args:
            blueprint_refs: List of blueprint reference strings
            
        Returns:
            Set of tokens
        """
        if not blueprint_refs:
            return set()
        
        tokens = set()
        for ref in blueprint_refs:
            if not ref:
                continue
            # Convert to lowercase
            ref_lower = ref.lower()
            # Remove quotes and common punctuation (keep . _ - , : for splitting)
            ref_clean = re.sub(r'["\'\(\)\[\]\{\}]', '', ref_lower)
            # Split by delimiters: . _ - , : and whitespace
            parts = re.split(r'[._\-,:\s]+', ref_clean)
            # Add non-empty tokens
            tokens.update(p.strip() for p in parts if p.strip())
        
        return tokens
    
    @staticmethod
    def _jaccard_similarity(set1: Set[str], set2: Set[str]) -> float:
        """
        Calculate Jaccard similarity between two sets.
        
        J(A, B) = |A ∩ B| / |A ∪ B|
        
        Args:
            set1: First set
            set2: Second set
            
        Returns:
            Jaccard similarity score (0.0 to 1.0)
        """
        if not set1 and not set2:
            return 1.0  # Both empty means identical
        if not set1 or not set2:
            return 0.0  # One empty, one not
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0
    
    @staticmethod
    def _normalize_text(text: str) -> str:
        """
        Normalize text for embedding: lowercase and remove stopwords.
        
        Args:
            text: Input text
            
        Returns:
            Normalized text
        """
        if not text:
            return ""
        
        # Convert to lowercase
        text_lower = text.lower()
        
        # Tokenize (simple word splitting)
        words = re.findall(r'\b\w+\b', text_lower)
        
        # Remove stopwords
        filtered_words = [w for w in words if w not in STOPWORDS]
        
        return ' '.join(filtered_words)
    
    def _compute_sbert_cosine_similarity(self, text1: str, text2: str) -> float:
        """
        Compute cosine similarity between two texts using SBERT embeddings.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Cosine similarity score (0.0 to 1.0)
        """
        # Normalize texts
        norm_text1 = self._normalize_text(text1)
        norm_text2 = self._normalize_text(text2)
        
        # Handle empty texts
        if not norm_text1 and not norm_text2:
            return 1.0  # Both empty means identical
        if not norm_text1 or not norm_text2:
            return 0.0  # One empty, one not
        
        try:
            model = _get_sbert_model()
            
            # Get embeddings
            embeddings = model.encode([norm_text1, norm_text2], convert_to_numpy=True)
            
            # Compute cosine similarity
            vec1, vec2 = embeddings[0], embeddings[1]
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            cosine_sim = dot_product / (norm1 * norm2)
            
            # Clamp to [0, 1] (cosine can be negative for very different texts)
            return max(0.0, min(1.0, float(cosine_sim)))
            
        except Exception as e:
            self.logger.warning(f"SBERT similarity calculation failed: {e}. Returning 0.0")
            return 0.0
    
    def _compute_entry_similarity(
        self,
        entry_reflection: Dict[str, Any],
        strategy_reflection: Dict[str, Any],
    ) -> Tuple[float, float, float, float]:
        """
        Compute similarity between a feedback entry and a playbook strategy.
        
        Similarity is computed from three components:
        A) blueprint_refs similarity (S_ref)
        B) error_identification similarity (S_error)
        C) root_cause_analysis similarity (S_root_cause)
        
        Final score: S = 0.2 * S_ref + 0.45 * S_error + 0.35 * S_root_cause
        
        Args:
            entry_reflection: Reflection dict from feedback entry
            strategy_reflection: Reflection dict from playbook strategy
            
        Returns:
            Tuple of (S_total, S_error, S_ref, S_root_cause)
        """
        # A) Blueprint refs similarity
        entry_refs = entry_reflection.get("blueprint_refs", []) or []
        strategy_refs = strategy_reflection.get("blueprint_refs", []) or []
        
        # Token-level Jaccard
        entry_tokens = self._tokenize_blueprint_refs(entry_refs)
        strategy_tokens = self._tokenize_blueprint_refs(strategy_refs)
        sim_ref_tok = self._jaccard_similarity(entry_tokens, strategy_tokens)
        
        # Path-level Jaccard (exact string match)
        entry_paths = set(entry_refs) if entry_refs else set()
        strategy_paths = set(strategy_refs) if strategy_refs else set()
        sim_ref_path = self._jaccard_similarity(entry_paths, strategy_paths)
        
        # Combined blueprint similarity
        S_ref = 0.7 * sim_ref_path + 0.3 * sim_ref_tok
        
        # B) Error identification similarity
        entry_error = entry_reflection.get("error_identification", "") or ""
        strategy_error = strategy_reflection.get("error_identification", "") or ""
        S_error = self._compute_sbert_cosine_similarity(entry_error, strategy_error)
        
        # C) Root cause analysis similarity
        entry_root = entry_reflection.get("root_cause_analysis", "") or ""
        strategy_root = strategy_reflection.get("root_cause_analysis", "") or ""
        S_root_cause = self._compute_sbert_cosine_similarity(entry_root, strategy_root)
        
        # Final weighted score
        # Weights: error_identification (0.45) > root_cause (0.35) > blueprint (0.2)
        S_total = 0.2 * S_ref + 0.45 * S_error + 0.35 * S_root_cause
        
        return S_total, S_error, S_ref, S_root_cause
    
    def _find_best_matching_strategy(
        self,
        entry_reflection: Dict[str, Any],
        candidate_strategy_ids: List[str],
    ) -> Tuple[Optional[str], float, float]:
        """
        Find the best matching strategy for a feedback entry.
        
        Args:
            entry_reflection: Reflection dict from feedback entry
            candidate_strategy_ids: List of strategy IDs to compare against
            
        Returns:
            Tuple of (best_strategy_id, S_total, S_error)
            Returns (None, 0.0, 0.0) if no candidates
        """
        if not candidate_strategy_ids:
            return None, 0.0, 0.0
        
        best_id = None
        best_S_total = -1.0
        best_S_error = 0.0
        
        strategies = self.playbook.get("strategies", {})
        
        for strategy_id in candidate_strategy_ids:
            strategy = strategies.get(strategy_id)
            if not strategy:
                continue
            
            strategy_reflection = strategy.get("reflection", {})
            
            S_total, S_error, S_ref, S_root_cause = self._compute_entry_similarity(
                entry_reflection, strategy_reflection
            )
            
            self.logger.debug(
                f"Similarity [{entry_reflection.get('error_identification', '')[:30]}...] vs [{strategy_id}]: "
                f"S_total={S_total:.3f}, S_error={S_error:.3f}, S_ref={S_ref:.3f}, S_root={S_root_cause:.3f}"
            )
            
            if S_total > best_S_total:
                best_S_total = S_total
                best_S_error = S_error
                best_id = strategy_id
        
        return best_id, best_S_total, best_S_error

    # ------------------------------------------------------------------ #
    # Level 1: Real-time Persistence (WAL)
    # ------------------------------------------------------------------ #
    def _write_wal(self, operation: Dict[str, Any]) -> None:
        """
        Write an operation to the Write-Ahead Log (JSONL format).
        
        Args:
            operation: Operation dictionary to log
        """
        log_entry = {
            "ts": int(time.time()),
            **operation,
        }
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        except Exception as e:
            self.logger.error(f"Error writing to WAL: {e}")

    # ------------------------------------------------------------------ #
    # Level 2: Checkpoint Persistence (Snapshot)
    # ------------------------------------------------------------------ #
    def save_snapshot(self, tag: str) -> str:
        """
        Save a snapshot of the current playbook state.
        
        Args:
            tag: Tag for the snapshot (e.g., "iter_001", "first_compile_success", "best_loss_0.15")
        
        Returns:
            Path to the saved snapshot file
        """
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"playbook_{timestamp}_{tag}.json"
        snapshot_path = os.path.join(self.snapshots_dir, filename)
        
        # Save snapshot
        with open(snapshot_path, "w", encoding="utf-8") as f:
            json.dump(self.playbook, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Snapshot saved: {snapshot_path}")
        
        # Cleanup old snapshots (rolling window)
        self._cleanup_snapshots()
        
        return snapshot_path
    
    def save_iteration_snapshot(self, iteration: int) -> str:
        """Save a snapshot for a completed iteration."""
        tag = f"iter_{iteration:03d}"
        return self.save_snapshot(tag)
    
    def save_milestone_snapshot(self, milestone: str, value: Optional[Any] = None) -> str:
        """
        Save a milestone snapshot.
        
        Args:
            milestone: Milestone type (e.g., "first_compile_success", "best_loss")
            value: Optional value to include in tag (e.g., loss value)
        
        Returns:
            Path to the saved snapshot file
        """
        if value is not None:
            tag = f"{milestone}_{value}"
        else:
            tag = milestone
        return self.save_snapshot(tag)
    
    def _cleanup_snapshots(self) -> None:
        """
        Clean up old snapshots, keeping only the most recent MAX_SNAPSHOTS
        plus any milestone snapshots.
        """
        # Get all snapshot files
        pattern = os.path.join(self.snapshots_dir, "playbook_*.json")
        snapshot_files = glob.glob(pattern)
        
        if len(snapshot_files) <= self.MAX_SNAPSHOTS:
            return
        
        # Separate milestone and regular snapshots
        milestone_files = []
        regular_files = []
        
        for f in snapshot_files:
            filename = os.path.basename(f)
            is_milestone = any(tag in filename for tag in self.MILESTONE_TAGS)
            if is_milestone:
                milestone_files.append(f)
            else:
                regular_files.append(f)
        
        # Sort regular files by modification time (oldest first)
        regular_files.sort(key=lambda x: os.path.getmtime(x))
        
        # Calculate how many regular files to delete
        total_to_keep = self.MAX_SNAPSHOTS - len(milestone_files)
        files_to_delete = len(regular_files) - max(0, total_to_keep)
        
        # Delete oldest regular files
        for i in range(files_to_delete):
            try:
                os.remove(regular_files[i])
                self.logger.debug(f"Deleted old snapshot: {regular_files[i]}")
            except Exception as e:
                self.logger.error(f"Error deleting snapshot {regular_files[i]}: {e}")

    # ------------------------------------------------------------------ #
    # Level 3: Session Persistence (Final Archive)
    # ------------------------------------------------------------------ #
    def finalize(self) -> str:
        """
        Finalize the playbook session: convert strategies to insights and save final archive.
        
        Process:
        1. Extract from each strategy: strategy_id, meta_info, issue_type, key_insight, metric_links
        2. Convert status:
           - "resolved" → "solved" (successfully fixed)
           - "open", "queued", "in_progress" → "unsolved" (still open issues)
        3. Calculate token_count for extracted reflection content
        4. Preserve counters: usage_count, unusage_count, success_attribution, failure_attribution
        5. Store converted entries in playbook["insights"]
        6. Clear playbook["strategies"]
        
        Status mapping:
        - resolved -> solved (E_resolved occurred)
        - open/queued/in_progress -> unsolved (issue still exists)
        
        Returns:
            Path to the final archived playbook
        """
        strategies = self.playbook.get("strategies", {})
        
        # Build insights dictionary with extracted and converted data
        insights = {}
        solved_count = 0
        unsolved_count = 0
        deleted_count = 0
        
        for strategy_id, entry in strategies.items():
            # Get current status
            current_status = entry.get("meta_info", {}).get("status", self.STATUS_OPEN)
            # Backward compatibility
            if "meta_info" not in entry:
                current_status = entry.get("status", self.STATUS_OPEN)
            
            # Extract counter values from original meta_info
            original_meta = entry.get("meta_info", {})
            usage_count = original_meta.get("usage_count", entry.get("usage_count", 0))
            unusage_count = original_meta.get("unusage_count", entry.get("unusage_count", 0))
            success_attribution = original_meta.get("success_attribution", entry.get("success_attribution", 0))
            failure_attribution = original_meta.get("failure_attribution", entry.get("failure_attribution", 0))
            
            # Convert status for insights:
            # - resolved -> solved
            # - open/queued/in_progress -> unsolved
            if current_status == self.STATUS_RESOLVED:
                new_status = "solved"
                solved_count += 1
            elif current_status in [self.STATUS_OPEN, self.STATUS_QUEUED, self.STATUS_IN_PROGRESS]:
                new_status = "unsolved"
                unsolved_count += 1
            else:
                # Backward compatibility: handle old status values
                if current_status == "active":
                    new_status = "unsolved"
                    unsolved_count += 1
                elif current_status == "obsolete":
                    new_status = "solved"
                    solved_count += 1
                elif current_status == "rolled_back":
                    deleted_count += 1
                    self.logger.debug(f"Skipping rolled_back strategy: {strategy_id}")
                    continue
                else:
                    new_status = "unsolved"
                    unsolved_count += 1
            
            # Extract issue_type, key_insight, and metric_links from reflection
            reflection = entry.get("reflection", {})
            extracted_reflection = {
                "issue_type": reflection.get("issue_type", ""),
                "key_insight": reflection.get("key_insight", ""),
                "metric_links": reflection.get("metric_links", []),
            }
            
            # Calculate token_count for the extracted reflection content
            insight_token_count = self._count_reflection_tokens(extracted_reflection)
            
            # Build the meta_info with status, token_count, and preserved counters
            meta_info = {
                "status": new_status,
                "token_count": insight_token_count,
                "usage_count": usage_count,
                "unusage_count": unusage_count,
                "success_attribution": success_attribution,
                "failure_attribution": failure_attribution,
            }
            
            # Build the insight entry
            insight_entry = {
                "meta_info": meta_info,
                "reflection": extracted_reflection,
            }
            
            insights[strategy_id] = insight_entry
        
        # Store insights in playbook
        self.playbook["insights"] = insights
        
        # Clear strategies
        self.playbook["strategies"] = {}
        
        # Calculate final statistics
        total_insights = len(insights)
        total_tokens = sum(
            entry.get("meta_info", {}).get("token_count", 0)
            for entry in insights.values()
        )
        
        # Update metadata
        self.playbook["playbook_metadata"]["total_token_count"] = total_tokens
        self.playbook["playbook_metadata"]["total_insights"] = total_insights
        self.playbook["playbook_metadata"]["solved_count"] = solved_count
        self.playbook["playbook_metadata"]["unsolved_count"] = unsolved_count
        self.playbook["playbook_metadata"]["deleted_count"] = deleted_count
        self.playbook["playbook_metadata"]["finalized_at"] = datetime.utcnow().isoformat()
        
        # Save current playbook
        self._save_current()
        
        # Save final archive snapshot
        snapshot_path = self.save_snapshot("final_archive")
        
        # Clear WAL log
        try:
            if os.path.exists(self.log_path):
                os.remove(self.log_path)
                self.logger.info("WAL log cleared after finalization")
        except Exception as e:
            self.logger.error(f"Error clearing WAL log: {e}")
        
        self.logger.info(
            f"Playbook finalized. Insights: {total_insights} "
            f"(solved: {solved_count}, unsolved: {unsolved_count}, deleted: {deleted_count}), "
            f"Total tokens: {total_tokens}"
        )
        return snapshot_path

    # ------------------------------------------------------------------ #
    # Basic Persistence helpers
    # ------------------------------------------------------------------ #
    def _save_current(self) -> None:
        """Save current playbook to the current directory."""
        os.makedirs(self.current_dir, exist_ok=True)
        with open(self.playbook_path, "w", encoding="utf-8") as f:
            json.dump(self.playbook, f, indent=2, ensure_ascii=False)
    
    def save(self, path: Optional[str] = None) -> None:
        """Persist playbook to JSON (alias for _save_current or custom path)."""
        if path:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.playbook, f, indent=2, ensure_ascii=False)
            self.logger.info("Playbook saved to %s", path)
        else:
            self._save_current()
            self.logger.info("Playbook saved to %s", self.playbook_path)

    def load(self, path: Optional[str] = None) -> None:
        """Load playbook from JSON."""
        target = path or self.playbook_path
        if not os.path.exists(target):
            self.logger.warning("Playbook file not found: %s", target)
            return
        self.playbook = self._load_from_file(target)
        self.logger.info("Playbook loaded from %s", target)

    def load_snapshot(self, snapshot_name: str) -> bool:
        """
        Load a specific snapshot.
        
        Args:
            snapshot_name: Name of the snapshot file (with or without .json extension)
        
        Returns:
            True if loaded successfully, False otherwise
        """
        if not snapshot_name.endswith(".json"):
            snapshot_name += ".json"
        
        snapshot_path = os.path.join(self.snapshots_dir, snapshot_name)
        if not os.path.exists(snapshot_path):
            self.logger.warning(f"Snapshot not found: {snapshot_path}")
            return False
        
        self.playbook = self._load_from_file(snapshot_path)
        self._save_current()
        self.logger.info(f"Loaded snapshot: {snapshot_path}")
        return True

    def list_snapshots(self) -> List[str]:
        """List all available snapshots."""
        pattern = os.path.join(self.snapshots_dir, "playbook_*.json")
        snapshot_files = glob.glob(pattern)
        return sorted([os.path.basename(f) for f in snapshot_files])

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _touch_metadata(self, iteration: Optional[int] = None) -> None:
        """Update metadata timestamps."""
        meta = self.playbook.get("playbook_metadata", {})
        meta["last_updated_time"] = datetime.utcnow().isoformat()
        if iteration is not None:
            meta["last_updated_iteration"] = str(iteration)
        self.playbook["playbook_metadata"] = meta

    @staticmethod
    def _load_from_file(path: str) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)


def main():
    """Command-line interface for PlaybookManager cleanup operations."""
    parser = argparse.ArgumentParser(
        description="PlaybookManager cleanup tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Delete playbook.json
  python -m core.playbook_manager --delete-playbook
  
  # Delete playbook.log
  python -m core.playbook_manager --delete-log
  
  # Delete all snapshots
  python -m core.playbook_manager --delete-snapshots
  
  # Delete everything (playbook.json, playbook.log, and all snapshots)
  python -m core.playbook_manager --delete-all
        """
    )
    
    parser.add_argument(
        "--delete-playbook",
        action="store_true",
        help="Delete playbook.json file"
    )
    
    parser.add_argument(
        "--delete-log",
        action="store_true",
        help="Delete playbook.log file"
    )
    
    parser.add_argument(
        "--delete-snapshots",
        action="store_true",
        help="Delete all snapshot files in snapshots directory"
    )
    
    parser.add_argument(
        "--delete-all",
        action="store_true",
        help="Delete playbook.json, playbook.log, and all snapshots"
    )
    
    parser.add_argument(
        "--storage-root",
        type=str,
        default=None,
        help="Custom storage root directory (default: project_root/playbook_storage)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger("SOCIA.PlaybookManager.CLI")
    
    # If no operation specified, show help
    if not any([args.delete_playbook, args.delete_log, args.delete_snapshots, args.delete_all]):
        parser.print_help()
        return
    
    # Initialize PlaybookManager to get paths
    try:
        manager = PlaybookManager(storage_root=args.storage_root, logger=logger)
    except Exception as e:
        logger.error(f"Failed to initialize PlaybookManager: {e}")
        return
    
    deleted_count = 0
    errors = []
    
    # Delete playbook.json
    if args.delete_all or args.delete_playbook:
        if os.path.exists(manager.playbook_path):
            try:
                os.remove(manager.playbook_path)
                logger.info(f"✓ Deleted: {manager.playbook_path}")
                deleted_count += 1
            except Exception as e:
                error_msg = f"Failed to delete {manager.playbook_path}: {e}"
                logger.error(error_msg)
                errors.append(error_msg)
        else:
            logger.info(f"File not found (skipped): {manager.playbook_path}")
    
    # Delete playbook.log
    if args.delete_all or args.delete_log:
        if os.path.exists(manager.log_path):
            try:
                os.remove(manager.log_path)
                logger.info(f"✓ Deleted: {manager.log_path}")
                deleted_count += 1
            except Exception as e:
                error_msg = f"Failed to delete {manager.log_path}: {e}"
                logger.error(error_msg)
                errors.append(error_msg)
        else:
            logger.info(f"File not found (skipped): {manager.log_path}")
    
    # Delete all snapshots
    if args.delete_all or args.delete_snapshots:
        if os.path.exists(manager.snapshots_dir):
            snapshot_pattern = os.path.join(manager.snapshots_dir, "playbook_*.json")
            snapshot_files = glob.glob(snapshot_pattern)
            
            if snapshot_files:
                for snapshot_file in snapshot_files:
                    try:
                        os.remove(snapshot_file)
                        logger.info(f"✓ Deleted: {os.path.basename(snapshot_file)}")
                        deleted_count += 1
                    except Exception as e:
                        error_msg = f"Failed to delete {snapshot_file}: {e}"
                        logger.error(error_msg)
                        errors.append(error_msg)
                logger.info(f"Deleted {len(snapshot_files)} snapshot file(s)")
            else:
                logger.info(f"No snapshot files found in {manager.snapshots_dir}")
        else:
            logger.info(f"Directory not found (skipped): {manager.snapshots_dir}")
    
    # Summary
    logger.info("=" * 60)
    if deleted_count > 0:
        logger.info(f"✓ Successfully deleted {deleted_count} file(s)")
    else:
        logger.info("No files were deleted")
    
    if errors:
        logger.warning(f"⚠ {len(errors)} error(s) occurred:")
        for error in errors:
            logger.warning(f"  - {error}")


def test_add_feedback_entries():
    """
    Test function for add_feedback_entries with sample feedback data.
    
    This tests the similarity-based merging logic:
    1. Load existing playbook (or create new if not exists)
    2. First call: Add initial feedback entries
    3. Second call: Add similar/different entries to test merge behavior
    4. Test finalize() to convert strategies to insights
    """
    import tempfile
    import shutil
    
    # Set up logging
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger("SOCIA.PlaybookManager.Test")
    
    # Create a temporary directory for testing
    test_dir = tempfile.mkdtemp(prefix="playbook_test_")
    logger.info(f"Using temporary test directory: {test_dir}")
    
    try:
        # Initialize PlaybookManager with test directory (loads existing or creates new)
        manager = PlaybookManager(storage_root=test_dir, logger=logger)
        logger.info(f"PlaybookManager initialized. Playbook path: {manager.playbook_path}")
        
        # Sample feedback data (iteration 1)
        feedback_iter1 = {
            "missing-co-location-and-occupancy-layer": {
                "issue_type": "DESIGN_MISMATCH",
                "severity": "high",
                "from_user_feedback": False,
                "blueprint_refs": [
                    "interaction_topology.layers.co_location_contact",
                    "interaction_topology.layers.resident_visits_place",
                    "place.dynamic_states.current_occupancy"
                ],
                "code_refs": [
                    {"symbol": "MobilitySimulator.simulate_days", "lines": "unknown"},
                    {"symbol": "POI.current_occupancy", "lines": "unknown"}
                ],
                "evidence": {
                    "user_feedback": None,
                    "error_logs": None,
                    "metrics": "Simulator never outputs co-location/occupancy"
                },
                "error_identification": "Blueprint requires resident→place visit events updating occupancy and an optional co-location contact layer; current implementation generates trajectories but never updates POI occupancy.",
                "root_cause_analysis": "In MobilitySimulator.simulate_days(), arrivals are appended as Event objects but POI.current_occupancy is never incremented/decremented.",
                "correct_approach": "Implement explicit visit events with (arrival, departure) times. During simulation: increment poi.current_occupancy on arrival, decrement on departure.",
                "key_insight": "If the spec includes an interaction layer (occupancy/contact), add at least minimal state updates."
            },
            "calibration-optimizes-on-training-not-validation": {
                "issue_type": "DESIGN_MISMATCH",
                "severity": "high",
                "from_user_feedback": False,
                "blueprint_refs": [
                    "execution: Optimize calibratable parameters by minimizing validation loss",
                    "outputs: evaluation_results_on_validation"
                ],
                "code_refs": [
                    {"symbol": "main", "lines": "unknown"},
                    {"symbol": "RandomSearchCalibrator.fit", "lines": "unknown"}
                ],
                "evidence": {
                    "user_feedback": None,
                    "error_logs": None,
                    "metrics": "Calibration uses train_ground_truth_by_agent"
                },
                "error_identification": "Blueprint specifies calibrating parameters by minimizing validation loss, but calibration currently minimizes loss on training days only.",
                "root_cause_analysis": "In main(), RandomSearchCalibrator.fit() is called with train_ground_truth_by_agent=train_by_agent.",
                "correct_approach": "Change calibration inputs to use validation ground truth and validation dates.",
                "key_insight": "If the spec says 'calibrate on validation', the objective must be computed on holdout data."
            },
            "stop-count-distribution-smoothing-bug": {
                "issue_type": "CODE_BUG",
                "severity": "medium",
                "from_user_feedback": False,
                "blueprint_refs": [
                    "simulation_evaluation.metrics.daily_stop_count_distribution_kl: with smoothing"
                ],
                "code_refs": [
                    {"symbol": "Evaluator._stop_count_dist", "lines": "unknown"}
                ],
                "evidence": {
                    "user_feedback": None,
                    "error_logs": None,
                    "metrics": "stop_count_kl=0.52"
                },
                "error_identification": "Stop-count KL uses smoothing inconsistently: smoothing is applied only to keys present in each dataset's own support.",
                "root_cause_analysis": "Evaluator._stop_count_dist() constructs smoothed probabilities only for support=sorted(ctr.keys()).",
                "correct_approach": "Compute smoothed distributions on the union support directly.",
                "key_insight": "When comparing discrete distributions, apply smoothing on the same support you use for divergence."
            }
        }
        
        logger.info("=" * 70)
        logger.info("TEST 1: Adding initial feedback entries (iteration 1)")
        logger.info("=" * 70)
        
        added_ids_1 = manager.add_feedback_entries(feedback_iter1, iteration=1)
        logger.info(f"Added {len(added_ids_1)} entries: {added_ids_1}")
        
        # Print current playbook state
        logger.info(f"\nPlaybook strategies count: {len(manager.playbook.get('strategies', {}))}")
        for sid, entry in manager.playbook.get("strategies", {}).items():
            status = entry.get("meta_info", {}).get("status", "unknown")
            issue_type = entry.get("reflection", {}).get("issue_type", "unknown")
            logger.info(f"  - {sid}: status={status}, type={issue_type}")
        
        # Sample feedback data (iteration 2) - some similar, some different
        feedback_iter2 = {
            # Similar to "missing-co-location-and-occupancy-layer" - should MERGE
            "occupancy-tracking-not-implemented": {
                "issue_type": "DESIGN_MISMATCH",
                "severity": "high",
                "from_user_feedback": False,
                "blueprint_refs": [
                    "interaction_topology.layers.co_location_contact",
                    "place.dynamic_states.current_occupancy",
                    "protocol: compute co-location contacts"
                ],
                "code_refs": [
                    {"symbol": "MobilitySimulator.simulate_days", "lines": "150-200"}
                ],
                "evidence": {
                    "user_feedback": None,
                    "error_logs": None,
                    "metrics": "No occupancy data in output"
                },
                "error_identification": "The blueprint requires updating POI occupancy when residents visit places, but the current code never tracks or updates occupancy counts.",
                "root_cause_analysis": "MobilitySimulator.simulate_days() creates arrival events but does not maintain POI.current_occupancy state.",
                "correct_approach": "Add occupancy tracking: increment on arrival, decrement on departure, optionally compute co-location.",
                "key_insight": "Occupancy is a fundamental state that must be tracked for any location-based interaction model."
            },
            # Different issue - should be NEW
            "memory-leak-in-simulation-loop": {
                "issue_type": "CODE_BUG",
                "severity": "high",
                "from_user_feedback": False,
                "blueprint_refs": [],
                "code_refs": [
                    {"symbol": "MobilitySimulator.rollout", "lines": "300-350"}
                ],
                "evidence": {
                    "user_feedback": None,
                    "error_logs": "Memory usage grows linearly with simulation steps",
                    "metrics": "OOM after 10000 steps"
                },
                "error_identification": "Memory usage increases unboundedly during simulation rollout.",
                "root_cause_analysis": "Event history is accumulated in a list without any cleanup or windowing.",
                "correct_approach": "Implement a sliding window for event history or periodically flush old events.",
                "key_insight": "Long-running simulations need memory management strategies."
            },
            # Similar to "calibration-optimizes-on-training-not-validation" - should MERGE
            "wrong-calibration-dataset": {
                "issue_type": "DESIGN_MISMATCH",
                "severity": "high",
                "from_user_feedback": False,
                "blueprint_refs": [
                    "execution: minimize validation loss",
                    "calibration: use holdout data"
                ],
                "code_refs": [
                    {"symbol": "RandomSearchCalibrator.fit", "lines": "50-100"}
                ],
                "evidence": {
                    "user_feedback": None,
                    "error_logs": None,
                    "metrics": "Overfitting on training data"
                },
                "error_identification": "The calibration loop optimizes on training data instead of validation data as specified in the blueprint.",
                "root_cause_analysis": "RandomSearchCalibrator.fit() receives training ground truth instead of validation ground truth.",
                "correct_approach": "Pass validation data to the calibrator for parameter optimization.",
                "key_insight": "Calibration must use holdout data to avoid overfitting."
            }
        }
        
        logger.info("")
        logger.info("=" * 70)
        logger.info("TEST 2: Adding feedback entries (iteration 2) - testing merge")
        logger.info("=" * 70)
        
        added_ids_2 = manager.add_feedback_entries(feedback_iter2, iteration=2)
        logger.info(f"Added {len(added_ids_2)} entries: {added_ids_2}")
        
        # Print updated playbook state
        logger.info(f"\nPlaybook strategies count after iteration 2: {len(manager.playbook.get('strategies', {}))}")
        
        active_count = 0
        obsolete_count = 0
        for sid, entry in manager.playbook.get("strategies", {}).items():
            status = entry.get("meta_info", {}).get("status", "unknown")
            issue_type = entry.get("reflection", {}).get("issue_type", "unknown")
            error_id = entry.get("reflection", {}).get("error_identification", "")[:60]
            logger.info(f"  - {sid}")
            logger.info(f"      status: {status}, type: {issue_type}")
            logger.info(f"      error: {error_id}...")
            if status == "active":
                active_count += 1
            elif status == "obsolete":
                obsolete_count += 1
        
        logger.info(f"\nStatus summary: active={active_count}, obsolete={obsolete_count}")
        
        # Test finalize
        logger.info("")
        logger.info("=" * 70)
        logger.info("TEST 3: Finalizing playbook - converting strategies to insights")
        logger.info("=" * 70)
        
        archive_path = manager.finalize()
        logger.info(f"Archive saved to: {archive_path}")
        
        # Print insights
        logger.info("\nINSIGHTS after finalize:")
        logger.info(f"Total insights: {len(manager.playbook.get('insights', {}))}")
        
        for iid, insight in manager.playbook.get("insights", {}).items():
            status = insight.get("meta_info", {}).get("status", "unknown")
            issue_type = insight.get("reflection", {}).get("issue_type", "unknown")
            key_insight = insight.get("reflection", {}).get("key_insight", "")[:60]
            logger.info(f"  - {iid}")
            logger.info(f"      status: {status}, type: {issue_type}")
            logger.info(f"      key_insight: {key_insight}...")
        
        # Print final metadata
        logger.info("\nFinal playbook metadata:")
        for key, value in manager.playbook.get("playbook_metadata", {}).items():
            logger.info(f"  {key}: {value}")
        
        # Verify strategies is cleared
        strategies_count = len(manager.playbook.get("strategies", {}))
        logger.info(f"\nStrategies count after finalize: {strategies_count}")
        assert strategies_count == 0, "Strategies should be cleared after finalize!"
        
        logger.info("")
        logger.info("=" * 70)
        logger.info("TEST COMPLETED SUCCESSFULLY")
        logger.info("=" * 70)
        
    finally:
        # Cleanup test directory
        shutil.rmtree(test_dir, ignore_errors=True)
        logger.info(f"Cleaned up test directory: {test_dir}")


def test_with_existing_playbook(playbook_path: Optional[str] = None):
    """
    Test add_feedback_entries using existing playbook from default location.
    
    This tests the similarity-based merging logic:
    1. Load existing playbook (or create new if not exists)
    2. First call: Add initial feedback entries (3 entries)
    3. Second call: Add similar/different entries to test merge behavior (3 entries)
    4. Test finalize() to convert strategies to insights
    
    Args:
        playbook_path: Optional custom path to playbook storage root.
                       If None, uses default location (project_root/playbook_storage).
    """
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger("SOCIA.PlaybookManager.Test")
    
    # Initialize PlaybookManager (loads existing playbook or creates new)
    manager = PlaybookManager(storage_root=playbook_path, logger=logger)
    
    logger.info("=" * 70)
    logger.info("Loading existing playbook")
    logger.info("=" * 70)
    logger.info(f"Playbook path: {manager.playbook_path}")
    logger.info(f"Current strategies count: {len(manager.playbook.get('strategies', {}))}")
    logger.info(f"Current insights count: {len(manager.playbook.get('insights', {}))}")
    
    # Sample feedback data (iteration 1) - 3 entries
    feedback_iter1 = {
        "missing-co-location-and-occupancy-layer": {
            "issue_type": "DESIGN_MISMATCH",
            "severity": "high",
            "from_user_feedback": False,
            "blueprint_refs": [
                "interaction_topology.layers.co_location_contact",
                "interaction_topology.layers.resident_visits_place",
                "place.dynamic_states.current_occupancy"
            ],
            "code_refs": [
                {"symbol": "MobilitySimulator.simulate_days", "lines": "unknown"},
                {"symbol": "POI.current_occupancy", "lines": "unknown"}
            ],
            "evidence": {
                "user_feedback": None,
                "error_logs": None,
                "metrics": "Simulator never outputs co-location/occupancy"
            },
            "error_identification": "Blueprint requires resident→place visit events updating occupancy and an optional co-location contact layer; current implementation generates trajectories but never updates POI occupancy.",
            "root_cause_analysis": "In MobilitySimulator.simulate_days(), arrivals are appended as Event objects but POI.current_occupancy is never incremented/decremented.",
            "correct_approach": "Implement explicit visit events with (arrival, departure) times. During simulation: increment poi.current_occupancy on arrival, decrement on departure.",
            "key_insight": "If the spec includes an interaction layer (occupancy/contact), add at least minimal state updates."
        },
        "calibration-optimizes-on-training-not-validation": {
            "issue_type": "DESIGN_MISMATCH",
            "severity": "high",
            "from_user_feedback": False,
            "blueprint_refs": [
                "execution: Optimize calibratable parameters by minimizing validation loss",
                "outputs: evaluation_results_on_validation"
            ],
            "code_refs": [
                {"symbol": "main", "lines": "unknown"},
                {"symbol": "RandomSearchCalibrator.fit", "lines": "unknown"}
            ],
            "evidence": {
                "user_feedback": None,
                "error_logs": None,
                "metrics": "Calibration uses train_ground_truth_by_agent"
            },
            "error_identification": "Blueprint specifies calibrating parameters by minimizing validation loss, but calibration currently minimizes loss on training days only.",
            "root_cause_analysis": "In main(), RandomSearchCalibrator.fit() is called with train_ground_truth_by_agent=train_by_agent.",
            "correct_approach": "Change calibration inputs to use validation ground truth and validation dates.",
            "key_insight": "If the spec says 'calibrate on validation', the objective must be computed on holdout data."
        },
        "stop-count-distribution-smoothing-bug": {
            "issue_type": "CODE_BUG",
            "severity": "medium",
            "from_user_feedback": False,
            "blueprint_refs": [
                "simulation_evaluation.metrics.daily_stop_count_distribution_kl: with smoothing"
            ],
            "code_refs": [
                {"symbol": "Evaluator._stop_count_dist", "lines": "unknown"}
            ],
            "evidence": {
                "user_feedback": None,
                "error_logs": None,
                "metrics": "stop_count_kl=0.52"
            },
            "error_identification": "Stop-count KL uses smoothing inconsistently: smoothing is applied only to keys present in each dataset's own support.",
            "root_cause_analysis": "Evaluator._stop_count_dist() constructs smoothed probabilities only for support=sorted(ctr.keys()).",
            "correct_approach": "Compute smoothed distributions on the union support directly.",
            "key_insight": "When comparing discrete distributions, apply smoothing on the same support you use for divergence."
        }
    }
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("TEST 1: Adding initial feedback entries (iteration 1) - 3 entries")
    logger.info("=" * 70)
    
    added_ids_1 = manager.add_feedback_entries(feedback_iter1, iteration=1)
    logger.info(f"Added {len(added_ids_1)} entries: {added_ids_1}")
    
    # Print current playbook state
    logger.info(f"\nPlaybook strategies count: {len(manager.playbook.get('strategies', {}))}")
    for sid, entry in manager.playbook.get("strategies", {}).items():
        status = entry.get("meta_info", {}).get("status", "unknown")
        issue_type = entry.get("reflection", {}).get("issue_type", "unknown")
        logger.info(f"  - {sid}: status={status}, type={issue_type}")
    
    # Sample feedback data (iteration 2) - 3 entries, some similar for merge testing
    feedback_iter2 = {
        # Similar to "missing-co-location-and-occupancy-layer" - should MERGE
        "occupancy-tracking-not-implemented": {
            "issue_type": "DESIGN_MISMATCH",
            "severity": "high",
            "from_user_feedback": False,
            "blueprint_refs": [
                "interaction_topology.layers.co_location_contact",
                "place.dynamic_states.current_occupancy",
                "protocol: compute co-location contacts"
            ],
            "code_refs": [
                {"symbol": "MobilitySimulator.simulate_days", "lines": "150-200"}
            ],
            "evidence": {
                "user_feedback": None,
                "error_logs": None,
                "metrics": "No occupancy data in output"
            },
            "error_identification": "The blueprint requires updating POI occupancy when residents visit places, but the current code never tracks or updates occupancy counts.",
            "root_cause_analysis": "MobilitySimulator.simulate_days() creates arrival events but does not maintain POI.current_occupancy state.",
            "correct_approach": "Add occupancy tracking: increment on arrival, decrement on departure, optionally compute co-location.",
            "key_insight": "Occupancy is a fundamental state that must be tracked for any location-based interaction model."
        },
        # Different issue - should be NEW
        "memory-leak-in-simulation-loop": {
            "issue_type": "CODE_BUG",
            "severity": "high",
            "from_user_feedback": False,
            "blueprint_refs": [],
            "code_refs": [
                {"symbol": "MobilitySimulator.rollout", "lines": "300-350"}
            ],
            "evidence": {
                "user_feedback": None,
                "error_logs": "Memory usage grows linearly with simulation steps",
                "metrics": "OOM after 10000 steps"
            },
            "error_identification": "Memory usage increases unboundedly during simulation rollout.",
            "root_cause_analysis": "Event history is accumulated in a list without any cleanup or windowing.",
            "correct_approach": "Implement a sliding window for event history or periodically flush old events.",
            "key_insight": "Long-running simulations need memory management strategies."
        },
        # Similar to "calibration-optimizes-on-training-not-validation" - should MERGE
        "wrong-calibration-dataset": {
            "issue_type": "DESIGN_MISMATCH",
            "severity": "high",
            "from_user_feedback": False,
            "blueprint_refs": [
                "execution: minimize validation loss",
                "calibration: use holdout data"
            ],
            "code_refs": [
                {"symbol": "RandomSearchCalibrator.fit", "lines": "50-100"}
            ],
            "evidence": {
                "user_feedback": None,
                "error_logs": None,
                "metrics": "Overfitting on training data"
            },
            "error_identification": "The calibration loop optimizes on training data instead of validation data as specified in the blueprint.",
            "root_cause_analysis": "RandomSearchCalibrator.fit() receives training ground truth instead of validation ground truth.",
            "correct_approach": "Pass validation data to the calibrator for parameter optimization.",
            "key_insight": "Calibration must use holdout data to avoid overfitting."
        }
    }
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("TEST 2: Adding feedback entries (iteration 2) - 3 entries, testing merge")
    logger.info("=" * 70)
    
    added_ids_2 = manager.add_feedback_entries(feedback_iter2, iteration=2)
    logger.info(f"Added {len(added_ids_2)} entries: {added_ids_2}")
    
    # Print updated playbook state
    logger.info(f"\nPlaybook strategies count after iteration 2: {len(manager.playbook.get('strategies', {}))}")
    
    active_count = 0
    obsolete_count = 0
    for sid, entry in manager.playbook.get("strategies", {}).items():
        status = entry.get("meta_info", {}).get("status", "unknown")
        issue_type = entry.get("reflection", {}).get("issue_type", "unknown")
        error_id = entry.get("reflection", {}).get("error_identification", "")[:60]
        logger.info(f"  - {sid}")
        logger.info(f"      status: {status}, type: {issue_type}")
        logger.info(f"      error: {error_id}...")
        if status == "active":
            active_count += 1
        elif status == "obsolete":
            obsolete_count += 1
    
    logger.info(f"\nStatus summary: active={active_count}, obsolete={obsolete_count}")
    
    # Test finalize
    logger.info("")
    logger.info("=" * 70)
    logger.info("TEST 3: Finalizing playbook - converting strategies to insights")
    logger.info("=" * 70)
    
    archive_path = manager.finalize()
    logger.info(f"Archive saved to: {archive_path}")
    
    # Print insights
    logger.info("\nINSIGHTS after finalize:")
    logger.info(f"Total insights: {len(manager.playbook.get('insights', {}))}")
    
    for iid, insight in manager.playbook.get("insights", {}).items():
        status = insight.get("meta_info", {}).get("status", "unknown")
        issue_type = insight.get("reflection", {}).get("issue_type", "unknown")
        key_insight = insight.get("reflection", {}).get("key_insight", "")[:60]
        logger.info(f"  - {iid}")
        logger.info(f"      status: {status}, type: {issue_type}")
        logger.info(f"      key_insight: {key_insight}...")
    
    # Print final metadata
    logger.info("\nFinal playbook metadata:")
    for key, value in manager.playbook.get("playbook_metadata", {}).items():
        logger.info(f"  {key}: {value}")
    
    # Verify strategies is cleared
    strategies_count = len(manager.playbook.get("strategies", {}))
    logger.info(f"\nStrategies count after finalize: {strategies_count}")
    assert strategies_count == 0, "Strategies should be cleared after finalize!"
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("TEST COMPLETED SUCCESSFULLY - playbook has been updated")
    logger.info("=" * 70)
    logger.info(f"Run 'python -m core.playbook_manager --delete-playbook' to cleanup if needed")


if __name__ == "__main__":
    import sys
    
    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--test":
            # Run full test with temporary directory
            test_add_feedback_entries()
        elif sys.argv[1] == "--test-existing":
            # Test with existing playbook
            playbook_path = sys.argv[2] if len(sys.argv) > 2 else None
            test_with_existing_playbook(playbook_path)
        else:
            # Run cleanup commands
            main()
    else:
        main()
