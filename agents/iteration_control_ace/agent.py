"""
IterationControlAgent: Controls the iteration process, deciding when to continue or stop.

Decision Logic:
1. Metrics Record Management:
   - If current simulation_metrics not empty and history empty -> LLM analysis to establish baseline
   - If metrics record exists and fields match -> compare and update
   - If metrics record exists but fields differ -> LLM analysis to update primary metrics

2. Hard Guardrails (NEVER stop if any is true):
   - Playbook has blocker/high severity unresolved issues (especially from_user_feedback=true blockers)
   - Run stage is not "ran_with_metrics" (crash, incomplete, no metrics)
   - Any primary metric regressed by > threshold (e.g., 3%)

3. Soft Stop Conditions:
   - Plateau: 2 consecutive rounds with primary metrics improvement < 3% and no single metric > 5%
   - Low-issues: 2 consecutive rounds with all new+persist issues being low severity and count <= 3

4. Final Decision:
   - Stop only if: No hard guardrails triggered AND at least one soft stop condition met
"""

import json
import logging
from typing import Dict, Any, Optional, List, Tuple

from agents.base_agent import BaseAgent


class IterationControlAgent(BaseAgent):
    """
    Iteration Control Agent for ACE mode with comprehensive decision logic.
    
    Tracks metrics history and uses a combination of hard guardrails and
    soft stop conditions to determine when to stop iteration.
    """
    
    # Configuration constants
    REGRESSION_THRESHOLD = 0.03  # 3% regression threshold
    IMPROVEMENT_THRESHOLD = 0.03  # 3% improvement threshold for plateau detection
    SIGNIFICANT_IMPROVEMENT = 0.05  # 5% threshold for significant single-metric improvement
    MAX_LOW_ISSUES = 3  # Maximum low-severity issues for soft stop
    PLATEAU_CONSECUTIVE_ROUNDS = 2  # Number of rounds for plateau detection
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Metrics history tracking
        self._metrics_record: Optional[Dict[str, Any]] = None
        self._metrics_history: List[Dict[str, float]] = []  # List of metrics per iteration
        self._improvement_history: List[Dict[str, float]] = []  # List of improvement rates per iteration
        
        # Issue history tracking for soft stop detection
        self._issue_history: List[Dict[str, Any]] = []
    
    def reset_history(self):
        """Reset all history tracking. Call this when starting a new task."""
        self._metrics_record = None
        self._metrics_history = []
        self._improvement_history = []
        self._issue_history = []
        self.logger.info("Iteration control history reset")
    
    def process(
        self,
        current_iteration: int,
        max_iterations: int,
        simulation_results: Optional[Dict[str, Any]] = None,
        feedback: Optional[Dict[str, Any]] = None,
        playbook: Optional[Dict[str, Any]] = None,
        verification_results: Optional[Dict[str, Any]] = None,
        evaluation_results: Optional[Dict[str, Any]] = None,
        task_spec: Optional[Dict[str, Any]] = None,
        auto_mode: bool = True
    ) -> Dict[str, Any]:
        """
        Decide whether to continue with another iteration.

        Args:
            current_iteration: Current iteration number (0-based)
            max_iterations: Maximum number of iterations
            simulation_results: Results from the Simulation Execution ACE Agent
            feedback: Feedback from the Feedback Generation ACE Agent
            playbook: Current playbook state (for checking unresolved issues)
            verification_results: Results from the Code Verification Agent (optional)
            evaluation_results: Results from the Result Evaluation Agent (optional)
            task_spec: Task specification (optional)
            auto_mode: Whether running in automatic mode

        Returns:
            Dictionary containing iteration decision with 'continue', 'reason', and analysis details
        """
        self.logger.info(f"Making iteration decision for iteration {current_iteration}")
        
        # ============================================================================
        # COMMENTED OUT: Complex decision logic (metrics tracking, guardrails, etc.)
        # ============================================================================
        # # Step 1: Update metrics record
        # metrics_update_info = self._update_metrics_record(simulation_results, current_iteration)
        # 
        # # Step 2: Update issue history from feedback
        # issue_summary = self._update_issue_history(feedback, current_iteration)
        # 
        # # Step 3: Check hard guardrails
        # hard_stop_blocked, guardrail_reasons = self._check_hard_guardrails(
        #     simulation_results=simulation_results,
        #     feedback=feedback,
        #     playbook=playbook,
        #     current_iteration=current_iteration
        # )
        # 
        # # Step 4: Check soft stop conditions
        # soft_stop_triggered, soft_stop_reasons = self._check_soft_stop_conditions(
        #     current_iteration=current_iteration
        # )
        # 
        # # Step 5: Make final decision
        # # Hard max iterations check
        # if current_iteration >= max_iterations - 1:
        #     should_continue = False
        #     reason = f"Reached maximum iteration limit ({max_iterations})"
        # elif hard_stop_blocked:
        #     # Hard guardrails triggered - MUST continue
        #     should_continue = True
        #     reason = f"Hard guardrails active: {'; '.join(guardrail_reasons)}"
        # elif soft_stop_triggered:
        #     # No hard guardrails and soft stop conditions met - can stop
        #     should_continue = False
        #     reason = f"Soft stop conditions met: {'; '.join(soft_stop_reasons)}"
        # else:
        #     # Default: continue
        #     should_continue = True
        #     reason = "No stop conditions met, continuing iteration"
        # 
        # iteration_decision = {
        #     "continue": should_continue,
        #     "reason": reason,
        #     "analysis": {
        #         "metrics_update": metrics_update_info,
        #         "issue_summary": issue_summary,
        #         "hard_guardrails": {
        #             "blocked": hard_stop_blocked,
        #             "reasons": guardrail_reasons
        #         },
        #         "soft_stop": {
        #             "triggered": soft_stop_triggered,
        #             "reasons": soft_stop_reasons
        #         },
        #         "metrics_record": self._metrics_record,
        #         "current_iteration": current_iteration,
        #         "max_iterations": max_iterations
        #     }
        # }
        # ============================================================================
        
        # Simplified decision logic: always continue unless max_iterations reached
        if current_iteration >= max_iterations - 1:
            should_continue = False
            reason = f"Reached maximum iteration limit ({max_iterations})"
        else:
            should_continue = True
            reason = f"Continuing iteration (current: {current_iteration + 1}/{max_iterations})"
        
        iteration_decision = {
            "continue": should_continue,
            "reason": reason,
            "analysis": {
                "current_iteration": current_iteration,
                "max_iterations": max_iterations,
                "note": "Simplified decision logic: always continue unless max_iterations reached"
            }
        }
        
        self.logger.info(f"Iteration decision: {'CONTINUE' if should_continue else 'STOP'} - {reason}")
        return iteration_decision
    
    def _update_metrics_record(
        self,
        simulation_results: Optional[Dict[str, Any]],
        current_iteration: int
    ) -> Dict[str, Any]:
        """
        Update metrics record based on current simulation results.
        
        Logic:
        1. If current metrics not empty and history empty -> LLM analysis to establish baseline
        2. If metrics record exists and fields match -> compare and update
        3. If metrics record exists but fields differ -> LLM analysis to update primary metrics

        Returns:
            Dictionary with update status and details
        """
        update_info = {
            "action": "none",
            "details": ""
        }
        
        if not simulation_results:
            update_info["details"] = "No simulation results available"
            return update_info
        
        # Extract current metrics
        current_metrics = simulation_results.get("simulation_metrics", {})
        
        # Also try alternative paths
        if not current_metrics:
            sim_output = simulation_results.get("simulation_output", {})
            eval_results = sim_output.get("evaluation_results_on_validation", {})
            current_metrics = eval_results.get("metrics", {})
        
        if not current_metrics:
            update_info["details"] = "No metrics in simulation results"
            return update_info
        
        # Store current metrics in history
        self._metrics_history.append(dict(current_metrics))
        
        if self._metrics_record is None:
            # Case 1: First time seeing metrics - establish baseline with LLM analysis
            update_info["action"] = "initialize"
            update_info["details"] = "Initializing metrics record with LLM analysis"
            
            self._metrics_record = self._analyze_metrics_with_llm(current_metrics, None)
            self._metrics_record["current_values"] = dict(current_metrics)
            self._metrics_record["iteration"] = current_iteration
            
        else:
            # Check if fields match
            current_keys = set(current_metrics.keys())
            record_keys = set(self._metrics_record.get("current_values", {}).keys())
            
            if current_keys == record_keys:
                # Case 2: Fields match - compare and update
                update_info["action"] = "update"
                update_info["details"] = "Updating metrics record with new values"
                
                # Calculate improvements
                improvements = self._calculate_improvements(
                    self._metrics_record["current_values"],
                    current_metrics,
                    self._metrics_record.get("metric_directions", {})
                )
                
                self._improvement_history.append(improvements)
                self._metrics_record["previous_values"] = self._metrics_record["current_values"]
                self._metrics_record["current_values"] = dict(current_metrics)
                self._metrics_record["latest_improvements"] = improvements
                self._metrics_record["iteration"] = current_iteration
                
            else:
                # Case 3: Fields differ - need LLM analysis to update primary metrics
                update_info["action"] = "reanalyze"
                update_info["details"] = f"Metrics fields changed, reanalyzing. Old: {record_keys}, New: {current_keys}"
                
                # Keep old primary metrics info but reanalyze
                old_primary = self._metrics_record.get("primary_metrics", [])
                new_record = self._analyze_metrics_with_llm(current_metrics, old_primary)
                
                # Preserve history where possible
                if self._metrics_record.get("current_values"):
                    new_record["previous_values"] = self._metrics_record["current_values"]
                
                new_record["current_values"] = dict(current_metrics)
                new_record["iteration"] = current_iteration
                self._metrics_record = new_record
        
        return update_info
    
    def _analyze_metrics_with_llm(
        self,
        metrics: Dict[str, float],
        existing_primary: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Use LLM to analyze metrics and determine primary metrics and directions.

        Args:
            metrics: Current metrics dictionary
            existing_primary: Existing primary metrics (if any) to consider

        Returns:
            Dictionary with primary_metrics, metric_directions, etc.
        """
        # Build prompt for LLM analysis
        metrics_str = json.dumps(metrics, indent=2)
        existing_primary_str = json.dumps(existing_primary) if existing_primary else "None"
        
        prompt = f"""Analyze the following simulation metrics and determine:
1. Which metrics are PRIMARY (most important for evaluating simulation quality)
2. For each metric, whether lower_is_better or higher_is_better

Current metrics:
{metrics_str}

Previously identified primary metrics (may need updating): {existing_primary_str}

Output your analysis in the following JSON format:
{{
  "primary_metrics": ["metric1", "metric2"],
  "metric_directions": {{
    "metric1": "lower_is_better",
    "metric2": "higher_is_better"
  }},
  "analysis_notes": "Brief explanation of why these are primary metrics"
}}

IMPORTANT: 
- Primary metrics should be the 2-4 most important metrics for evaluating overall simulation quality
- Common patterns: loss/error metrics are lower_is_better, accuracy/score metrics are higher_is_better
- Output ONLY valid JSON, no additional text
"""
        
        try:
            response = self._call_llm(prompt)
            
            # Parse JSON response
            json_str = self._extract_json_from_response(response)
            analysis = json.loads(json_str)
            
            # Validate structure
            if "primary_metrics" not in analysis:
                analysis["primary_metrics"] = list(metrics.keys())[:2]
            if "metric_directions" not in analysis:
                analysis["metric_directions"] = self._infer_metric_directions(metrics)
            
            self.logger.info(f"LLM metrics analysis: {len(analysis['primary_metrics'])} primary metrics identified")
            return analysis
            
        except Exception as e:
            self.logger.warning(f"LLM metrics analysis failed: {e}, using heuristics")
            # Fallback to heuristic analysis
            return self._heuristic_metrics_analysis(metrics, existing_primary)
    
    def _heuristic_metrics_analysis(
        self,
        metrics: Dict[str, float],
        existing_primary: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Fallback heuristic analysis when LLM is not available.
        """
        # Infer directions based on metric names
        directions = self._infer_metric_directions(metrics)
        
        # Determine primary metrics
        if existing_primary:
            # Keep existing primaries that still exist
            primary = [m for m in existing_primary if m in metrics]
        else:
            primary = []
        
        # If not enough primary metrics, add based on common patterns
        if len(primary) < 2:
            priority_patterns = ['loss', 'error', 'rmse', 'mae', 'accuracy', 'score', 'f1', 'auc']
            for pattern in priority_patterns:
                for metric_name in metrics.keys():
                    if pattern in metric_name.lower() and metric_name not in primary:
                        primary.append(metric_name)
                        if len(primary) >= 3:
                            break
                if len(primary) >= 3:
                    break
        
        # If still not enough, just take first few
        if len(primary) < 2:
            for metric_name in metrics.keys():
                if metric_name not in primary:
                    primary.append(metric_name)
                    if len(primary) >= 2:
                        break
        
        return {
            "primary_metrics": primary,
            "metric_directions": directions,
            "analysis_notes": "Heuristic analysis based on metric naming patterns"
        }
    
    def _infer_metric_directions(self, metrics: Dict[str, float]) -> Dict[str, str]:
        """
        Infer whether each metric is lower_is_better or higher_is_better based on name.
        """
        directions = {}
        lower_patterns = ['loss', 'error', 'rmse', 'mae', 'mse', 'distance', 'divergence', 'cost']
        higher_patterns = ['accuracy', 'score', 'f1', 'auc', 'precision', 'recall', 'r2', 'correlation']
        
        for metric_name in metrics.keys():
            name_lower = metric_name.lower()
            
            is_lower = any(p in name_lower for p in lower_patterns)
            is_higher = any(p in name_lower for p in higher_patterns)
            
            if is_lower and not is_higher:
                directions[metric_name] = "lower_is_better"
            elif is_higher and not is_lower:
                directions[metric_name] = "higher_is_better"
            else:
                # Default assumption: lower is better (common for loss-type metrics)
                directions[metric_name] = "lower_is_better"
        
        return directions
    
    def _calculate_improvements(
        self,
        previous: Dict[str, float],
        current: Dict[str, float],
        directions: Dict[str, str]
    ) -> Dict[str, float]:
        """
        Calculate improvement rates for each metric.
        
        Returns:
            Dictionary mapping metric names to improvement rates (positive = improved)
        """
        improvements = {}
        
        for metric_name in current.keys():
            if metric_name not in previous:
                continue
            
            prev_val = previous[metric_name]
            curr_val = current[metric_name]
            direction = directions.get(metric_name, "lower_is_better")
            
            if prev_val == 0:
                if curr_val == 0:
                    improvements[metric_name] = 0.0
                else:
                    # Large change from zero
                    improvements[metric_name] = 1.0 if (direction == "higher_is_better" and curr_val > 0) or \
                                                       (direction == "lower_is_better" and curr_val < 0) else -1.0
            else:
                relative_change = (curr_val - prev_val) / abs(prev_val)
                
                # Convert to improvement (positive = better)
                if direction == "lower_is_better":
                    improvements[metric_name] = -relative_change  # Decrease is improvement
                else:
                    improvements[metric_name] = relative_change  # Increase is improvement
        
        return improvements
    
    def _update_issue_history(
        self,
        feedback: Optional[Dict[str, Any]],
        current_iteration: int
    ) -> Dict[str, Any]:
        """
        Update issue history from feedback.
        
        Returns:
            Summary of current issues
        """
        summary = {
            "total_issues": 0,
            "blocker_count": 0,
            "high_count": 0,
            "medium_count": 0,
            "low_count": 0,
            "from_user_feedback_count": 0,
            "new_issues": 0,
            "persist_issues": 0
        }
        
        if not feedback:
            self._issue_history.append(summary)
            return summary
        
        # Count issues by severity
        for issue_id, issue_data in feedback.items():
            if not isinstance(issue_data, dict):
                continue
            
            summary["total_issues"] += 1
            
            severity = issue_data.get("severity", "medium").lower()
            if severity == "blocker":
                summary["blocker_count"] += 1
            elif severity == "high":
                summary["high_count"] += 1
            elif severity == "medium":
                summary["medium_count"] += 1
            else:
                summary["low_count"] += 1
            
            if issue_data.get("from_user_feedback", False):
                summary["from_user_feedback_count"] += 1
        
        # Note: new_issues and persist_issues would need to be tracked through playbook events
        # For now, we approximate based on feedback structure
        summary["new_issues"] = summary["total_issues"]  # Simplified: all feedback issues are "new" in this round
        
        self._issue_history.append(summary)
        return summary
    
    def _check_hard_guardrails(
        self,
        simulation_results: Optional[Dict[str, Any]],
        feedback: Optional[Dict[str, Any]],
        playbook: Optional[Dict[str, Any]],
        current_iteration: int
    ) -> Tuple[bool, List[str]]:
        """
        Check hard guardrails that prevent stopping.
        
        Returns:
            Tuple of (blocked: bool, reasons: List[str])
        """
        blocked = False
        reasons = []
        
        # 1. Check for blocker/high severity unresolved issues in playbook
        if playbook:
            strategies = playbook.get("strategies", {})
            for strategy_id, entry in strategies.items():
                meta_info = entry.get("meta_info", {})
                status = meta_info.get("status", "open")
                
                # Only check unresolved (non-resolved) strategies
                if status in ["open", "queued", "in_progress"]:
                    reflection = entry.get("reflection", {})
                    severity = reflection.get("severity", "medium").lower()
                    from_user = reflection.get("from_user_feedback", False)
                    
                    if severity == "blocker":
                        blocked = True
                        if from_user:
                            reasons.append(f"Blocker issue from user feedback: {strategy_id}")
                        else:
                            reasons.append(f"Blocker issue: {strategy_id}")
                    elif severity == "high":
                        blocked = True
                        reasons.append(f"High severity issue: {strategy_id}")
        
        # Also check current feedback for new blockers/high
        if feedback:
            for issue_id, issue_data in feedback.items():
                if not isinstance(issue_data, dict):
                    continue
                severity = issue_data.get("severity", "medium").lower()
                from_user = issue_data.get("from_user_feedback", False)
                
                if severity == "blocker":
                    blocked = True
                    if from_user:
                        reasons.append(f"New blocker from user feedback: {issue_id}")
                    else:
                        reasons.append(f"New blocker issue: {issue_id}")
                elif severity == "high":
                    blocked = True
                    reasons.append(f"New high severity issue: {issue_id}")
        
        # 2. Check run stage - not "ran_with_metrics" means crash/incomplete
        if simulation_results:
            run_stage = simulation_results.get("run_stage", "")
            if run_stage and run_stage != "ran_with_metrics":
                blocked = True
                reasons.append(f"Simulation did not complete successfully: run_stage={run_stage}")
        
        # 3. Check for metric regression
        if self._metrics_record and "latest_improvements" in self._metrics_record:
            primary_metrics = self._metrics_record.get("primary_metrics", [])
            improvements = self._metrics_record.get("latest_improvements", {})
            
            for metric_name in primary_metrics:
                if metric_name in improvements:
                    improvement = improvements[metric_name]
                    if improvement < -self.REGRESSION_THRESHOLD:
                        blocked = True
                        reasons.append(f"Metric regression: {metric_name} worsened by {abs(improvement)*100:.1f}%")
        
        return blocked, reasons
    
    def _check_soft_stop_conditions(
        self,
        current_iteration: int
    ) -> Tuple[bool, List[str]]:
        """
        Check soft stop conditions.
        
        Returns:
            Tuple of (triggered: bool, reasons: List[str])
        """
        triggered = False
        reasons = []
        
        # Need at least 2 rounds of history for plateau/low-issues detection
        if current_iteration < self.PLATEAU_CONSECUTIVE_ROUNDS:
            return False, ["Not enough iterations for soft stop evaluation"]
        
        # Condition 1: Plateau detection
        plateau_detected = self._check_plateau_condition()
        if plateau_detected:
            triggered = True
            reasons.append("Plateau: Primary metrics improvement < 3% for 2 consecutive rounds")
        
        # Condition 2: Low-issues condition
        low_issues_detected = self._check_low_issues_condition()
        if low_issues_detected:
            triggered = True
            reasons.append(f"Low-issues: All issues are low severity and count <= {self.MAX_LOW_ISSUES} for 2 rounds")
        
        return triggered, reasons
    
    def _check_plateau_condition(self) -> bool:
        """
        Check if we're in a plateau: 2 consecutive rounds with:
        - Primary metrics improvement < 3%
        - No single primary metric improved > 5%
        """
        if len(self._improvement_history) < self.PLATEAU_CONSECUTIVE_ROUNDS:
            return False
        
        if not self._metrics_record:
            return False
        
        primary_metrics = self._metrics_record.get("primary_metrics", [])
        if not primary_metrics:
            return False
        
        # Check last 2 rounds
        plateau_rounds = 0
        for i in range(-self.PLATEAU_CONSECUTIVE_ROUNDS, 0):
            if abs(i) > len(self._improvement_history):
                return False
            
            improvements = self._improvement_history[i]
        
            # Check if any primary metric improved significantly
            significant_improvement = False
            all_below_threshold = True
            
            for metric in primary_metrics:
                if metric in improvements:
                    imp = improvements[metric]
                    if imp >= self.SIGNIFICANT_IMPROVEMENT:
                        significant_improvement = True
                    if imp >= self.IMPROVEMENT_THRESHOLD:
                        all_below_threshold = False
            
            # Plateau round: no significant improvement AND all below threshold
            if not significant_improvement and all_below_threshold:
                plateau_rounds += 1
        
        return plateau_rounds >= self.PLATEAU_CONSECUTIVE_ROUNDS
    
    def _check_low_issues_condition(self) -> bool:
        """
        Check if we have low-issues condition: 2 consecutive rounds with:
        - All new+persist issues are low severity
        - Issue count <= MAX_LOW_ISSUES
        """
        if len(self._issue_history) < self.PLATEAU_CONSECUTIVE_ROUNDS:
            return False
        
        low_issue_rounds = 0
        for i in range(-self.PLATEAU_CONSECUTIVE_ROUNDS, 0):
            if abs(i) > len(self._issue_history):
                return False
            
            summary = self._issue_history[i]
            
            # Check if all issues are low severity
            has_high_severity = (
                summary.get("blocker_count", 0) > 0 or
                summary.get("high_count", 0) > 0 or
                summary.get("medium_count", 0) > 0
            )
            
            total_issues = summary.get("total_issues", 0)
            
            if not has_high_severity and total_issues <= self.MAX_LOW_ISSUES:
                low_issue_rounds += 1
        
        return low_issue_rounds >= self.PLATEAU_CONSECUTIVE_ROUNDS
    
    def _extract_json_from_response(self, response: str) -> str:
        """Extract JSON from LLM response, handling markdown code blocks."""
        # Try to find JSON in code blocks
        import re
        
        # Try ```json ... ``` first
        json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if json_match:
            return json_match.group(1).strip()
        
        # Try ``` ... ``` 
        code_match = re.search(r'```\s*(.*?)\s*```', response, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()
        
        # Try to find raw JSON object
        brace_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
        if brace_match:
            return brace_match.group(0)
        
        # Return as-is and let JSON parser handle it
        return response.strip()
    
    def _create_default_decision(self, current_iteration: int, max_iterations: int, auto_mode: bool = True) -> Dict[str, Any]:
        """
        Create a default iteration decision based on the iteration count.
        Kept for backward compatibility.
        """
        continue_iteration = current_iteration < max_iterations - 1

        decision = {
            "continue": continue_iteration,
            "reason": "Default decision based on iteration count",
            "analysis": {
                "metrics_update": {"action": "none", "details": "Default decision"},
                "issue_summary": {},
                "hard_guardrails": {"blocked": False, "reasons": []},
                "soft_stop": {"triggered": False, "reasons": []}
            }
        }
        
        return decision
