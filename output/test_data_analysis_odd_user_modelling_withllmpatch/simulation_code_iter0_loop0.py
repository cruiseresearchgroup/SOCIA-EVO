#!/usr/bin/env python3
"""
simulate.py

Production-grade, end-to-end executable Python program to simulate multi-user product
rating (1–5 stars) and review generation for an e-commerce platform, with automatic
parameter calibration on a training split and evaluation on a held-out validation split.

Key capabilities:
- Data ingestion from environment-configured absolute paths or CLI --data_dir
- Agent-based pipeline: DataIndexer → PersonaProfiler → ItemProfiler → PlanComposer → ReviewAuthor → StarRater → QAConsistency → Evaluator
- Information propagation via user neighbor influence and item reputation broadcast
- Temporal or random holdout (train/validation split)
- Pluggable parameter tuner (random search) for calibration
- LLM integration (OpenAI Responses API, model "gpt-5") for review generation with retries and caching; deterministic mock available via --mock_llm
- Deterministic behavior via fixed random seed
- Metrics: RMSE/MAE on stars, Jaccard and optional semantic text similarity, sentiment agreement, aspect coverage, consistency score, length deviation
- Outputs: calibrated_parameters.json, simulation_traces.jsonl, evaluation_metrics.json, ablation_report.json

Usage:
  python simulate.py --seed 42 --interactions_file interactions.csv

Environment variables:
  - PROJECT_ROOT: Absolute project root path (optional)
  - DATA_PATH: Relative data directory within project root (optional)
  - OPENAI_API_KEY: API key for OpenAI Responses API

Notes:
  - This program requires a valid OpenAI API key to generate reviews on the validation
    dataset via the LLM. You can enable a deterministic mock generator for development
    using the --mock_llm flag (only for non-production testing).
"""

import argparse
import csv
import datetime
import json
import logging
import math
import os
import random
import re
import statistics
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Optional import handling for OpenAI. Will validate at call time.
try:
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    OpenAI = None  # type: ignore

# Optional semantic similarity
try:
    from sentence_transformers import SentenceTransformer  # type: ignore
    import numpy as np  # type: ignore
    _HAS_ST = True
except Exception:
    _HAS_ST = False


# -------------------------- Path handling utilities --------------------------

def _get_default_data_dir() -> str:
    """
    Determine default data directory using environment variables with robust fallbacks.
    """
    project_root = os.environ.get("PROJECT_ROOT")
    data_path = os.environ.get("DATA_PATH")
    if project_root and data_path:
        base = project_root
        rel = data_path
    elif data_path and os.path.isabs(data_path):
        return data_path
    else:
        base = os.getcwd()
        rel = data_path or "data"
    return os.path.abspath(os.path.join(base, rel))


DATA_DIR = _get_default_data_dir()


# -------------------------- OpenAI API Utilities --------------------------


def get_openai_api_key() -> str:
    """
    Fetch the OpenAI API key from the environment.

    Returns:
        str: The API key string.

    Raises:
        ValueError: If OPENAI_API_KEY is not found in environment variables.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        return api_key
    raise ValueError("OpenAI API key not found in environment")


def call_gpt5_with_responses_api(
    prompt: str,
    model: str = "gpt-5",
    max_output_tokens: int = 4000,
    temperature: Optional[float] = None,
    retries: int = 3,
    backoff_base: float = 0.8
) -> str:
    """
    Invoke the OpenAI Responses API to generate text using the specified model.

    The function builds a call to client.responses.create(), as required by the
    task specification, and extracts text output from the response object.

    Args:
        prompt (str): Prompt text to send to the model.
        model (str): Model identifier (default: "gpt-5").
        max_output_tokens (int): Maximum output token budget.
        temperature (Optional[float]): Sampling temperature if supported by API.
        retries (int): Number of automatic retries on transient failures.
        backoff_base (float): Base for exponential backoff (seconds).

    Returns:
        str: The generated text content.

    Raises:
        ImportError: If the openai package is not installed.
        ValueError: If the API key is missing.
        RuntimeError: If the API call fails after retries.
    """
    if OpenAI is None:
        raise ImportError(
            "The 'openai' package is required to call the LLM. "
            "Install with: pip install openai>=1.0.0"
        )

    api_key = get_openai_api_key()
    client = OpenAI(api_key=api_key)

    responses_kwargs: Dict[str, Any] = {
        "model": model,
        "input": [
            {"role": "user", "content": [{"type": "input_text", "text": prompt}]}
        ],
        "max_output_tokens": int(max_output_tokens),
    }
    if temperature is not None:
        responses_kwargs["temperature"] = float(temperature)

    last_err: Optional[Exception] = None
    for attempt in range(retries):
        try:
            resp = client.responses.create(**responses_kwargs)

            def extract_response(resp_obj: Any) -> str:
                """
                Extract text content from an OpenAI Responses API response object.

                Args:
                    resp_obj (Any): The response object returned by client.responses.create().

                Returns:
                    str: The extracted text, or stringified fallback.
                """
                # Preferred attribute
                if hasattr(resp_obj, "output_text") and isinstance(resp_obj.output_text, str):
                    return resp_obj.output_text
                # General fallback
                try:
                    output = getattr(resp_obj, "output", None)
                    if output and isinstance(output, list) and len(output) > 0:
                        first = output[0]
                        if isinstance(first, dict):
                            content = first.get("content")
                            if isinstance(content, list) and len(content) > 0:
                                text = content[0].get("text")
                                if isinstance(text, str):
                                    return text
                except Exception:
                    pass
                return str(resp_obj)

            return extract_response(resp)
        except Exception as e:
            last_err = e
            sleep_s = backoff_base * (2 ** attempt) + random.random() * 0.1
            logging.warning("OpenAI API call failed on attempt %d/%d: %s. Retrying in %.2fs.",
                            attempt + 1, retries, e, sleep_s)
            time.sleep(sleep_s)

    raise RuntimeError(f"OpenAI API call failed after {retries} attempts: {last_err}")


# ------------------------------ Utilities ---------------------------------


def set_global_determinism(seed: int) -> None:
    """
    Set deterministic behavior across Python's random module.

    Args:
        seed (int): Random seed.
    """
    random.seed(seed)


def ensure_dir(path: str) -> None:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        path (str): Directory path.

    Raises:
        OSError: If directory creation fails.
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def read_csv_dicts(file_path: str) -> List[Dict[str, str]]:
    """
    Read a CSV file into a list of dictionaries (string values).

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        List[Dict[str, str]]: List of row dictionaries with string values.

    Raises:
        FileNotFoundError: If file does not exist.
    """
    # Accept relative paths by resolving against DATA_DIR
    if not os.path.isabs(file_path):
        file_path = os.path.join(DATA_DIR, file_path)

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Data file not found: {file_path}. Ensure the file exists, "
            f"pass an absolute path, or place it under data dir: {DATA_DIR}."
        )

    rows: List[Dict[str, str]] = []
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"No header found in CSV: {file_path}")
        for row in reader:
            rows.append({k: (v if v is not None else "") for k, v in row.items()})
    return rows


def to_float(value: Any, default: float = 0.0) -> float:
    """
    Convert a value to float.

    Args:
        value (Any): Input value.
        default (float): Fallback if conversion fails.

    Returns:
        float: Parsed float.
    """
    try:
        return float(value)
    except Exception:
        return default


def parse_optional_float(value: Any) -> Optional[float]:
    """
    Parse a value to float, returning None on failure.

    Args:
        value (Any): Input value.

    Returns:
        Optional[float]: Parsed float or None.
    """
    try:
        if value is None:
            return None
        s = str(value).strip()
        if s == "":
            return None
        return float(s)
    except Exception:
        return None


def to_int(value: Any, default: int = 0) -> int:
    """
    Convert a value to int.

    Args:
        value (Any): Input value.
        default (int): Fallback if conversion fails.

    Returns:
        int: Parsed int.
    """
    try:
        return int(value)
    except Exception:
        try:
            return int(float(value))
        except Exception:
            return default


def parse_date(value: str) -> Optional[datetime.datetime]:
    """
    Parse a date/time string into a datetime object.

    Args:
        value (str): Date string.

    Returns:
        Optional[datetime.datetime]: Parsed datetime or None if parsing fails.
    """
    if not value or not isinstance(value, str):
        return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S"):
        try:
            return datetime.datetime.strptime(value.strip(), fmt)
        except Exception:
            continue
    return None


def clamp(v: float, lo: float, hi: float) -> float:
    """
    Clamp a float value to [lo, hi].

    Args:
        v (float): Value.
        lo (float): Lower bound.
        hi (float): Upper bound.

    Returns:
        float: Clamped value.
    """
    return max(lo, min(hi, v))


def sigmoid(x: float) -> float:
    """
    Numerically stable sigmoid.

    Args:
        x (float): Input.

    Returns:
        float: Sigmoid(x) in [0,1].
    """
    if x >= 0:
        z = math.exp(-x)
        return 1 / (1 + z)
    else:
        z = math.exp(x)
        return z / (1 + z)


def tokenize(text: str) -> List[str]:
    """
    Tokenize a text into lowercase word tokens.

    Args:
        text (str): Input text.

    Returns:
        List[str]: List of tokens.
    """
    text = text.lower()
    tokens = re.findall(r"[a-zA-Z]+", text)
    return tokens


def jaccard_similarity(a_tokens: List[str], b_tokens: List[str]) -> float:
    """
    Compute Jaccard similarity between two token sets.

    Args:
        a_tokens (List[str]): Tokens of first text.
        b_tokens (List[str]): Tokens of second text.

    Returns:
        float: Jaccard similarity in [0,1].
    """
    set_a = set(a_tokens)
    set_b = set(b_tokens)
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / float(len(set_a | set_b))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:  # type: ignore
    """
    Compute cosine similarity between two 1-D vectors.

    Args:
        a (np.ndarray): Vector a.
        b (np.ndarray): Vector b.

    Returns:
        float: Cosine similarity.
    """
    num = float((a * b).sum())
    den = float((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12)
    return num / den


# --------------------------- Data Definitions ------------------------------


@dataclass
class InteractionRecord:
    """
    Single interaction record between a user and an item.

    Attributes:
        user_id (str): Identifier of the user.
        item_id (str): Identifier of the item.
        stars (float): Ground-truth star rating (1–5).
        review (str): Ground-truth review text.
        datatype (str): Split label (e.g., 'train', 'test') or empty if unspecified.
        timestamp (Optional[datetime.datetime]): Interaction timestamp, if provided.
        extra (Dict[str, Any]): Additional fields from the CSV rows, preserved.
    """
    user_id: str
    item_id: str
    stars: float
    review: str
    datatype: str
    timestamp: Optional[datetime.datetime]
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UserProfile:
    """
    Optional user profile.

    Attributes:
        user_id (str): User identifier.
        average_stars (Optional[float]): User-reported or aggregated average stars if available.
        friends (List[str]): Friend user IDs (for peer influence).
        extra (Dict[str, Any]): Additional available attributes.
    """
    user_id: str
    average_stars: Optional[float] = None
    friends: List[str] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ItemMetadata:
    """
    Optional item metadata.

    Attributes:
        item_id (str): Item identifier.
        category (Optional[str]): Category/domain tag.
        extra (Dict[str, Any]): Additional metadata fields.
    """
    item_id: str
    category: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


# ----------------------- Core Agent Implementations ------------------------


class SentimentTool:
    """
    Lightweight sentiment analyzer using a curated word lexicon.

    Provides:
    - Polarity score in [-1, 1]
    - Simple aspect detection via keyword matching around aspects
    """

    POSITIVE_WORDS = {
        "good", "great", "excellent", "amazing", "awesome", "love", "like", "nice",
        "fantastic", "perfect", "satisfied", "happy", "wonderful", "best", "positive",
        "recommend", "pleased", "smooth", "durable", "reliable", "fast"
    }
    NEGATIVE_WORDS = {
        "bad", "terrible", "awful", "hate", "poor", "disappointed", "slow", "worse",
        "worst", "broken", "boring", "annoying", "buggy", "fragile", "expensive",
        "cheap", "dirty", "damaged", "late", "noisy", "unhappy", "problem"
    }

    STOPWORDS = {
        "the", "and", "a", "an", "to", "for", "with", "of", "in", "on", "is", "it",
        "this", "that", "was", "were", "are", "be", "as", "at", "by", "from", "or",
        "i", "you", "he", "she", "they", "we", "but", "so", "if", "not"
    }

    @classmethod
    def polarity(cls, text: str) -> float:
        """
        Compute sentiment polarity score in [-1,1] based on lexicon counts.

        Args:
            text (str): Input text.

        Returns:
            float: Polarity score in [-1,1].
        """
        toks = tokenize(text)
        if not toks:
            return 0.0
        pos = sum(1 for t in toks if t in cls.POSITIVE_WORDS)
        neg = sum(1 for t in toks if t in cls.NEGATIVE_WORDS)
        score = pos - neg
        norm = pos + neg
        if norm == 0:
            return 0.0
        # Map to [-1, 1]
        return clamp(score / float(norm), -1.0, 1.0)

    @classmethod
    def aspect_sentiment(cls, text: str, aspects: List[str]) -> Dict[str, Dict[str, int]]:
        """
        Assess aspect mentions with sentiment counts.

        Args:
            text (str): Review text.
            aspects (List[str]): Aspect vocabulary.

        Returns:
            Dict[str, Dict[str, int]]: aspect -> {'pos': count, 'neg': count, 'total': count}
        """
        toks = tokenize(text)
        result: Dict[str, Dict[str, int]] = {}
        for aspect in aspects:
            a = aspect.lower()
            count = sum(1 for t in toks if t == a)
            # Sentiment in text overall as proxy (simple)
            pos = sum(1 for t in toks if t in cls.POSITIVE_WORDS)
            neg = sum(1 for t in toks if t in cls.NEGATIVE_WORDS)
            result[a] = {"pos": int(pos), "neg": int(neg), "total": int(count)}
        return result


class DataIndexer:
    """
    Builds indices and priors from interaction data, and serves as a memory bus.

    Responsibilities:
    - Group by user_id and item_id
    - Compute user and item priors (average stars, variance)
    - Provide aspect vocabulary and platform policy
    - Optional friend graph for neighbor influence
    - Simple embedding or tokenization utilities

    Attributes:
        interactions (List[InteractionRecord]): All parsed interaction records.
        users (Dict[str, List[InteractionRecord]]): User histories.
        items (Dict[str, List[InteractionRecord]]): Item histories.
        user_prior_mean (Dict[str, float]): User average stars.
        item_prior_mean (Dict[str, float]): Item average stars.
        item_prior_var (Dict[str, float]): Item star variance.
        aspect_vocab (List[str]): Generic product aspects.
        platform_policy_weight (float): Penalty weight for style violations in QA.
        user_profiles (Dict[str, UserProfile]): Optional profiles keyed by user_id.
        item_metadata (Dict[str, ItemMetadata]): Optional metadata keyed by item_id.
        cache_hits (int): Tracking cache hits for potential caching.
        last_refreshed_version (int): Versioning marker.
        embedding_store_refs (Dict[str, Any]): Placeholder for embedding store references.
    """

    DEFAULT_ASPECT_VOCAB = [
        "price", "quality", "durability", "design", "usability",
        "performance", "battery", "shipping", "packaging", "customer"
    ]

    def __init__(
        self,
        interactions: List[InteractionRecord],
        user_profiles: Optional[List[UserProfile]] = None,
        item_metadata: Optional[List[ItemMetadata]] = None,
        platform_policy_weight: float = 0.8
    ) -> None:
        """
        Initialize the DataIndexer.

        Args:
            interactions (List[InteractionRecord]): Interaction records.
            user_profiles (Optional[List[UserProfile]]): Optional user profiles.
            item_metadata (Optional[List[ItemMetadata]]): Optional item metadata.
            platform_policy_weight (float): Platform policy penalty weight in [0,1].
        """
        self.interactions = interactions
        self.users: Dict[str, List[InteractionRecord]] = {}
        self.items: Dict[str, List[InteractionRecord]] = {}
        self.user_prior_mean: Dict[str, float] = {}
        self.item_prior_mean: Dict[str, float] = {}
        self.item_prior_var: Dict[str, float] = {}
        self.aspect_vocab: List[str] = list(self.DEFAULT_ASPECT_VOCAB)
        self.platform_policy_weight = clamp(platform_policy_weight, 0.0, 1.0)
        self.user_profiles: Dict[str, UserProfile] = {}
        self.item_metadata: Dict[str, ItemMetadata] = {}
        self.cache_hits = 0
        self.last_refreshed_version = 0
        self.embedding_store_refs: Dict[str, Any] = {}

        # Aspect summary cache
        self._aspect_summary_cache: Dict[str, Dict[str, Dict[str, int]] = {}

        if user_profiles:
            self.user_profiles = {p.user_id: p for p in user_profiles}
        if item_metadata:
            self.item_metadata = {m.item_id: m for m in item_metadata}

        self._build_indices_and_priors()
        # Warm initial aspect cache
        self._warm_aspect_cache()

    def _build_indices_and_priors(self) -> None:
        """Create lookup maps and compute priors from interactions."""
        for rec in self.interactions:
            self.users.setdefault(rec.user_id, []).append(rec)
            self.items.setdefault(rec.item_id, []).append(rec)

        # Compute user priors
        for uid, recs in self.users.items():
            stars = [r.stars for r in recs if r.stars is not None]
            self.user_prior_mean[uid] = statistics.mean(stars) if stars else 3.0

        # Compute item priors
        for iid, recs in self.items.items():
            stars = [r.stars for r in recs if r.stars is not None]
            if stars:
                self.item_prior_mean[iid] = statistics.mean(stars)
                self.item_prior_var[iid] = statistics.pvariance(stars) if len(stars) > 1 else 0.0
            else:
                self.item_prior_mean[iid] = 3.0
                self.item_prior_var[iid] = 0.0

        self.last_refreshed_version += 1

    def rebuild_aspect_vocab_from_records(self, records: List[InteractionRecord], top_k: int = 12) -> None:
        """
        Rebuild aspect vocabulary from tokens in provided records (typically training split).

        Args:
            records (List[InteractionRecord]): Records to mine for aspect terms.
            top_k (int): Maximum vocabulary size.

        Notes:
            Fallbacks to default vocab if extraction is too sparse.
        """
        freq: Dict[str, int] = {}
        for r in records:
            toks = tokenize(r.review or "")
            for t in toks:
                if len(t) < 4:
                    continue
                if t in SentimentTool.STOPWORDS:
                    continue
                freq[t] = freq.get(t, 0) + 1
        # Choose top tokens that look like aspects
        terms = sorted(freq.items(), key=lambda kv: kv[1], reverse=True)[: top_k]
        candidates = [t for t, _ in terms]
        if candidates:
            self.aspect_vocab = candidates
            logging.info("Aspect vocab rebuilt from training data: %s", ", ".join(self.aspect_vocab))
        else:
            logging.info("Aspect vocab fell back to default due to sparse training data.")
            self.aspect_vocab = list(self.DEFAULT_ASPECT_VOCAB)
        # Invalidate and warm cache with new vocab
        self._aspect_summary_cache.clear()
        self._warm_aspect_cache()

    def _warm_aspect_cache(self) -> None:
        """Precompute aspect summaries per item for current aspect_vocab."""
        for iid in self.items.keys():
            self._aspect_summary_cache[iid] = self._compute_aspect_summary_for_item(iid)

    def _compute_aspect_summary_for_item(self, item_id: str) -> Dict[str, Dict[str, int]]:
        recs = self.items.get(item_id, [])
        summary: Dict[str, Dict[str, int]] = {a: {"pos": 0, "neg": 0, "total": 0} for a in self.aspect_vocab}
        for r in recs:
            aspects = SentimentTool.aspect_sentiment(r.review, self.aspect_vocab)
            for a, counts in aspects.items():
                summary[a]["pos"] += counts["pos"]
                summary[a]["neg"] += counts["neg"]
                summary[a]["total"] += counts["total"]
        return summary

    def get_user_prior(self, user_id: str) -> float:
        """
        Retrieve user prior average stars.

        Args:
            user_id (str): User identifier.

        Returns:
            float: User average stars, defaulting to global mean (3.0).
        """
        return self.user_prior_mean.get(user_id, 3.0)

    def get_item_prior(self, item_id: str) -> float:
        """
        Retrieve item prior average stars.

        Args:
            item_id (str): Item identifier.

        Returns:
            float: Item average stars, defaulting to global mean (3.0).
        """
        return self.item_prior_mean.get(item_id, 3.0)

    def get_user_friends(self, user_id: str) -> List[str]:
        """
        Get friend user IDs for neighbor influence.

        Args:
            user_id (str): User identifier.

        Returns:
            List[str]: List of friend user IDs, empty if not available.
        """
        prof = self.user_profiles.get(user_id)
        return prof.friends if prof else []

    def get_item_domain_tags(self, item_id: str) -> List[str]:
        """
        Retrieve domain tags for an item.

        Args:
            item_id (str): Item identifier.

        Returns:
            List[str]: Domain tags (e.g., category), if available.
        """
        meta = self.item_metadata.get(item_id)
        tags: List[str] = []
        if meta and meta.category:
            tags.append(meta.category.lower())
        return tags

    def sentiment(self, text: str) -> float:
        """
        Compute sentiment polarity for text.

        Args:
            text (str): Input text.

        Returns:
            float: Polarity score in [-1,1].
        """
        return SentimentTool.polarity(text)

    def aspect_summary_for_item(self, item_id: str) -> Dict[str, Dict[str, int]]:
        """
        Retrieve cached aspect sentiment summary for an item.

        Args:
            item_id (str): Item identifier.

        Returns:
            Dict[str, Dict[str, int]]: Aspect summary with pos/neg/total counts.
        """
        if item_id in self._aspect_summary_cache:
            self.cache_hits += 1
            return self._aspect_summary_cache[item_id]
        summary = self._compute_aspect_summary_for_item(item_id)
        self._aspect_summary_cache[item_id] = summary
        return summary

    def refresh(self) -> None:
        """Refresh caches and priors. Placeholder for future updates."""
        self.cache_hits = 0
        # In a dynamic ingestion scenario, recompute priors here.
        self.last_refreshed_version += 1


@dataclass
class PersonaState:
    """
    Dynamic persona state.

    Attributes:
        leniency_drift (float): Drift adjustment to user's leniency.
        aspect_preference_weights (Dict[str, float]): Aspect preference weights.
        style_vector (Dict[str, float]): Style descriptors (e.g., formality).
        recent_sentiment_bias (float): Recent sentiment bias from interactions.
    """
    leniency_drift: float = 0.0
    aspect_preference_weights: Dict[str, float] = field(default_factory=dict)
    style_vector: Dict[str, float] = field(default_factory=dict)
    recent_sentiment_bias: float = 0.0


class PersonaProfiler:
    """
    PersonaProfiler constructs user persona attributes and updates them over time.

    Static attributes estimated per user:
    - baseline_leniency
    - verbosity_prior
    - tone_style_prior
    - domain_familiarity

    Dynamic states updated via EMA or neighbor influence:
    - leniency_drift
    - aspect_preference_weights
    - style_vector
    - recent_sentiment_bias
    """

    def __init__(
        self,
        indexer: DataIndexer,
        leniency_drift_rate: float = 0.05,
        verbosity_scale: float = 1.0,
        neighbor_weight: float = 0.0,
        aspect_weight_decay: float = 0.9
    ) -> None:
        """
        Initialize PersonaProfiler.

        Args:
            indexer (DataIndexer): Shared data indexer.
            leniency_drift_rate (float): EMA rate for leniency drift.
            verbosity_scale (float): Scale factor for verbosity estimation.
            neighbor_weight (float): Weight for neighbor influence [0, 0.5].
            aspect_weight_decay (float): Decay factor for aspect preference updates.
        """
        self.indexer = indexer
        self.leniency_drift_rate = clamp(leniency_drift_rate, 0.0, 1.0)
        self.verbosity_scale = max(0.1, verbosity_scale)
        self.neighbor_weight = clamp(neighbor_weight, 0.0, 0.5)
        self.aspect_weight_decay = clamp(aspect_weight_decay, 0.0, 1.0)
        self._state_cache: Dict[str, PersonaState] = {}

    def _estimate_style_vector(self, user_id: str) -> Dict[str, float]:
        """
        Estimate style descriptors from user historical texts.

        Args:
            user_id (str): User identifier.

        Returns:
            Dict[str, float]: Style vector e.g., {'formality': 0.3, 'exuberance': 0.7}
        """
        recs = self.indexer.users.get(user_id, [])
        if not recs:
            return {"formality": 0.5, "exuberance": 0.5}
        # Heuristics: punctuation/exclamation signal exuberance; average sentence length for formality.
        total_sentences = 0
        total_words = 0
        exclaim_count = 0
        for r in recs:
            text = r.review or ""
            sentences = re.split(r"[.!?]+", text)
            total_sentences += max(1, len([s for s in sentences if s.strip()]))
            total_words += len(tokenize(text))
            exclaim_count += text.count("!")
        avg_sentence_len = total_words / max(1, total_sentences)
        formality = clamp(avg_sentence_len / 20.0, 0.0, 1.0)  # longer sentences => more formal
        exuberance = clamp(exclaim_count / max(1, len(recs)) / 3.0, 0.0, 1.0)
        return {"formality": formality, "exuberance": exuberance}

    def _estimate_verbosity(self, user_id: str) -> float:
        """
        Estimate verbosity prior from user history.

        Args:
            user_id (str): User identifier.

        Returns:
            float: Verbosity prior (approximate sentence count target).
        """
        recs = self.indexer.users.get(user_id, [])
        if not recs:
            return 3.0
        sentence_counts = []
        for r in recs:
            text = r.review or ""
            sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
            sentence_counts.append(len(sentences))
        return clamp(statistics.mean(sentence_counts) * self.verbosity_scale, 1.0, 8.0)

    def _neighbor_leniency(self, user_id: str) -> float:
        """
        Compute neighbor-averaged leniency for peer influence.

        Args:
            user_id (str): User identifier.

        Returns:
            float: Neighbor average stars (prior), default 3.0 if not available.
        """
        friends = self.indexer.get_user_friends(user_id)
        if not friends:
            return 3.0
        vals = [self.indexer.get_user_prior(f) for f in friends]
        if not vals:
            return 3.0
        return statistics.mean(vals)

    def profile(self, user_id: str) -> Tuple[float, float, Dict[str, float], List[str], PersonaState]:
        """
        Construct persona for the given user.

        Args:
            user_id (str): User identifier.

        Returns:
            Tuple containing:
                - baseline_leniency (float)
                - verbosity_prior (float)
                - tone_style_prior (dict)
                - domain_familiarity (list of tags)
                - PersonaState (state object)
        """
        baseline = self.indexer.get_user_prior(user_id)
        neighbor_mean = self._neighbor_leniency(user_id)
        # Incorporate neighbor influence
        baseline = (1 - self.neighbor_weight) * baseline + self.neighbor_weight * neighbor_mean

        verbosity_prior = self._estimate_verbosity(user_id)
        tone_style_prior = self._estimate_style_vector(user_id)
        # Domain familiarity: collect categories from user's items
        tags: List[str] = []
        for rec in self.indexer.users.get(user_id, []):
            tags.extend(self.indexer.get_item_domain_tags(rec.item_id))
        domain_familiarity = sorted(list(set(tags)))[:3]

        state = self._state_cache.get(user_id)
        if state is None:
            state = PersonaState(
                leniency_drift=0.0,
                aspect_preference_weights={a: 1.0 for a in self.indexer.aspect_vocab},
                style_vector=tone_style_prior.copy(),
                recent_sentiment_bias=0.0
            )
            self._state_cache[user_id] = state
        return baseline, verbosity_prior, tone_style_prior, domain_familiarity, state

    def update_from_interaction(self, user_id: str, sentiment: float) -> None:
        """
        Update user dynamic state with recent sentiment.

        Args:
            user_id (str): User identifier.
            sentiment (float): Polarity score in [-1,1].
        """
        state = self._state_cache.get(user_id)
        if not state:
            return
        # EMA update for recent sentiment bias
        state.recent_sentiment_bias = (
            (1 - self.leniency_drift_rate) * state.recent_sentiment_bias +
            self.leniency_drift_rate * sentiment
        )
        # Style vector modest adjustment
        state.style_vector["exuberance"] = clamp(
            state.style_vector.get("exuberance", 0.5) + 0.02 * sentiment, 0.0, 1.0
        )


@dataclass
class ItemState:
    """
    Dynamic item state.

    Attributes:
        freshness_score (float): Recency-weighted activity proxy.
        controversy (float): Rating variance proxy.
        aspect_confidence (float): Confidence in aspect summary.
    """
    freshness_score: float = 0.5
    controversy: float = 0.0
    aspect_confidence: float = 0.5


class ItemProfiler:
    """
    ItemProfiler aggregates item prior quality and aspect summary.

    Responsibilities:
    - quality_prior (mean/variance)
    - aspect_summary (pros/cons)
    - domain_tags
    - dynamic states (freshness, controversy, aspect confidence)
    """

    def __init__(
        self,
        indexer: DataIndexer,
        aspect_smoothing_alpha: float = 0.2,
        reputation_inertia: float = 0.85,
        min_reviews_for_confidence: int = 3
    ) -> None:
        """
        Initialize ItemProfiler.

        Args:
            indexer (DataIndexer): Shared indexer.
            aspect_smoothing_alpha (float): Smoothing for aspect counts.
            reputation_inertia (float): Inertia for reputation updates.
            min_reviews_for_confidence (int): Minimum reviews for high confidence.
        """
        self.indexer = indexer
        self.aspect_smoothing_alpha = clamp(aspect_smoothing_alpha, 0.0, 1.0)
        self.reputation_inertia = clamp(reputation_inertia, 0.0, 1.0)
        self.min_reviews_for_confidence = max(0, min_reviews_for_confidence)
        self._item_state: Dict[str, ItemState] = {}
        self._simulated_interactions_count: Dict[str, int] = {}

    def profile(self, item_id: str) -> Tuple[float, float, Dict[str, Dict[str, int]], List[str], ItemState]:
        """
        Build item profile.

        Args:
            item_id (str): Item identifier.

        Returns:
            Tuple of:
                - quality_prior mean (float)
                - quality variance (float)
                - aspect_summary (dict)
                - domain_tags (list)
                - ItemState
        """
        mean = self.indexer.get_item_prior(item_id)
        var = self.indexer.item_prior_var.get(item_id, 0.0)
        aspect_summary = self.indexer.aspect_summary_for_item(item_id)
        tags = self.indexer.get_item_domain_tags(item_id)
        state = self._item_state.get(item_id)
        if state is None:
            # Basic initialization
            controversy = var
            hist_len = len(self.indexer.items.get(item_id, []))
            conf = 1.0 if hist_len >= self.min_reviews_for_confidence else clamp(hist_len / max(1, self.min_reviews_for_confidence), 0.0, 1.0)
            state = ItemState(freshness_score=0.5, controversy=controversy, aspect_confidence=conf)
            self._item_state[item_id] = state
        return mean, var, aspect_summary, tags, state

    def update_from_interaction(self, item_id: str, stars: float) -> None:
        """
        Update dynamic item state (freshness, reputation inertia).

        Args:
            item_id (str): Item identifier.
            stars (float): Observed or simulated stars.
        """
        state = self._item_state.get(item_id)
        if not state:
            return
        # Freshness bump
        state.freshness_score = clamp(0.7 * state.freshness_score + 0.3, 0.0, 1.0)
        # Update controversy with simple inertia to variance proxy
        cur_var = self.indexer.item_prior_var.get(item_id, 0.0)
        state.controversy = (
            self.reputation_inertia * state.controversy +
            (1 - self.reputation_inertia) * cur_var
        )
        # Update aspect confidence based on interaction volume (including simulated)
        self._simulated_interactions_count[item_id] = self._simulated_interactions_count.get(item_id, 0) + 1
        hist_len = len(self.indexer.items.get(item_id, [])) + self._simulated_interactions_count[item_id]
        state.aspect_confidence = clamp(hist_len / max(1, self.min_reviews_for_confidence), 0.0, 1.0)


@dataclass
class PlanState:
    """
    PlanComposer dynamic state.

    Attributes:
        planned_aspects (List[str]): Selected aspects for the review.
        tone_targets (Dict[str, float]): Tone target vector.
        length_target (int): Target sentence count.
    """
    planned_aspects: List[str] = field(default_factory=list)
    tone_targets: Dict[str, float] = field(default_factory=dict)
    length_target: int = 3


class PlanComposer:
    """
    Composes aspect and tone plans for reviews.

    Parameters of interest:
        - aspect_topk (int): Number of aspects to include.
        - length_target_mean (float): Target sentence count mean.
        - ctx_merge_weight (float): Blend of persona vs item aspects [0.2, 0.8].
        - plan_diversity_temp (float): Softmax temperature for aspect selection diversity.
    """

    def __init__(
        self,
        indexer: DataIndexer,
        aspect_topk: int = 4,
        length_target_mean: float = 3.0,
        ctx_merge_weight: float = 0.5,
        plan_diversity_temp: float = 0.8
    ) -> None:
        """
        Initialize PlanComposer.

        Args:
            indexer (DataIndexer): Shared indexer.
            aspect_topk (int): Number of aspects to include.
            length_target_mean (float): Target sentence count mean.
            ctx_merge_weight (float): Blend weight persona vs item aspect signals.
            plan_diversity_temp (float): Softmax temperature for diverse selection.
        """
        self.indexer = indexer
        self.aspect_topk = max(1, int(round(aspect_topk)))
        self.length_target_mean = float(length_target_mean)
        self.ctx_merge_weight = clamp(ctx_merge_weight, 0.2, 0.8)
        self.plan_diversity_temp = clamp(plan_diversity_temp, 0.1, 2.0)

    @staticmethod
    def _weighted_sample_without_replacement(items: List[Tuple[str, float]], k: int) -> List[str]:
        """
        Sample k items without replacement in proportion to weights, deterministic via random module.

        Args:
            items (List[Tuple[str, float]]): (item, weight)
            k (int): number to sample

        Returns:
            List[str]: sampled item names
        """
        pool = items[:]
        selected: List[str] = []
        k = min(k, len(pool))
        for _ in range(k):
            total = sum(max(0.0, w) for _, w in pool) or 1.0
            r = random.random() * total
            cum = 0.0
            pick_idx = 0
            for idx, (name, w) in enumerate(pool):
                cum += max(0.0, w)
                if r <= cum:
                    pick_idx = idx
                    break
            selected.append(pool[pick_idx][0])
            del pool[pick_idx]
        return selected

    def compose_plan(
        self,
        user_state: PersonaState,
        tone_style_prior: Dict[str, float],
        item_aspect_summary: Dict[str, Dict[str, int]],
        persona_weights: Optional[Dict[str, float]] = None,
        verbosity_prior: float = 3.0
    ) -> PlanState:
        """
        Compose a plan: select aspects, tone targets, and length target.

        Args:
            user_state (PersonaState): User persona dynamic state.
            tone_style_prior (Dict[str, float]): Tone style prior.
            item_aspect_summary (Dict[str, Dict[str, int]]): Item aspect pros/cons.
            persona_weights (Optional[Dict[str, float]]): Persona aspect preferences.
            verbosity_prior (float): Verbosity signal (desired sentence count).

        Returns:
            PlanState: Structured plan for review author.
        """
        persona_weights = persona_weights or user_state.aspect_preference_weights
        # Compute item aspect weights: pos - neg as signal
        item_weights: Dict[str, float] = {}
        for a, counts in item_aspect_summary.items():
            item_weights[a] = counts.get("pos", 0) - counts.get("neg", 0)

        # Merge weights
        merged: List[Tuple[str, float]] = []
        for a in self.indexer.aspect_vocab:
            pv = persona_weights.get(a, 1.0)
            iv = item_weights.get(a, 0.0)
            score = self.ctx_merge_weight * pv + (1 - self.ctx_merge_weight) * iv
            merged.append((a, score))

        # Softmax with temperature
        max_score = max((s for _, s in merged), default=0.0)
        exp_scores = [(a, math.exp((s - max_score) / max(1e-6, self.plan_diversity_temp))) for a, s in merged]
        # Weighted sampling without replacement for diversity but deterministic given seed
        selected = self._weighted_sample_without_replacement(exp_scores, self.aspect_topk)

        # Tone targets: blend of persona style and user_state
        tone_targets = {
            "formality": clamp(0.5 * tone_style_prior.get("formality", 0.5) + 0.5 * user_state.style_vector.get("formality", 0.5), 0.0, 1.0),
            "exuberance": clamp(0.5 * tone_style_prior.get("exuberance", 0.5) + 0.5 * user_state.style_vector.get("exuberance", 0.5), 0.0, 1.0),
        }

        # Length target: round toward verbosity prior blended with length_target_mean
        length_target = int(round(clamp(
            0.6 * verbosity_prior + 0.4 * self.length_target_mean, 1.0, 10.0
        )))

        return PlanState(planned_aspects=selected, tone_targets=tone_targets, length_target=length_target)


class ReviewAuthor:
    """
    Generates review text following a structured plan and constraints.

    Parameters:
        - llm_temperature (float): Sampling temperature for LLM generation.
        - style_alignment_weight (float): Weight for persona/style conditioning.
        - max_revision_loops (int): Maximum automatic revision attempts.
        - mock_llm (bool): If True, uses a deterministic mock generator (development only).
    """

    def __init__(
        self,
        indexer: DataIndexer,
        llm_temperature: float = 0.5,
        style_alignment_weight: float = 0.7,
        max_revision_loops: int = 1,
        mock_llm: bool = False,
        llm_max_output_tokens: int = 400
    ) -> None:
        """
        Initialize ReviewAuthor.

        Args:
            indexer (DataIndexer): Shared indexer.
            llm_temperature (float): LLM sampling temperature in [0.2, 0.9].
            style_alignment_weight (float): Style alignment strength in [0.3, 1.0].
            max_revision_loops (int): Max number of auto-fix attempts if inconsistency is detected.
            mock_llm (bool): If True, uses a deterministic template instead of calling LLM.
            llm_max_output_tokens (int): Maximum tokens for LLM response.
        """
        self.indexer = indexer
        self.llm_temperature = clamp(llm_temperature, 0.2, 0.9)
        self.style_alignment_weight = clamp(style_alignment_weight, 0.3, 1.0)
        self.max_revision_loops = max(0, int(round(max_revision_loops)))
        # Keep deterministic mock for development pipelines without API access
        self.mock_llm = mock_llm
        self.llm_max_output_tokens = int(llm_max_output_tokens)
        self.last_generated_text_quality_score: float = 0.0
        self.revision_count = 0
        self._llm_cache: Dict[str, str] = {}

    def _build_prompt(
        self,
        user_id: str,
        item_id: str,
        plan: PlanState,
        persona_tone: Dict[str, float],
        domain_tags: List[str]
    ) -> str:
        """
        Construct the LLM prompt using user context, item context, and plan.

        Args:
            user_id (str): User identifier.
            item_id (str): Item identifier.
            plan (PlanState): Planning state with aspects and length target.
            persona_tone (Dict[str, float]): Persona tone/style vector.
            domain_tags (List[str]): Item domain tags.

        Returns:
            str: The prompt for the LLM.
        """
        user_prior = self.indexer.get_user_prior(user_id)
        item_prior = self.indexer.get_item_prior(item_id)
        platform_weight = self.indexer.platform_policy_weight
        aspects = ", ".join(plan.planned_aspects) if plan.planned_aspects else "overall experience"
        tone_desc = f"formality={persona_tone.get('formality', 0.5):.2f}, exuberance={persona_tone.get('exuberance', 0.5):.2f}"
        tags = ", ".join(domain_tags) if domain_tags else "general"
        length = plan.length_target

        style_guidance = (
            "Style alignment: "
            f"Adhere to persona tone by weight {self.style_alignment_weight:.2f} (higher -> more aligned); "
            "aim for clarity and helpfulness."
        )

        prompt = (
            "You are an honest customer writing a concise, specific product review.\n"
            f"User context:\n- historical leniency (avg stars): {user_prior:.2f}\n"
            f"- style preferences: {tone_desc}\n"
            f"{style_guidance}\n"
            f"Item context:\n- item reputation prior (avg stars): {item_prior:.2f}\n"
            f"- domain tags: {tags}\n"
            f"Plan:\n- key aspects to mention: {aspects}\n"
            f"- target length: about {length} sentences\n"
            "Constraints (platform policy):\n"
            "- stay respectful, no profanity, no personally identifiable information\n"
            "- avoid exaggeration; be factual and balanced\n"
            "- avoid repeating the same sentence\n"
            "- keep it helpful for other shoppers\n"
            f"- enforce style penalty weight={platform_weight:.2f}\n\n"
            "Write the review now. Use natural language and coherent flow.\n"
        )
        return prompt

    def _mock_generate(self, plan: PlanState, item_id: str) -> str:
        """
        Deterministic mock review generator (for development only).

        Args:
            plan (PlanState): Planning state.
            item_id (str): Item identifier.

        Returns:
            str: Generated mock review.
        """
        aspects = plan.planned_aspects or ["overall"]
        sentences = []
        for i in range(plan.length_target):
            a = aspects[i % len(aspects)]
            # Use item prior to modulate sentiment words
            ip = self.indexer.get_item_prior(item_id)
            if ip >= 4.0:
                phrase = f"The {a} is impressive and performs well for daily use."
            elif ip <= 2.5:
                phrase = f"The {a} is disappointing and needs significant improvement."
            else:
                phrase = f"The {a} is acceptable, with room for improvement."
            sentences.append(phrase)
        return " ".join(sentences)

    @staticmethod
    def _trim_to_sentence_count(text: str, target: int, tolerance: int = 2) -> str:
        """
        Trim text to approximately target sentences without reconstructing all punctuation.

        Args:
            text (str): Original text.
            target (int): Target sentence count.
            tolerance (int): Allowed extra sentences before trimming.

        Returns:
            str: Possibly trimmed text.
        """
        if target <= 0:
            return text
        # Split while keeping delimiters
        parts = re.split(r"([.!?])", text)
        sentences: List[str] = []
        current = ""
        for i in range(0, len(parts), 2):
            seg = parts[i].strip()
            punct = parts[i + 1] if i + 1 < len(parts) else "."
            if seg:
                sentences.append(seg + (punct if punct else "."))
        if len(sentences) <= target + tolerance:
            return text
        trimmed = "".join(sentences[: target + tolerance]).strip()
        return trimmed

    def generate(
        self,
        user_id: str,
        item_id: str,
        plan: PlanState,
        persona_tone: Dict[str, float],
        domain_tags: List[str]
    ) -> str:
        """
        Generate review text for a user-item event via LLM call (or mock).

        This method uses the OpenAI Responses API via client.responses.create() with model gpt-5.
        It requires OPENAI_API_KEY environment variable. If the call fails or mock_llm is True,
        a deterministic mock generator will be used as a fallback for development.

        Args:
            user_id (str): User identifier.
            item_id (str): Item identifier.
            plan (PlanState): Plan composed by PlanComposer.
            persona_tone (Dict[str, float]): Persona tone/style vector.
            domain_tags (List[str]): Item domain tags.

        Returns:
            str: Generated review text (primary source is the LLM).
        """
        prompt = self._build_prompt(user_id, item_id, plan, persona_tone, domain_tags)
        if self.mock_llm:
            text = self._mock_generate(plan, item_id)
            self.last_generated_text_quality_score = 0.8  # heuristic mock score
            self.revision_count = 0
            return text

        # Simple cache to reduce cost
        if prompt in self._llm_cache:
            llm_text = self._llm_cache[prompt]
        else:
            try:
                llm_text = call_gpt5_with_responses_api(
                    prompt=prompt,
                    model="gpt-5",
                    max_output_tokens=max(128, self.llm_max_output_tokens),
                    temperature=self.llm_temperature
                )
                self._llm_cache[prompt] = llm_text
            except Exception as e:
                logging.warning("LLM generation failed, falling back to mock generator: %s", e)
                text = self._mock_generate(plan, item_id)
                self.last_generated_text_quality_score = 0.6
                self.revision_count = 0
                return text

        # Optional post-processing: trim excessive sentences, avoid heavy reformatting
        processed = self._trim_to_sentence_count(llm_text, plan.length_target, tolerance=2).strip()
        if not processed:
            processed = self._mock_generate(plan, item_id)
        self.last_generated_text_quality_score = 0.9  # optimistic default
        self.revision_count = 0
        return processed


class StarRater:
    """
    Maps sentiment to a star rating using an ordinal/logistic-like link function
    with user and item bias priors.

    Parameters:
        - mapping_slope (float): Slope of the mapping [2.0, 8.0]
        - mapping_intercept (float): Intercept of the mapping [-2.0, 2.0]
        - user_bias_weight (float): Weight of user leniency prior [0.0, 1.0]
        - item_bias_weight (float): Weight of item reputation prior [0.0, 1.0]
        - uncertainty_scale (float): Noise scale [0.1, 1.0]
    """

    def __init__(
        self,
        indexer: DataIndexer,
        mapping_slope: float = 4.0,
        mapping_intercept: float = 0.0,
        user_bias_weight: float = 0.5,
        item_bias_weight: float = 0.5,
        uncertainty_scale: float = 0.2
    ) -> None:
        """
        Initialize StarRater.

        Args:
            indexer (DataIndexer): Shared indexer.
            mapping_slope (float): Slope parameter.
            mapping_intercept (float): Intercept parameter.
            user_bias_weight (float): Weight of user prior.
            item_bias_weight (float): Weight of item prior.
            uncertainty_scale (float): Gaussian noise scale.
        """
        self.indexer = indexer
        self.mapping_slope = clamp(mapping_slope, 2.0, 8.0)
        self.mapping_intercept = clamp(mapping_intercept, -2.0, 2.0)
        self.user_bias_weight = clamp(user_bias_weight, 0.0, 1.0)
        self.item_bias_weight = clamp(item_bias_weight, 0.0, 1.0)
        self.uncertainty_scale = clamp(uncertainty_scale, 0.1, 1.0)

    def rate(self, sentiment: float, user_id: str, item_id: str, add_noise: bool = True) -> float:
        """
        Map sentiment to star rating in [1,5].

        Args:
            sentiment (float): Polarity score in [-1,1].
            user_id (str): User identifier.
            item_id (str): Item identifier.
            add_noise (bool): Whether to add Gaussian noise.

        Returns:
            float: Star rating in [1,5] (not rounded).
        """
        # Logistic mapping to [1,5]
        x = self.mapping_slope * sentiment + self.mapping_intercept
        base = 1.0 + 4.0 * sigmoid(x)

        # Bias terms as deviations from neutral 3-star baseline
        user_prior = self.indexer.get_user_prior(user_id)
        item_prior = self.indexer.get_item_prior(item_id)
        bias_term = (self.user_bias_weight * (user_prior - 3.0)) + (self.item_bias_weight * (item_prior - 3.0))
        pred = base + bias_term

        if add_noise:
            noise = random.gauss(0.0, self.uncertainty_scale)
            pred += noise

        return clamp(pred, 1.0, 5.0)


class QAConsistency:
    """
    Checks rating–text consistency and decides on auto-revisions if required.

    Parameters:
        - consistency_threshold (float): Threshold in [0.6, 0.9] under which revisions trigger.
        - max_auto_fix_attempts (int): Maximum attempts to fix inconsistency (0–2).
        - penalty_weight_style_violations (float): Style violation penalty weight [0,1].
    """

    def __init__(
        self,
        indexer: DataIndexer,
        star_rater: StarRater,
        consistency_threshold: float = 0.75,
        max_auto_fix_attempts: int = 1,
        penalty_weight_style_violations: float = 0.5
    ) -> None:
        """
        Initialize QAConsistency.

        Args:
            indexer (DataIndexer): Shared indexer.
            star_rater (StarRater): Rater instance for internal consistency check.
            consistency_threshold (float): Threshold [0.6, 0.9].
            max_auto_fix_attempts (int): Max auto-fix attempts.
            penalty_weight_style_violations (float): Weight for style violations.
        """
        self.indexer = indexer
        self.star_rater = star_rater
        self.consistency_threshold = clamp(consistency_threshold, 0.6, 0.9)
        self.max_auto_fix_attempts = max(0, int(round(max_auto_fix_attempts)))
        self.penalty_weight_style_violations = clamp(penalty_weight_style_violations, 0.0, 1.0)
        self.consistency_score = 1.0
        self.auto_fix_attempts = 0

    def evaluate(self, text: str, stars: float, plan: PlanState) -> float:
        """
        Compute consistency score between text sentiment and stars, including style adherence.

        Args:
            text (str): Generated text.
            stars (float): Proposed stars.
            plan (PlanState): Plan for target length and aspects.

        Returns:
            float: Consistency score in [0,1].
        """
        sent = self.indexer.sentiment(text)
        implied_stars = 1.0 + 4.0 * sigmoid(self.star_rater.mapping_slope * sent + self.star_rater.mapping_intercept)
        rating_gap = abs(implied_stars - stars) / 4.0  # normalized
        rating_consistency = 1.0 - rating_gap

        # Style penalty: length deviation
        sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
        len_dev = abs(len(sentences) - plan.length_target) / max(1, plan.length_target)
        style_penalty = self.indexer.platform_policy_weight * len_dev
        score = clamp(rating_consistency - self.penalty_weight_style_violations * style_penalty, 0.0, 1.0)
        self.consistency_score = score
        return score

    def needs_revision(self, text: str, stars: float, plan: PlanState) -> bool:
        """
        Decide if auto-revision is necessary.

        Args:
            text (str): Generated text.
            stars (float): Proposed stars.
            plan (PlanState): Plan.

        Returns:
            bool: True if revision is needed.
        """
        score = self.evaluate(text, stars, plan)
        return score < self.consistency_threshold


# ----------------------- Tuning and Evaluation -----------------------------


@dataclass
class TunerConfig:
    """
    Configuration for the ParameterTuner.

    Attributes:
        num_trials (int): Number of random search trials.
        early_stop_patience (int): Early stop once best hasn't improved after this many trials.
        objective_weights (Dict[str, float]): Weights for star error, text mismatch, and consistency penalties.
    """
    num_trials: int = 12
    early_stop_patience: int = 5
    objective_weights: Dict[str, float] = field(default_factory=lambda: {"stars": 0.7, "text": 0.2, "consistency": 0.1})


class ParameterTuner:
    """
    Random search tuner for calibrating simulator parameters.

    It evaluates a lightweight objective on the training set, prioritizing rating
    error and consistency using ground-truth texts to avoid heavy LLM calls during tuning.
    """

    PARAM_BOUNDS: Dict[str, Tuple[str, Tuple[float, float]]] = {
        "neighbor_weight": ("float", (0.0, 0.5)),
        "ctx_merge_weight": ("float", (0.2, 0.8)),
        "aspect_topk": ("int", (3, 6)),
        "length_target_mean": ("float", (2.0, 6.0)),
        "plan_diversity_temp": ("float", (0.3, 1.2)),
        "llm_temperature": ("float", (0.2, 0.9)),
        "style_alignment_weight": ("float", (0.3, 1.0)),
        "mapping_slope": ("float", (2.0, 8.0)),
        "mapping_intercept": ("float", (-2.0, 2.0)),
        "user_bias_weight": ("float", (0.0, 1.0)),
        "item_bias_weight": ("float", (0.0, 1.0)),
        "uncertainty_scale": ("float", (0.1, 1.0)),
        "consistency_threshold": ("float", (0.6, 0.9)),
        "max_auto_fix_attempts": ("int", (0, 2)),
    }

    def __init__(self, indexer: DataIndexer, config: Optional[TunerConfig] = None) -> None:
        """
        Initialize ParameterTuner.

        Args:
            indexer (DataIndexer): Shared indexer.
            config (Optional[TunerConfig]): Tuner configuration.
        """
        self.indexer = indexer
        self.config = config or TunerConfig()
        self.current_params: Dict[str, Any] = {}
        self.best_params: Dict[str, Any] = {}
        self.history_of_trials: List[Dict[str, Any]] = []

    def _sample_param(self, name: str) -> Any:
        """Sample a random parameter value within bounds."""
        ptype, (lo, hi) = self.PARAM_BOUNDS[name]
        if ptype == "float":
            return random.uniform(lo, hi)
        elif ptype == "int":
            return random.randint(int(lo), int(hi))
        else:
            raise ValueError(f"Unknown parameter type for {name}")

    def _normalize_weights(self, w: Dict[str, float]) -> Dict[str, float]:
        """Normalize objective weights to sum to 1."""
        total = sum(max(0.0, float(v)) for v in w.values())
        if total <= 0:
            return {"stars": 1.0, "text": 0.0, "consistency": 0.0}
        return {k: max(0.0, float(v)) / total for k, v in w.items()}

    def _objective(
        self,
        train_records: List[InteractionRecord],
        params: Dict[str, Any]
    ) -> float:
        """
        Compute objective for given params on the training set.

        This objective uses:
        - rating MSE between predicted stars (from ground-truth text sentiment) and actual stars
        - consistency penalty between predicted stars and ground-truth text
        - text loss is set to 0 during tuning to avoid needing LLM

        Args:
            train_records (List[InteractionRecord]): Training interactions.
            params (Dict[str, Any]): Proposed parameters.

        Returns:
            float: Objective value (lower is better).
        """
        # Instantiate minimal components using proposed params
        persona = PersonaProfiler(
            self.indexer,
            neighbor_weight=float(params.get("neighbor_weight", 0.0))
        )
        item_prof = ItemProfiler(self.indexer)
        plan_comp = PlanComposer(
            self.indexer,
            aspect_topk=int(round(params.get("aspect_topk", 4))),
            length_target_mean=float(params.get("length_target_mean", 3.0)),
            ctx_merge_weight=float(params.get("ctx_merge_weight", 0.5)),
            plan_diversity_temp=float(params.get("plan_diversity_temp", 0.8)),
        )
        # For determinism during tuning, reduce uncertainty
        uncertainty_scale = min(0.2, float(params.get("uncertainty_scale", 0.2)))
        star_rater = StarRater(
            self.indexer,
            mapping_slope=float(params.get("mapping_slope", 4.0)),
            mapping_intercept=float(params.get("mapping_intercept", 0.0)),
            user_bias_weight=float(params.get("user_bias_weight", 0.5)),
            item_bias_weight=float(params.get("item_bias_weight", 0.5)),
            uncertainty_scale=uncertainty_scale,
        )
        qa = QAConsistency(
            self.indexer,
            star_rater,
            consistency_threshold=float(params.get("consistency_threshold", 0.75)),
            max_auto_fix_attempts=int(round(params.get("max_auto_fix_attempts", 1))),
            penalty_weight_style_violations=self.indexer.platform_policy_weight,
        )

        mse_sum = 0.0
        cons_pen_sum = 0.0
        n = 0

        # Use ground-truth review text to compute sentiment and map to stars
        for rec in train_records:
            n += 1
            u, i = rec.user_id, rec.item_id
            baseline, verb, tone, dom, ustate = persona.profile(u)
            ip_mean, ip_var, asp_summary, tags, istate = item_prof.profile(i)
            plan = plan_comp.compose_plan(ustate, tone, asp_summary, ustate.aspect_preference_weights, verbosity_prior=verb)

            sent = self.indexer.sentiment(rec.review or "")
            pred = star_rater.rate(sent, u, i, add_noise=False)
            mse_sum += (pred - rec.stars) ** 2

            # Consistency penalty: compare predicted stars and ground-truth text
            cons_score = qa.evaluate(rec.review or "", pred, plan)
            cons_pen_sum += (1.0 - cons_score)

        if n == 0:
            # Edge case: no training records
            return 1e6

        rmse = math.sqrt(mse_sum / n)
        avg_cons_pen = cons_pen_sum / n

        weights = self._normalize_weights(self.config.objective_weights)
        # text loss set to 0 during tuning (no LLM generation for speed)
        obj = weights["stars"] * rmse + weights["consistency"] * avg_cons_pen
        return obj

    def fit(self, train_records: List[InteractionRecord]) -> Dict[str, Any]:
        """
        Run random search to find parameters minimizing the objective on the training set.

        Args:
            train_records (List[InteractionRecord]): Training data.

        Returns:
            Dict[str, Any]: Best found parameters.
        """
        logging.info("Starting parameter tuning with %d trials...", self.config.num_trials)
        best_obj = float("inf")
        best_params: Dict[str, Any] = {}
        patience_left = self.config.early_stop_patience

        for trial in range(self.config.num_trials):
            # Sample parameters
            params = {
                name: self._sample_param(name)
                for name in self.PARAM_BOUNDS.keys()
            }
            self.current_params = params.copy()
            obj = self._objective(train_records, params)
            trial_rec = {"trial": trial, "params": params, "objective": obj}
            self.history_of_trials.append(trial_rec)
            logging.debug("Trial %d objective=%.4f params=%s", trial, obj, json.dumps(params))

            if obj < best_obj:
                best_obj = obj
                best_params = params.copy()
                patience_left = self.config.early_stop_patience
                logging.info("New best objective=%.4f on trial %d", obj, trial)
            else:
                patience_left -= 1
                if patience_left <= 0:
                    logging.info("Early stopping after trial %d. Best objective=%.4f", trial, best_obj)
                    break

        self.best_params = best_params
        logging.info("Tuning complete. Best objective=%.4f with params=%s", best_obj, json.dumps(best_params))
        return best_params


class Evaluator:
    """
    Computes evaluation metrics on validation data comparing generated outputs with ground-truth.

    Metrics:
        - RMSE_stars
        - MAE_stars
        - Text_Similarity (Jaccard token similarity or semantic cosine if available)
        - Sentiment_Agreement
        - Aspect_Coverage
        - Consistency_Score
        - Length_Deviation
    """

    def __init__(self) -> None:
        """Initialize Evaluator."""
        self.last_eval_metrics: Dict[str, Any] = {}
        self.by_segment_metrics: Dict[str, Any] = {}
        self._st_model: Optional[Any] = None

    def _get_st_model(self) -> Optional[Any]:
        if not _HAS_ST:
            return None
        if self._st_model is None:
            try:
                self._st_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            except Exception as e:
                logging.warning("Failed to load sentence-transformers model: %s", e)
                self._st_model = None
        return self._st_model

    def _semantic_similarity(self, a: str, b: str) -> Optional[float]:
        model = self._get_st_model()
        if model is None:
            return None
        try:
            vecs = model.encode([a, b], normalize_embeddings=True)
            sim = float((vecs[0] * vecs[1]).sum())
            return sim
        except Exception as e:
            logging.debug("Semantic similarity computation failed: %s", e)
            return None

    def compute_metrics(
        self,
        validation_records: List[InteractionRecord],
        sim_outputs: List[Dict[str, Any]],
        aspect_vocab: List[str]
    ) -> Dict[str, Any]:
        """
        Compute aggregate metrics.

        Args:
            validation_records (List[InteractionRecord]): Ground-truth validation data.
            sim_outputs (List[Dict[str, Any]]): Simulation outputs with keys: stars, text, plan, consistency_score.
            aspect_vocab (List[str]): Aspect vocabulary.

        Returns:
            Dict[str, Any]: Aggregate metrics and by-segment metrics.
        """
        if len(validation_records) != len(sim_outputs):
            raise ValueError("validation_records and sim_outputs must be aligned and of equal length")

        n = len(validation_records)
        if n == 0:
            raise ValueError("No validation records available to evaluate.")

        rmse_sum = 0.0
        mae_sum = 0.0
        text_sim_sum = 0.0
        senti_agree_sum = 0.0
        aspect_cov_sum = 0.0
        cons_score_sum = 0.0
        len_dev_sum = 0.0

        # Segments by user/item frequency tertiles
        user_counts: Dict[str, int] = {}
        item_counts: Dict[str, int] = {}
        for r in validation_records:
            user_counts[r.user_id] = user_counts.get(r.user_id, 0) + 1
            item_counts[r.item_id] = item_counts.get(r.item_id, 0) + 1

        def tertile_label(val: int, sorted_vals: List[int]) -> str:
            if not sorted_vals:
                return "all"
            # Partition by tertiles
            t1 = sorted_vals[int(0.33 * (len(sorted_vals) - 1))]
            t2 = sorted_vals[int(0.66 * (len(sorted_vals) - 1))]
            if val <= t1:
                return "low"
            elif val <= t2:
                return "mid"
            else:
                return "high"

        sorted_user_freqs = sorted(user_counts.values())
        sorted_item_freqs = sorted(item_counts.values())
        seg_accum: Dict[str, Dict[str, float]] = {}
        seg_counts: Dict[str, int] = {}

        for r, out in zip(validation_records, sim_outputs):
            gt_star = r.stars
            sim_star = float(out.get("stars", 3.0))
            sim_text = str(out.get("text", ""))
            sim_plan = out.get("plan", {"planned_aspects": [], "length_target": 3})
            plan_aspects = sim_plan.get("planned_aspects", [])
            length_target = int(sim_plan.get("length_target", 3))
            cons_score = float(out.get("consistency_score", 0.0))

            rmse_sum += (sim_star - gt_star) ** 2
            mae_sum += abs(sim_star - gt_star)

            # Text similarity
            jacc = jaccard_similarity(tokenize(sim_text), tokenize(r.review or ""))
            sem = self._semantic_similarity(sim_text, r.review or "")
            text_sim = float(jacc)
            if isinstance(sem, float):
                # Combine semantic and jaccard (weighted)
                text_sim = 0.7 * sem + 0.3 * jacc
            text_sim_sum += text_sim

            # Sentiment agreement: compare sentiment sign bucket with rating bucket
            sent = SentimentTool.polarity(sim_text)

            def bucket(st: float) -> int:
                if st <= 2.0:
                    return -1
                elif st >= 4.0:
                    return 1
                else:
                    return 0

            def sign(s: float) -> int:
                if s >= 0.2:
                    return 1
                elif s <= -0.2:
                    return -1
                else:
                    return 0

            agree = 1.0 if bucket(sim_star) == sign(sent) else 0.0
            senti_agree_sum += agree

            # Aspect coverage
            coverage = 0.0
            if plan_aspects:
                covered = sum(1 for a in plan_aspects if a in tokenize(sim_text))
                coverage = covered / len(plan_aspects)
            aspect_cov_sum += coverage

            cons_score_sum += cons_score

            # Length deviation
            sim_sentences = [s for s in re.split(r"[.!?]+", sim_text) if s.strip()]
            len_dev_sum += abs(len(sim_sentences) - length_target)

            # Segments
            u_seg = tertile_label(user_counts.get(r.user_id, 1), sorted_user_freqs)
            i_seg = tertile_label(item_counts.get(r.item_id, 1), sorted_item_freqs)
            for seg in (f"user_{u_seg}", f"item_{i_seg}", "user_all", "item_all"):
                srec = seg_accum.setdefault(seg, {
                    "rmse_sum": 0.0, "mae_sum": 0.0, "text_sim_sum": 0.0,
                    "senti_agree_sum": 0.0, "aspect_cov_sum": 0.0,
                    "cons_score_sum": 0.0, "len_dev_sum": 0.0
                })
                srec["rmse_sum"] += (sim_star - gt_star) ** 2
                srec["mae_sum"] += abs(sim_star - gt_star)
                srec["text_sim_sum"] += text_sim
                srec["senti_agree_sum"] += agree
                srec["aspect_cov_sum"] += coverage
                srec["cons_score_sum"] += cons_score
                srec["len_dev_sum"] += abs(len(sim_sentences) - length_target)
                seg_counts[seg] = seg_counts.get(seg, 0) + 1

        metrics = {
            "RMSE_stars": math.sqrt(rmse_sum / n),
            "MAE_stars": mae_sum / n,
            "Text_Similarity": text_sim_sum / n,
            "Sentiment_Agreement": senti_agree_sum / n,
            "Aspect_Coverage": aspect_cov_sum / n,
            "Consistency_Score": cons_score_sum / n,
            "Length_Deviation": len_dev_sum / n,
            "n_records": n,
        }

        # By-segment aggregation
        seg_metrics: Dict[str, Any] = {}
        for seg, sums in seg_accum.items():
            cnt = seg_counts.get(seg, 1)
            seg_metrics[seg] = {
                "RMSE_stars": math.sqrt(sums["rmse_sum"] / cnt),
                "MAE_stars": sums["mae_sum"] / cnt,
                "Text_Similarity": sums["text_sim_sum"] / cnt,
                "Sentiment_Agreement": sums["senti_agree_sum"] / cnt,
                "Aspect_Coverage": sums["aspect_cov_sum"] / cnt,
                "Consistency_Score": sums["cons_score_sum"] / cnt,
                "Length_Deviation": sums["len_dev_sum"] / cnt,
                "n": cnt
            }

        metrics["by_segment"] = seg_metrics
        self.last_eval_metrics = metrics
        self.by_segment_metrics = seg_metrics
        return metrics


# ------------------------------ Simulation ---------------------------------


class Simulator:
    """
    Forward simulation engine for validation split rollout.

    Processes events:
        - Fetch persona and item profiles
        - Compose plan
        - Generate draft review via LLM
        - Map text sentiment to stars using StarRater
        - QA check and optional revision loop
        - Emit final review and stars
    """

    def __init__(
        self,
        indexer: DataIndexer,
        persona: PersonaProfiler,
        item_profiler: ItemProfiler,
        plan_composer: PlanComposer,
        author: ReviewAuthor,
        star_rater: StarRater,
        qa: QAConsistency
    ) -> None:
        """
        Initialize Simulator with agent roles.

        Args:
            indexer (DataIndexer): Shared indexer.
            persona (PersonaProfiler): Persona agent.
            item_profiler (ItemProfiler): Item agent.
            plan_composer (PlanComposer): Planning agent.
            author (ReviewAuthor): Review generator.
            star_rater (StarRater): Rating agent.
            qa (QAConsistency): Consistency checker.
        """
        self.indexer = indexer
        self.persona = persona
        self.item_profiler = item_profiler
        self.plan_composer = plan_composer
        self.author = author
        self.star_rater = star_rater
        self.qa = qa

    def rollout(self, records: List[InteractionRecord]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Simulate outputs for a batch of interaction records.

        Args:
            records (List[InteractionRecord]): Events to simulate.

        Returns:
            Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]: (outputs, traces)
        """
        outputs: List[Dict[str, Any]] = []
        traces: List[Dict[str, Any]] = []

        for rec in records:
            u, i = rec.user_id, rec.item_id
            baseline, verb, tone_prior, dom, ustate = self.persona.profile(u)
            mean, var, asp_summary, tags, istate = self.item_profiler.profile(i)
            plan = self.plan_composer.compose_plan(ustate, tone_prior, asp_summary, ustate.aspect_preference_weights, verbosity_prior=verb)

            # Generate review (LLM or mock)
            # Keep immutable copy of prior tone for trace
            tone_prior_copy = dict(tone_prior)
            text = self.author.generate(u, i, plan, tone_prior_copy, dom + tags)
            # Proposed stars from sentiment
            sentiment = self.indexer.sentiment(text)
            proposed_stars = self.star_rater.rate(sentiment, u, i, add_noise=True)

            # QA revision loop with budget from both QA and Author
            attempts = 0
            max_attempts = min(self.qa.max_auto_fix_attempts, self.author.max_revision_loops)
            while self.qa.needs_revision(text, proposed_stars, plan) and attempts < max_attempts:
                # Simple revision policy: adjust stars toward implied sentiment, re-prompt for a concise rewrite
                implied_stars = 1.0 + 4.0 * sigmoid(self.star_rater.mapping_slope * sentiment + self.star_rater.mapping_intercept)
                proposed_stars = clamp(0.5 * proposed_stars + 0.5 * implied_stars, 1.0, 5.0)
                # Influence author with revised tone (slightly)
                tone_prior_copy["exuberance"] = clamp(tone_prior_copy.get("exuberance", 0.5) * 0.95, 0.0, 1.0)
                text = self.author.generate(u, i, plan, tone_prior_copy, dom + tags)
                sentiment = self.indexer.sentiment(text)
                attempts += 1

            cons_score = self.qa.evaluate(text, proposed_stars, plan)

            outputs.append({
                "user_id": u,
                "item_id": i,
                "stars": round(proposed_stars, 3),
                "text": text,
                "plan": {
                    "planned_aspects": plan.planned_aspects,
                    "length_target": plan.length_target,
                    "tone_targets": plan.tone_targets
                },
                "consistency_score": round(cons_score, 4),
            })

            traces.append({
                "user_id": u,
                "item_id": i,
                "inputs": {
                    "persona": {
                        "baseline_leniency": baseline,
                        "verbosity_prior": verb,
                        "tone_style_prior": tone_prior,  # immutable original prior
                        "domain_familiarity": dom,
                    },
                    "item": {
                        "quality_prior": mean,
                        "variance": var,
                        "aspect_summary": asp_summary,
                        "domain_tags": tags
                    }
                },
                "plan": {
                    "aspects": plan.planned_aspects,
                    "length_target": plan.length_target,
                    "tone_targets": plan.tone_targets
                },
                "generated_text": text,
                "diagnostics": {
                    "sentiment": sentiment,
                    "proposed_stars": proposed_stars,
                    "consistency_score": cons_score,
                    "revision_count": attempts
                }
            })

            # Update dynamic states post interaction
            self.persona.update_from_interaction(u, sentiment)
            self.item_profiler.update_from_interaction(i, proposed_stars)

        return outputs, traces


# --------------------------- Data Loading ----------------------------------


def load_data(
    interactions_file: str,
    user_profiles_file: Optional[str] = None,
    item_metadata_file: Optional[str] = None
) -> Tuple[List[InteractionRecord], List[UserProfile], List[ItemMetadata]]:
    """
    Load interactions and optional profile/metadata CSVs.

    Args:
        interactions_file (str): Absolute or relative path to interactions.csv.
        user_profiles_file (Optional[str]): Path to user_profiles.csv.
        item_metadata_file (Optional[str]): Path to item_metadata.csv.

    Returns:
        Tuple[List[InteractionRecord], List[UserProfile], List[ItemMetadata]]

    Raises:
        ValueError: For validation errors (missing columns).
    """
    # Interactions
    rows = read_csv_dicts(interactions_file)
    interactions: List[InteractionRecord] = []
    required = {"user_id", "item_id", "stars", "review"}
    header = set(rows[0].keys()) if rows else set()
    if not required.issubset(header):
        raise ValueError(
            f"interactions.csv must contain columns: {sorted(required)}. Found: {sorted(header)}"
        )

    for r in rows:
        user_id = r.get("user_id", "").strip()
        item_id = r.get("item_id", "").strip()
        if not user_id or not item_id:
            continue
        stars_val = to_float(r.get("stars", 0.0), 0.0)
        # Validate/clamp stars to [1,5]
        if stars_val < 1.0 or stars_val > 5.0:
            logging.warning("Star rating out of range for user=%s item=%s: %.3f. Clamped to [1,5].",
                            user_id, item_id, stars_val)
        stars = clamp(stars_val, 1.0, 5.0)
        review = r.get("review", "")
        datatype = r.get("datatype", r.get("split", "")).strip().lower()
        ts = parse_date(r.get("timestamp", r.get("date", "")))
        extra = {k: v for k, v in r.items() if k not in {"user_id", "item_id", "stars", "review", "datatype", "split", "timestamp", "date"}}
        interactions.append(InteractionRecord(user_id=user_id, item_id=item_id, stars=stars, review=review, datatype=datatype, timestamp=ts, extra=extra))

    # User profiles (optional)
    profiles: List[UserProfile] = []
    if user_profiles_file and os.path.exists(user_profiles_file):
        prows = read_csv_dicts(user_profiles_file)
        for r in prows:
            uid = r.get("user_id", "").strip()
            if not uid:
                continue
            avg = parse_optional_float(r.get("average_stars", None))
            friends: List[str] = []
            if "friends" in r and r["friends"].strip():
                friends = [x.strip() for x in r["friends"].split(";") if x.strip()]
            extra = {k: v for k, v in r.items() if k not in {"user_id", "average_stars", "friends"}}
            profiles.append(UserProfile(user_id=uid, average_stars=avg, friends=friends, extra=extra))

    # Item metadata (optional)
    imetas: List[ItemMetadata] = []
    if item_metadata_file and os.path.exists(item_metadata_file):
        irows = read_csv_dicts(item_metadata_file)
        for r in irows:
            iid = r.get("item_id", "").strip()
            if not iid:
                continue
            cat = r.get("category", None)
            extra = {k: v for k, v in r.items() if k not in {"item_id", "category"}}
            imetas.append(ItemMetadata(item_id=iid, category=cat, extra=extra))

    return interactions, profiles, imetas


def holdout_split(records: List[InteractionRecord], seed: int = 42) -> Tuple[List[InteractionRecord], List[InteractionRecord]]:
    """
    Partition data into training and validation sets per blueprint.

    Priority:
        - Use datatype labels if present ('train'/'test')
        - Else if timestamps exist, use temporal split (first 80% by date train, last 20% test)
        - Else random split with fixed seed (80/20)

    Args:
        records (List[InteractionRecord]): All interactions.
        seed (int): Random seed for randomized split.

    Returns:
        Tuple[List[InteractionRecord], List[InteractionRecord]]: (train, validation)
    """
    has_labels = any(r.datatype for r in records)
    if has_labels:
        train = [r for r in records if r.datatype == "train"]
        val = [r for r in records if r.datatype == "test"]
        # Fallback if custom labels provided
        if not train or not val:
            rest = [r for r in records if r.datatype not in {"train", "test"}]
            random.Random(seed).shuffle(rest)
            k = int(0.8 * len(rest))
            train.extend(rest[:k])
            val.extend(rest[k:])
        return train, val

    has_timestamps = any(r.timestamp for r in records)
    if has_timestamps:
        recs = sorted(records, key=lambda r: r.timestamp or datetime.datetime.min)
        k = int(0.8 * len(recs))
        return recs[:k], recs[k:]

    # Random split fallback
    rnd = random.Random(seed)
    shuffled = records[:]
    rnd.shuffle(shuffled)
    k = int(0.8 * len(shuffled))
    return shuffled[:k], shuffled[k:]


# ---------------------------- Orchestration --------------------------------


def parse_cli() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Multi-agent review and rating simulator with calibration.")
    parser.add_argument("--seed", type=int, default=42, help="Global random seed for determinism.")
    parser.add_argument("--data_dir", type=str, default=_get_default_data_dir(), help="Directory containing data files. Defaults to env PROJECT_ROOT/DATA_PATH or ./data.")
    parser.add_argument("--interactions_file", type=str, default="interactions.csv", help="Interactions CSV filename or path.")
    parser.add_argument("--user_profiles_file", type=str, default="user_profiles.csv", help="User profiles CSV filename or path (optional).")
    parser.add_argument("--item_metadata_file", type=str, default="item_metadata.csv", help="Item metadata CSV filename or path (optional).")
    parser.add_argument("--num_trials", type=int, default=12, help="Number of calibration trials.")
    parser.add_argument("--early_stop_patience", type=int, default=5, help="Early stop patience for tuner.")
    parser.add_argument("--mock_llm", action="store_true", help="Use deterministic mock review generator instead of calling OpenAI (development only).")
    parser.add_argument("--llm_max_output_tokens", type=int, default=400, help="Max output tokens for LLM generation.")
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level.")
    args = parser.parse_args()

    # Resolve absolute paths using provided data_dir if arguments are relative
    global DATA_DIR
    DATA_DIR = os.path.abspath(args.data_dir)

    def _resolve_path(p: str) -> str:
        return p if os.path.isabs(p) else os.path.join(DATA_DIR, p)

    args.interactions_file = _resolve_path(args.interactions_file)
    args.user_profiles_file = _resolve_path(args.user_profiles_file)
    args.item_metadata_file = _resolve_path(args.item_metadata_file)
    return args


def build_network_and_agents(
    interactions: List[InteractionRecord],
    profiles: List[UserProfile],
    metadata: List[ItemMetadata],
    calibrated_params: Optional[Dict[str, Any]] = None,
    mock_llm: bool = False,
    llm_max_output_tokens: int = 400
) -> Tuple[DataIndexer, PersonaProfiler, ItemProfiler, PlanComposer, ReviewAuthor, StarRater, QAConsistency, Evaluator]:
    """
    Construct all agent roles and evaluator, optionally using calibrated parameters.

    Args:
        interactions (List[InteractionRecord]): All interactions.
        profiles (List[UserProfile]): User profiles.
        metadata (List[ItemMetadata]): Item metadata.
        calibrated_params (Optional[Dict[str, Any]]): Best found parameters to configure agents.
        mock_llm (bool): If True, use mock review generation.
        llm_max_output_tokens (int): Max tokens for LLM output.

    Returns:
        Tuple of (DataIndexer, PersonaProfiler, ItemProfiler, PlanComposer, ReviewAuthor, StarRater, QAConsistency, Evaluator)
    """
    calibrated_params = calibrated_params or {}

    indexer = DataIndexer(
        interactions=interactions,
        user_profiles=profiles,
        item_metadata=metadata,
        platform_policy_weight=0.8
    )

    persona = PersonaProfiler(
        indexer=indexer,
        neighbor_weight=float(calibrated_params.get("neighbor_weight", 0.0))
    )

    item_profiler = ItemProfiler(indexer=indexer)

    plan_composer = PlanComposer(
        indexer=indexer,
        aspect_topk=int(round(calibrated_params.get("aspect_topk", 4))),
        length_target_mean=float(calibrated_params.get("length_target_mean", 3.0)),
        ctx_merge_weight=float(calibrated_params.get("ctx_merge_weight", 0.5)),
        plan_diversity_temp=float(calibrated_params.get("plan_diversity_temp", 0.8)),
    )

    author = ReviewAuthor(
        indexer=indexer,
        llm_temperature=float(calibrated_params.get("llm_temperature", 0.5)),
        style_alignment_weight=float(calibrated_params.get("style_alignment_weight", 0.7)),
        max_revision_loops=int(round(calibrated_params.get("max_auto_fix_attempts", 1))),
        mock_llm=mock_llm,
        llm_max_output_tokens=llm_max_output_tokens
    )

    star_rater = StarRater(
        indexer=indexer,
        mapping_slope=float(calibrated_params.get("mapping_slope", 4.0)),
        mapping_intercept=float(calibrated_params.get("mapping_intercept", 0.0)),
        user_bias_weight=float(calibrated_params.get("user_bias_weight", 0.5)),
        item_bias_weight=float(calibrated_params.get("item_bias_weight", 0.5)),
        uncertainty_scale=float(calibrated_params.get("uncertainty_scale", 0.2)),
    )

    qa = QAConsistency(
        indexer=indexer,
        star_rater=star_rater,
        consistency_threshold=float(calibrated_params.get("consistency_threshold", 0.75)),
        max_auto_fix_attempts=int(round(calibrated_params.get("max_auto_fix_attempts", 1))),
        penalty_weight_style_violations=indexer.platform_policy_weight
    )

    evaluator = Evaluator()

    return indexer, persona, item_profiler, plan_composer, author, star_rater, qa, evaluator


def save_results(
    data_dir: str,
    calibrated_params: Dict[str, Any],
    indexer: DataIndexer,
    traces: List[Dict[str, Any]],
    metrics: Dict[str, Any],
    ablation_metrics: Optional[Dict[str, Any]] = None
) -> None:
    """
    Save simulation outputs and evaluation metrics to files.

    Args:
        data_dir (str): Output directory (DATA_DIR).
        calibrated_params (Dict[str, Any]): Best parameters.
        indexer (DataIndexer): Data indexer (for priors).
        traces (List[Dict[str, Any]]): Simulation traces per record.
        metrics (Dict[str, Any]): Evaluation metrics.
        ablation_metrics (Optional[Dict[str, Any]]): Optional ablation report metrics.
    """
    ensure_dir(data_dir)

    # 1. calibrated_parameters.json
    calib_path = os.path.join(data_dir, "calibrated_parameters.json")
    priors = {
        "user_prior_mean": indexer.user_prior_mean,
        "item_prior_mean": indexer.item_prior_mean,
        "item_prior_var": indexer.item_prior_var,
        "aspect_vocab": indexer.aspect_vocab,
        "platform_policy_weight": indexer.platform_policy_weight
    }
    with open(calib_path, "w", encoding="utf-8") as f:
        json.dump({"calibrated_params": calibrated_params, "priors": priors}, f, indent=2)

    # 2. simulation_traces.jsonl
    traces_path = os.path.join(data_dir, "simulation_traces.jsonl")
    with open(traces_path, "w", encoding="utf-8") as f:
        for rec in traces:
            f.write(json.dumps(rec) + "\n")

    # 3. evaluation_metrics.json
    eval_path = os.path.join(data_dir, "evaluation_metrics.json")
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # 4. ablation_report.json (optional)
    if ablation_metrics is not None:
        ablation_path = os.path.join(data_dir, "ablation_report.json")
        with open(ablation_path, "w", encoding="utf-8") as f:
            json.dump({"neighbor_weight=0.0": ablation_metrics}, f, indent=2)

    logging.info("Results saved: %s, %s, %s%s",
                 calib_path, traces_path, eval_path,
                 f", and ablation_report.json" if ablation_metrics is not None else "")


# --------------------------------- main ------------------------------------


def main() -> None:
    """
    Orchestrator for the simulation pipeline.

    Steps:
        - parse_cli()
        - load_data()
        - build_network_and_agents()
        - holdout_split()
        - calibrator.fit()
        - rebuild aspect vocabulary from training data
        - simulator.rollout()
        - evaluator.compute_metrics()
        - save_results()
    """
    args = parse_cli()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
        stream=sys.stdout
    )

    # Determinism
    set_global_determinism(args.seed)

    # Load data
    logging.info("Loading data from %s", DATA_DIR)
    interactions, profiles, metadata = load_data(
        interactions_file=args.interactions_file,
        user_profiles_file=args.user_profiles_file if os.path.exists(args.user_profiles_file) else None,
        item_metadata_file=args.item_metadata_file if os.path.exists(args.item_metadata_file) else None
    )
    logging.info("Loaded %d interactions, %d user profiles, %d item metadata rows.",
                 len(interactions), len(profiles), len(metadata))

    # Initialize components with default params for initial indexing
    indexer, persona, item_prof, plan_comp, author, star_rater, qa, evaluator = build_network_and_agents(
        interactions=interactions,
        profiles=profiles,
        metadata=metadata,
        calibrated_params=None,
        mock_llm=args.mock_llm,
        llm_max_output_tokens=args.llm_max_output_tokens
    )

    # Holdout split
    train_records, val_records = holdout_split(interactions, seed=args.seed)
    logging.info("Split: %d training, %d validation records.", len(train_records), len(val_records))

    # Rebuild aspect vocabulary from training split for better alignment
    try:
        indexer.rebuild_aspect_vocab_from_records(train_records, top_k=12)
    except Exception as e:
        logging.warning("Rebuilding aspect vocabulary failed; using defaults. Reason: %s", e)

    # Calibrate parameters on training split
    tuner = ParameterTuner(indexer=indexer, config=TunerConfig(
        num_trials=int(args.num_trials),
        early_stop_patience=int(args.early_stop_patience),
        objective_weights={"stars": 0.75, "text": 0.0, "consistency": 0.25}
    ))
    best_params = tuner.fit(train_records)

    # Rebuild agents with calibrated parameters for rollout
    indexer, persona, item_prof, plan_comp, author, star_rater, qa, evaluator = build_network_and_agents(
        interactions=interactions,
        profiles=profiles,
        metadata=metadata,
        calibrated_params=best_params,
        mock_llm=args.mock_llm,
        llm_max_output_tokens=args.llm_max_output_tokens
    )

    # Ensure aspect vocab aligns with training-based rebuild for rollout
    try:
        indexer.rebuild_aspect_vocab_from_records(train_records, top_k=12)
    except Exception:
        pass

    # Forward simulation on validation split (LLM used unless mock_llm=True)
    simulator = Simulator(indexer, persona, item_prof, plan_comp, author, star_rater, qa)
    outputs, traces = simulator.rollout(val_records)

    # Evaluation metrics
    metrics = evaluator.compute_metrics(val_records, outputs, indexer.aspect_vocab)
    logging.info("Evaluation metrics: %s", json.dumps(metrics, indent=2))

    # Optional ablation: neighbor influence off
    ablation_metrics = None
    try:
        ablation_params = dict(best_params)
        ablation_params["neighbor_weight"] = 0.0
        indexer_a, persona_a, item_prof_a, plan_comp_a, author_a, star_rater_a, qa_a, evaluator_a = build_network_and_agents(
            interactions=interactions,
            profiles=profiles,
            metadata=metadata,
            calibrated_params=ablation_params,
            mock_llm=True,  # use mock to avoid extra API calls for ablation
            llm_max_output_tokens=args.llm_max_output_tokens
        )
        # Align aspect vocab
        try:
            indexer_a.rebuild_aspect_vocab_from_records(train_records, top_k=12)
        except Exception:
            pass
        simulator_a = Simulator(indexer_a, persona_a, item_prof_a, plan_comp_a, author_a, star_rater_a, qa_a)
        outputs_a, _ = simulator_a.rollout(val_records)
        ablation_metrics = evaluator_a.compute_metrics(val_records, outputs_a, indexer_a.aspect_vocab)
    except Exception as e:
        logging.warning("Ablation run failed or skipped due to: %s", e)

    # Save results
    save_results(
        data_dir=DATA_DIR,
        calibrated_params=best_params,
        indexer=indexer,
        traces=traces,
        metrics=metrics,
        ablation_metrics=ablation_metrics
    )


# Execute main for both direct execution and sandbox wrapper invocation
main()