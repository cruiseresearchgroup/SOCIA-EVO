#!/usr/bin/env python3
import argparse
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

# Global deterministic seed
GLOBAL_RANDOM_SEED = 42
random.seed(GLOBAL_RANDOM_SEED)
np.random.seed(GLOBAL_RANDOM_SEED)

# Path handling per instructions (may be overridden by CLI)
PROJECT_ROOT = os.environ.get("PROJECT_ROOT")
DATA_PATH = os.environ.get("DATA_PATH")
DATA_DIR = os.path.join(PROJECT_ROOT, DATA_PATH) if PROJECT_ROOT and DATA_PATH else None

# LLM API import handling
try:
    from openai import OpenAI  # noqa: F401
    _OPENAI_AVAILABLE = True
    _OPENAI_IMPORT_ERROR = None
except Exception as _e:
    OpenAI = None  # type: ignore
    _OPENAI_AVAILABLE = False
    _OPENAI_IMPORT_ERROR = _e  # type: ignore

# Simple in-memory cache for LLM responses to reduce cost during tuning
_LLM_CACHE: Dict[Tuple[str, str, float, int], str] = {}

def get_openai_api_key():
    """
    Retrieve the OpenAI API key from the environment.

    Returns:
        str: The API key.

    Raises:
        ValueError: If the OPENAI_API_KEY environment variable is missing.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        return api_key
    raise ValueError("OpenAI API key not found in environment")


def call_gpt5_with_responses_api(prompt: str, model: str = "gpt-5", max_output_tokens: int = 4000, llm_temperature: float = 0.5) -> str:
    """
    Call the OpenAI Responses API with the provided prompt and return the generated text.

    This function uses client.responses.create with the "input" schema and max_output_tokens.

    Args:
        prompt (str): The input text to pass to the model.
        model (str): The model name to use. Default: "gpt-5".
        max_output_tokens (int): Maximum number of output tokens. Default: 4000.
        llm_temperature (float): The sampling temperature to send to the API.

    Returns:
        str: The extracted text from the response.

    Raises:
        RuntimeError: If the openai package is not installed.
        ValueError: If the API key is missing from environment.
        Exception: Any other exception raised by the OpenAI client will propagate.
    """
    # Memoization first
    cache_key = (model, prompt, float(llm_temperature), int(max_output_tokens))
    if cache_key in _LLM_CACHE:
        return _LLM_CACHE[cache_key]

    if not _OPENAI_AVAILABLE or OpenAI is None:
        msg = (
            "The 'openai' package is required but not installed or failed to import.\n"
            f"Original import error: {_OPENAI_IMPORT_ERROR if '_OPENAI_IMPORT_ERROR' in globals() else 'Unknown'}\n"
            "Install with: pip install openai"
        )
        raise RuntimeError(msg)

    api_key = get_openai_api_key()
    # Ensure environment-based auth for newer SDKs
    if os.environ.get("OPENAI_API_KEY") != api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    client = OpenAI()

    responses_kwargs = {
        "model": model,
        "input": [
            {"role": "user", "content": [{"type": "input_text", "text": prompt}]}
        ],
        "max_output_tokens": max_output_tokens,
        "temperature": float(llm_temperature),
    }

    resp = client.responses.create(**responses_kwargs)

    def extract_response(resp_obj):
        # Preferred path
        if hasattr(resp_obj, "output_text") and isinstance(resp_obj.output_text, str):
            return resp_obj.output_text
        # Try to parse known structures
        try:
            output = getattr(resp_obj, "output", None)
            if output and isinstance(output, list):
                blk = output[0]
                content = blk.get("content") if isinstance(blk, dict) else None
                if isinstance(content, list):
                    # look for any text fields in content entries
                    for part in content:
                        if isinstance(part, dict):
                            # Types like 'output_text' or 'text'
                            ptype = part.get("type")
                            if ptype in ("output_text", "text"):
                                # Some SDKs put the text directly under 'text'
                                if isinstance(part.get("text"), str):
                                    return part["text"]
                                # Some put under 'content' key
                                if isinstance(part.get("content"), str):
                                    return part["content"]
                            # Common 'text' field
                            if "text" in part and isinstance(part["text"], str):
                                return part["text"]
                            # Nested structure fallback
                            if "content" in part and isinstance(part["content"], dict):
                                inner = part["content"]
                                if "text" in inner and isinstance(inner["text"], str):
                                    return inner["text"]
                            # Annotations or other nested structures
                            if "annotations" in part and isinstance(part["annotations"], list):
                                for ann in part["annotations"]:
                                    if isinstance(ann, dict):
                                        txt = ann.get("text")
                                        if isinstance(txt, str) and txt.strip():
                                            return txt
        except Exception:
            pass
        # Fallback to string representation with logging to stderr
        sys.stderr.write("Warning: Falling back to string representation for LLM response parsing.\n")
        return str(resp_obj)

    text = extract_response(resp)
    _LLM_CACHE[cache_key] = text
    return text


# ------------------------------ Utilities ------------------------------


def set_global_seed(seed: int) -> None:
    """
    Set random seeds for deterministic behavior.

    Args:
        seed (int): The seed value to set for Python's random and NumPy.
    """
    if not isinstance(seed, int):
        raise ValueError("Seed must be an integer")
    global GLOBAL_RANDOM_SEED
    GLOBAL_RANDOM_SEED = seed
    random.seed(seed)
    np.random.seed(seed)


def ensure_data_dir_valid(data_dir: Optional[str] = None) -> str:
    """
    Validate a data directory. If not provided, try environment variables PROJECT_ROOT/DATA_PATH.
    If still not available, default to ./data. Ensures directory exists.

    Args:
        data_dir (Optional[str]): Override directory.

    Returns:
        str: Resolved data directory.

    Raises:
        ValueError: If the directory does not exist.
    """
    if data_dir and os.path.isdir(data_dir):
        return data_dir
    # Try env-based
    if PROJECT_ROOT and DATA_PATH:
        combined = os.path.join(PROJECT_ROOT, DATA_PATH)
        if os.path.isdir(combined):
            return combined
    # Default to ./data
    fallback = os.path.join(os.getcwd(), "data")
    if os.path.isdir(fallback):
        return fallback
    raise ValueError(
        "Data directory not found. Provide --data_dir, set PROJECT_ROOT/DATA_PATH env vars, "
        "or create a ./data directory."
    )


def safe_json_dump(obj: Any, path: str) -> None:
    """
    Write object as pretty JSON to a file, ensuring directories exist.

    Args:
        obj (Any): Object to serialize.
        path (str): File path to write.
    """
    dirpath = os.path.dirname(path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def safe_jsonl_dump(records: Iterable[Dict[str, Any]], path: str) -> None:
    """
    Write records as JSONL to a file.

    Args:
        records (Iterable[Dict[str, Any]]): Iterable of dictionaries to write as JSON lines.
        path (str): File path to write.
    """
    dirpath = os.path.dirname(path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


_STOPWORDS = set(
    """
    a an the and or but if while of in on at for to from with without as by about is are was were be been being
    it this that these those i you he she they we me him her them my your his their our ours yours its
    do does did doing done have has had having not no yes can could should would may might will shall
    up down over under again further then once here there when where why how all any both each few more most other some such
    than too very s t just don now into out across after before because until
    """.split()
)

@lru_cache(maxsize=20000)
def _tokenize_core(text: str) -> Tuple[str, ...]:
    if not isinstance(text, str):
        return tuple()
    tokens: List[str] = []
    cur: List[str] = []
    for ch in text.lower():
        if ch.isalnum():
            cur.append(ch)
        else:
            if cur:
                token = "".join(cur)
                if token and token not in _STOPWORDS:
                    tokens.append(token)
                cur = []
    if cur:
        token = "".join(cur)
        if token and token not in _STOPWORDS:
            tokens.append(token)
    return tuple(tokens)

def tokenize(text: str) -> List[str]:
    """
    Basic tokenizer: lowercasing, alphanumeric filtering, stopword removal.

    Args:
        text (str): Input text.

    Returns:
        List[str]: List of tokens.
    """
    return list(_tokenize_core(text or ""))


@lru_cache(maxsize=20000)
def _build_tf_cached(tokens_tuple: Tuple[str, ...]) -> Dict[str, float]:
    if not tokens_tuple:
        return {}
    counts = Counter(tokens_tuple)
    total = float(sum(counts.values()))
    return {t: c / total for t, c in counts.items()}


def build_tf(tokens: List[str]) -> Dict[str, float]:
    """
    Build term-frequency dictionary from tokens.

    Args:
        tokens (List[str]): Tokens.

    Returns:
        Dict[str, float]: Mapping token -> normalized frequency.
    """
    t = tuple(tokens) if isinstance(tokens, list) else tuple(tokens or [])
    return _build_tf_cached(t)


def cosine_sim_from_tfs(tf1: Dict[str, float], tf2: Dict[str, float]) -> float:
    """
    Compute cosine similarity from two TF dictionaries.

    Args:
        tf1 (Dict[str, float]): Term frequency dict 1.
        tf2 (Dict[str, float]): Term frequency dict 2.

    Returns:
        float: Cosine similarity in [0, 1].
    """
    if not tf1 or not tf2:
        return 0.0
    keys = set(tf1) | set(tf2)
    v1 = np.array([tf1.get(k, 0.0) for k in keys], dtype=float)
    v2 = np.array([tf2.get(k, 0.0) for k in keys], dtype=float)
    denom = float(np.linalg.norm(v1) * np.linalg.norm(v2))
    if denom <= 0:
        return 0.0
    sim = float(np.dot(v1, v2) / denom)
    sim = max(0.0, min(1.0, sim))
    return sim


_POSITIVE_WORDS = set(
    """
    good great excellent amazing fantastic love loved like liked awesome perfect nice recommend recommended positive
    satisfied satisfying happy delightful wonderful superb outstanding impressive enjoyable pleasant
    """.split()
)
_NEGATIVE_WORDS = set(
    """
    bad terrible awful poor disappointing hate hated dislike disliked horrible worst broken flawed negative
    unsatisfied unsatisfying unhappy frustrating bug buggy issue issues problem problems
    """.split()
)

@lru_cache(maxsize=20000)
def sentiment_score(text: str) -> float:
    """
    Compute a naive sentiment score in [-1, 1] using counts of a small lexicon.

    Args:
        text (str): Input text.

    Returns:
        float: Score in [-1, 1].
    """
    toks = tokenize(text)
    if not toks:
        return 0.0
    pos = sum(1 for t in toks if t in _POSITIVE_WORDS)
    neg = sum(1 for t in toks if t in _NEGATIVE_WORDS)
    total = max(1, len(toks))
    score = (pos - neg) / float(total)
    score = max(-1.0, min(1.0, score))
    return score


def sentence_count(text: str) -> int:
    """
    Count number of sentences by naive punctuation splitting.

    Args:
        text (str): Input text.

    Returns:
        int: Approximate sentence count.
    """
    if not isinstance(text, str) or not text:
        return 0
    parts = [p.strip() for p in text.replace("!", ".").replace("?", ".").split(".")]
    parts = [p for p in parts if p]
    return len(parts)


def trim_to_sentences(text: str, target_sentences: int) -> str:
    """
    Trim or keep text to the first N sentences.

    Args:
        text (str): Input text.
        target_sentences (int): Number of sentences to keep.

    Returns:
        str: Text with at most target_sentences sentences.
    """
    if not isinstance(text, str) or target_sentences <= 0:
        return ""
    sentences = []
    cur = []
    for ch in text:
        cur.append(ch)
        if ch in ".!?":
            sent = "".join(cur).strip()
            if sent:
                sentences.append(sent)
            cur = []
    if cur:
        leftover = "".join(cur).strip()
        if leftover:
            sentences.append(leftover)
    return " ".join(sentences[: max(1, target_sentences)])


# ------------------------------ DataIndexer (Memory) ------------------------------


class DataIndexer:
    """
    DataIndexer constructs and maintains indices, priors, and corpora from the interaction data.

    Responsibilities:
    - Parse all records and construct lookups by user_id and item_id.
    - Compute priors: user average stars, item average stars, variances (on training subset to avoid leakage).
    - Build aspect vocabulary from review text tokens.
    - Maintain platform policy and global statistics.
    """

    def __init__(
        self,
        interactions: pd.DataFrame,
        users: Optional[pd.DataFrame],
        items: Optional[pd.DataFrame],
        prior_indices: Optional[Iterable[int]] = None,
    ):
        """
        Initialize DataIndexer.

        Args:
            interactions (pd.DataFrame): Interactions with columns including user_id, item_id, stars, review.
            users (Optional[pd.DataFrame]): Optional user profiles.
            items (Optional[pd.DataFrame]): Optional item metadata.
            prior_indices (Optional[Iterable[int]]): Indices to use for computing priors (typically train indices).

        Raises:
            ValueError: If interactions lacks required columns.
        """
        required_cols = {"user_id", "item_id", "stars"}
        missing = required_cols - set(interactions.columns)
        if missing:
            raise ValueError(f"interactions.csv missing required columns: {missing}")
        self.interactions = interactions.copy()
        self.users = users
        self.items = items

        # Timestamp awareness
        self.has_timestamp = "timestamp" in self.interactions.columns and pd.api.types.is_datetime64_any_dtype(
            self.interactions["timestamp"]
        )

        # Build indices
        self.user_index: Dict[str, List[int]] = defaultdict(list)
        self.item_index: Dict[str, List[int]] = defaultdict(list)
        for idx, row in self.interactions.iterrows():
            self.user_index[str(row["user_id"])].append(idx)
            self.item_index[str(row["item_id"])].append(idx)

        # Determine indices to compute priors (train-only to avoid leakage)
        if prior_indices is not None:
            self.prior_indices_set = set(int(i) for i in prior_indices)
        else:
            self.prior_indices_set = set(self.interactions.index.tolist())

        # Compute priors on training subset
        self.global_mean_stars = float(self.interactions.loc[list(self.prior_indices_set), "stars"].mean())
        self.user_priors: Dict[str, Dict[str, float]] = {}
        self.item_priors: Dict[str, Dict[str, float]] = {}
        self._compute_priors()

        # Aspect vocabulary
        self.aspect_vocab = self._build_aspect_vocab(top_k=50)

        # Split index: datatype if available
        self.split_index: Dict[str, List[int]] = defaultdict(list)
        if "datatype" in self.interactions.columns:
            for idx, row in self.interactions.iterrows():
                dtype = str(row["datatype"])
                self.split_index[dtype].append(idx)

        # Platform policy (exogenous signal weight)
        self.platform_policy = {
            "style_weight": 0.8,  # weight of style penalty
            "max_length_sentences": 8,  # soft cap
            "profanity": False,  # no profanity allowed
            "name": "amazon/product",
        }

        # Additional internal state
        self.cache_hits = 0
        self.last_refreshed_version = int(time.time())
        self.embedding_store_refs: Dict[int, Dict[str, float]] = {}  # idx -> tf dict

        # Precompute lightweight embeddings (term frequencies) for reviews
        if "review" in self.interactions.columns:
            for idx, row in self.interactions.iterrows():
                tf = build_tf(tokenize(str(row.get("review", ""))))
                self.embedding_store_refs[idx] = tf

    def _compute_priors(self) -> None:
        """Compute user and item priors: mean stars and variance using training subset to avoid leakage."""
        # User priors
        for user_id, idxs in self.user_index.items():
            idxs_train = [i for i in idxs if i in self.prior_indices_set]
            stars = self.interactions.loc[idxs_train, "stars"].dropna().astype(float).values
            if len(stars) == 0:
                mean, var = self.global_mean_stars, 1.0
            else:
                mean = float(np.mean(stars))
                var = float(np.var(stars)) if len(stars) > 1 else 1.0
            self.user_priors[user_id] = {"avg": mean, "var": var}
        # Item priors
        for item_id, idxs in self.item_index.items():
            idxs_train = [i for i in idxs if i in self.prior_indices_set]
            stars = self.interactions.loc[idxs_train, "stars"].dropna().astype(float).values
            if len(stars) == 0:
                mean, var = self.global_mean_stars, 1.0
            else:
                mean = float(np.mean(stars))
                var = float(np.var(stars)) if len(stars) > 1 else 1.0
            self.item_priors[item_id] = {"avg": mean, "var": var}

    def _build_aspect_vocab(self, top_k: int = 50) -> List[str]:
        """
        Build a simple aspect vocabulary by selecting top tokens from all reviews.

        Args:
            top_k (int): Number of aspects to retain.

        Returns:
            List[str]: Aspect vocabulary.
        """
        counter = Counter()
        if "review" in self.interactions.columns:
            for _, row in self.interactions.iterrows():
                counter.update(tokenize(str(row.get("review", ""))))
        # Remove overly generic tokens
        for t in list(counter.keys()):
            if len(t) <= 2:
                del counter[t]
        most_common = [t for t, _ in counter.most_common(top_k)]
        if not most_common:
            # Fallback generic aspects
            most_common = ["quality", "price", "shipping", "packaging", "durability", "usability"]
        return most_common

    def get_user_history(
        self,
        user_id: str,
        cutoff_idx: Optional[int] = None,
        cutoff_timestamp: Optional[pd.Timestamp] = None,
    ) -> List[int]:
        """
        Get indices for a user's interaction history, optionally up to cutoff (index or timestamp).

        Args:
            user_id (str): User ID.
            cutoff_idx (Optional[int]): If provided, restrict history to indices < cutoff_idx.
            cutoff_timestamp (Optional[pd.Timestamp]): If provided, restrict to timestamps < cutoff_timestamp.

        Returns:
            List[int]: List of indices.
        """
        ids = self.user_index.get(str(user_id), [])
        if cutoff_timestamp is not None and self.has_timestamp:
            # filter strictly before cutoff_timestamp
            prior_ids = [i for i in ids if pd.notna(self.interactions.at[i, "timestamp"]) and self.interactions.at[i, "timestamp"] < cutoff_timestamp]
            return prior_ids
        if cutoff_idx is not None:
            return [i for i in ids if i < cutoff_idx]
        return ids

    def get_item_history(
        self,
        item_id: str,
        cutoff_idx: Optional[int] = None,
        cutoff_timestamp: Optional[pd.Timestamp] = None,
    ) -> List[int]:
        """
        Get indices for an item's interaction history, optionally up to cutoff.

        Args:
            item_id (str): Item ID.
            cutoff_idx (Optional[int]): Cutoff index.
            cutoff_timestamp (Optional[pd.Timestamp]): Cutoff timestamp.

        Returns:
            List[int]: List of indices.
        """
        ids = self.item_index.get(str(item_id), [])
        if cutoff_timestamp is not None and self.has_timestamp:
            prior_ids = [i for i in ids if pd.notna(self.interactions.at[i, "timestamp"]) and self.interactions.at[i, "timestamp"] < cutoff_timestamp]
            return prior_ids
        if cutoff_idx is not None:
            return [i for i in ids if i < cutoff_idx]
        return ids

    def get_user_prior(self, user_id: str) -> Dict[str, float]:
        """Return user prior statistics (computed on training subset)."""
        return self.user_priors.get(str(user_id), {"avg": self.global_mean_stars, "var": 1.0})

    def get_item_prior(self, item_id: str) -> Dict[str, float]:
        """Return item prior statistics (computed on training subset)."""
        return self.item_priors.get(str(item_id), {"avg": self.global_mean_stars, "var": 1.0})

    def refresh(self) -> None:
        """
        Refresh caches/priors (placeholder for more complex updates).
        """
        self.cache_hits = 0
        self.last_refreshed_version = int(time.time())
        # In a real system, we might recompute embeddings or priors here.


# ------------------------------ PersonaProfiler ------------------------------


@dataclass
class PersonaState:
    """Dynamic state for a persona."""
    leniency_drift: float = 0.0
    aspect_preference_weights: Dict[str, float] = field(default_factory=dict)
    style_vector: Dict[str, float] = field(default_factory=dict)
    recent_sentiment_bias: float = 0.0


class PersonaProfiler:
    """
    PersonaProfiler estimates and maintains per-user persona characteristics from history.
    """

    def __init__(
        self,
        data_indexer: DataIndexer,
        neighbor_weight: float = 0.0,
        leniency_drift_rate: float = 0.05,
        verbosity_scale: float = 1.0,
        aspect_weight_decay: float = 0.9,
    ):
        """
        Initialize PersonaProfiler.

        Args:
            data_indexer (DataIndexer): Reference to DataIndexer.
            neighbor_weight (float): Weight for peer influence (0 if no social graph).
            leniency_drift_rate (float): Drift rate for leniency.
            verbosity_scale (float): Scale for verbosity prior.
            aspect_weight_decay (float): EMA decay for aspect weights.
        """
        self.data_indexer = data_indexer
        self.neighbor_weight = float(neighbor_weight)
        self.leniency_drift_rate = float(leniency_drift_rate)
        self.verbosity_scale = float(verbosity_scale)
        self.aspect_weight_decay = float(aspect_weight_decay)

        self.personas: Dict[str, PersonaState] = {}

    def _compute_neighbor_influence(
        self,
        user_id: str,
        user_history_idxs: List[int],
        cutoff_timestamp: Optional[pd.Timestamp],
    ) -> Tuple[float, Dict[str, float]]:
        """
        Approximate neighbor influence as the average behavior of co-reviewers on the same items
        seen in the user's past history. If no neighbors or weight is zero, returns zeros.

        Returns:
            Tuple[float, Dict[str, float]]: (neighbor_leniency, neighbor_aspect_weights)
        """
        if self.neighbor_weight <= 0.0 or not user_history_idxs:
            return 0.0, {}
        idx = self.data_indexer
        neighbor_stars: List[float] = []
        aspect_counts = Counter()
        seen_items = set(str(idx.interactions.loc[i, "item_id"]) for i in user_history_idxs)
        for item_id in seen_items:
            item_hist = idx.get_item_history(item_id=item_id, cutoff_idx=None, cutoff_timestamp=cutoff_timestamp)
            for j in item_hist:
                uid = str(idx.interactions.loc[j, "user_id"])
                if uid != user_id:
                    # neighbor interaction
                    try:
                        neighbor_stars.append(float(idx.interactions.loc[j, "stars"]))
                    except Exception:
                        pass
                    if "review" in idx.interactions.columns:
                        toks = tokenize(str(idx.interactions.loc[j, "review"]))
                        aspect_counts.update([t for t in toks if t in idx.aspect_vocab])
        neighbor_leniency = float(np.mean(neighbor_stars)) if neighbor_stars else idx.global_mean_stars
        total_aspects = sum(aspect_counts.values())
        neighbor_aspect_weights = {a: c / total_aspects for a, c in aspect_counts.items()} if total_aspects > 0 else {}
        return neighbor_leniency, neighbor_aspect_weights

    def build_persona(self, user_id: str, cutoff_idx: Optional[int] = None, cutoff_timestamp: Optional[pd.Timestamp] = None) -> Dict[str, Any]:
        """
        Build or update persona for user_id using history up to cutoff.

        Args:
            user_id (str): The user identifier.
            cutoff_idx (Optional[int]): Cutoff index to avoid leakage when timestamps aren't available.
            cutoff_timestamp (Optional[pd.Timestamp]): Cutoff timestamp to avoid leakage.

        Returns:
            Dict[str, Any]: Persona profile dict.
        """
        user_id = str(user_id)
        prior = self.data_indexer.get_user_prior(user_id)
        history_idxs = self.data_indexer.get_user_history(user_id, cutoff_idx=cutoff_idx, cutoff_timestamp=cutoff_timestamp)
        reviews = [str(self.data_indexer.interactions.loc[i, "review"]) if "review" in self.data_indexer.interactions.columns else "" for i in history_idxs]
        stars = [float(self.data_indexer.interactions.loc[i, "stars"]) for i in history_idxs]

        # Baseline leniency vs global
        baseline_leniency = prior["avg"]
        if len(stars) > 0:
            baseline_leniency = float(np.mean(stars))

        # Verbosity prior from count
        verbosity_prior = self.verbosity_scale * (1.0 + math.log(1 + len(reviews)))

        # Style/tone from simple token cues
        style_vector = {"formal": 0.5, "humorous": 0.5}
        pos_scores = [sentiment_score(r) for r in reviews] if reviews else [0.0]
        recent_sentiment_bias = float(np.mean(pos_scores)) if pos_scores else 0.0

        # Aspect preferences from token frequencies
        aspect_counts = Counter()
        for r in reviews:
            toks = tokenize(r)
            aspect_counts.update([t for t in toks if t in self.data_indexer.aspect_vocab])

        total_aspects = sum(aspect_counts.values())
        if total_aspects > 0:
            aspect_pref = {a: c / total_aspects for a, c in aspect_counts.items()}
        else:
            # Default equal weights on known aspects
            equal = 1.0 / max(1, len(self.data_indexer.aspect_vocab))
            aspect_pref = {a: equal for a in self.data_indexer.aspect_vocab[:10]}

        # Neighbor influence (co-reviewers on items in user's history)
        neighbor_leniency, neighbor_aspects = self._compute_neighbor_influence(
            user_id=user_id,
            user_history_idxs=history_idxs,
            cutoff_timestamp=cutoff_timestamp,
        )

        # Maintain dynamic state
        state = self.personas.get(user_id, PersonaState())
        # EMA update for aspect weights
        new_weights = {}
        all_aspects = set(aspect_pref) | set(state.aspect_preference_weights) | set(neighbor_aspects)
        for a in all_aspects:
            prev = state.aspect_preference_weights.get(a, 0.0)
            cur = aspect_pref.get(a, 0.0)
            neigh = neighbor_aspects.get(a, 0.0)
            blended = (1.0 - self.neighbor_weight) * cur + self.neighbor_weight * neigh
            val = self.aspect_weight_decay * prev + (1 - self.aspect_weight_decay) * blended
            new_weights[a] = val
        state.aspect_preference_weights = new_weights
        # Drift leniency slightly toward recent sentiment
        state.leniency_drift = (1 - self.leniency_drift_rate) * state.leniency_drift + self.leniency_drift_rate * recent_sentiment_bias
        state.style_vector = style_vector
        state.recent_sentiment_bias = recent_sentiment_bias
        self.personas[user_id] = state

        # Compose persona profile dict
        persona_profile = {
            "user_id": user_id,
            "baseline_leniency": (1.0 - self.neighbor_weight) * baseline_leniency + self.neighbor_weight * neighbor_leniency,
            "verbosity_prior": verbosity_prior,
            "tone_style_prior": style_vector,
            "domain_familiarity": len(set(aspect_counts)),  # rough proxy
            "state": state,
        }
        return persona_profile


# ------------------------------ ItemProfiler ------------------------------


@dataclass
class ItemState:
    """Dynamic state for an item."""
    freshness_score: float = 0.0
    controversy: float = 0.0
    aspect_confidence: float = 0.5
    aspect_summary: Dict[str, float] = field(default_factory=dict)


class ItemProfiler:
    """
    ItemProfiler aggregates item-centric history to estimate quality and aspect summaries.
    """

    def __init__(
        self,
        data_indexer: DataIndexer,
        aspect_smoothing_alpha: float = 0.6,
        reputation_inertia: float = 0.8,
        min_reviews_for_confidence: int = 3,
    ):
        """
        Initialize ItemProfiler.

        Args:
            data_indexer (DataIndexer): Reference to DataIndexer.
            aspect_smoothing_alpha (float): Smoothing parameter for aspect summaries.
            reputation_inertia (float): How slowly reputation (quality prior) updates.
            min_reviews_for_confidence (int): Minimum reviews for high confidence.
        """
        self.data_indexer = data_indexer
        self.aspect_smoothing_alpha = float(aspect_smoothing_alpha)
        self.reputation_inertia = float(reputation_inertia)
        self.min_reviews_for_confidence = int(min_reviews_for_confidence)

        self.items: Dict[str, ItemState] = {}

    def build_item_profile(self, item_id: str, cutoff_idx: Optional[int] = None, cutoff_timestamp: Optional[pd.Timestamp] = None) -> Dict[str, Any]:
        """
        Build or update item profile for the given item_id.

        Args:
            item_id (str): Item identifier.
            cutoff_idx (Optional[int]): Cutoff index for history.
            cutoff_timestamp (Optional[pd.Timestamp]): Cutoff timestamp for history.

        Returns:
            Dict[str, Any]: Item profile dictionary with priors and aspect summaries.
        """
        item_id = str(item_id)
        prior = self.data_indexer.get_item_prior(item_id)
        history_idxs = self.data_indexer.get_item_history(item_id, cutoff_idx=cutoff_idx, cutoff_timestamp=cutoff_timestamp)
        reviews = [str(self.data_indexer.interactions.loc[i, "review"]) if "review" in self.data_indexer.interactions.columns else "" for i in history_idxs]
        stars = [float(self.data_indexer.interactions.loc[i, "stars"]) for i in history_idxs]

        # Aspect extraction via token frequency over aspect vocab
        aspect_counts = Counter()
        for r in reviews:
            toks = tokenize(r)
            aspect_counts.update([t for t in toks if t in self.data_indexer.aspect_vocab])

        total_counts = sum(aspect_counts.values())
        raw_summary = {a: c / total_counts for a, c in aspect_counts.items()} if total_counts > 0 else {}

        # Maintain dynamic state with smoothing
        state = self.items.get(item_id, ItemState())
        smoothed = {}
        all_aspects = set(raw_summary) | set(state.aspect_summary)
        for a in all_aspects:
            prev = state.aspect_summary.get(a, 0.0)
            cur = raw_summary.get(a, 0.0)
            smoothed[a] = self.aspect_smoothing_alpha * prev + (1 - self.aspect_smoothing_alpha) * cur
        state.aspect_summary = smoothed

        # Quality/controversy
        if len(stars) >= 2:
            state.controversy = float(np.var(stars))
        else:
            state.controversy = prior["var"]

        # Freshness: recency-weighted activity (approximate)
        state.freshness_score = min(1.0, len(history_idxs) / 10.0)

        # Confidence
        state.aspect_confidence = 1.0 if len(history_idxs) >= self.min_reviews_for_confidence else 0.5

        self.items[item_id] = state

        # Domain tags (very simple): top 3 aspects for this item
        domain_tags = sorted(state.aspect_summary.items(), key=lambda x: x[1], reverse=True)[:3]
        domain_tags = [a for a, _ in domain_tags] if domain_tags else ["general"]

        # Blend quality prior with global mean using reputation inertia
        quality_prior_blended = self.reputation_inertia * prior["avg"] + (1.0 - self.reputation_inertia) * self.data_indexer.global_mean_stars

        profile = {
            "item_id": item_id,
            "quality_prior": quality_prior_blended,
            "variance": prior["var"],
            "aspect_summary": state.aspect_summary,
            "domain_tags": domain_tags,
            "state": state,
        }
        return profile


# ------------------------------ PlanComposer ------------------------------


@dataclass
class PlanState:
    """Dynamic plan state."""
    planned_aspects: List[str] = field(default_factory=list)
    tone_targets: Dict[str, float] = field(default_factory=dict)
    length_target: int = 3


class PlanComposer:
    """
    Compose an aspect and tone plan for a given (user, item) event.
    """

    def __init__(
        self,
        aspect_vocab: List[str],
        aspect_topk: int = 4,
        length_target_mean: int = 3,
        ctx_merge_weight: float = 0.5,
        plan_diversity_temp: float = 0.7,
    ):
        """
        Initialize PlanComposer.

        Args:
            aspect_vocab (List[str]): Global aspect vocabulary.
            aspect_topk (int): Number of aspects to plan.
            length_target_mean (int): Target number of sentences.
            ctx_merge_weight (float): Weight to blend persona vs item aspects.
            plan_diversity_temp (float): Temperature for diversity in selection.
        """
        self.aspect_vocab = aspect_vocab
        self.aspect_topk = int(aspect_topk)
        self.length_target_mean = int(length_target_mean)
        self.ctx_merge_weight = float(ctx_merge_weight)
        self.plan_diversity_temp = float(plan_diversity_temp)
        self.state = PlanState()

    def compose(
        self,
        persona: Dict[str, Any],
        item_profile: Dict[str, Any],
        platform_policy: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Compose an aspect/tone/length plan.

        Args:
            persona (Dict[str, Any]): Persona profile.
            item_profile (Dict[str, Any]): Item profile.
            platform_policy (Dict[str, Any]): Platform policy.

        Returns:
            Dict[str, Any]: Plan object used by ReviewAuthor.
        """
        # Merge aspect preferences: persona vs item aspects
        p_weights = persona["state"].aspect_preference_weights if "state" in persona else {}
        i_weights = item_profile.get("aspect_summary", {})
        all_aspects = set(self.aspect_vocab) | set(p_weights) | set(i_weights)
        scores = {}
        for a in all_aspects:
            pv = p_weights.get(a, 0.0)
            iv = i_weights.get(a, 0.0)
            score = self.ctx_merge_weight * pv + (1.0 - self.ctx_merge_weight) * iv
            scores[a] = score + 1e-6  # avoid zero

        # Diversity via softmax with temperature
        aspects = list(scores.keys())
        vals = np.array([scores[a] for a in aspects], dtype=float)
        vals = vals / max(1e-12, np.max(vals))
        logits = np.log(vals + 1e-8) / max(1e-6, self.plan_diversity_temp)
        probs = np.exp(logits - np.max(logits))
        probs = probs / max(1e-12, probs.sum())
        # Sample top-k without replacement according to probs
        chosen_size = min(self.aspect_topk, len(aspects))
        if chosen_size > 0:
            chosen_idxs = np.random.choice(len(aspects), size=chosen_size, replace=False, p=probs)
            planned_aspects = [aspects[i] for i in chosen_idxs]
        else:
            planned_aspects = []

        # Tone target: shift around persona leniency
        base_leniency = persona.get("baseline_leniency", 3.0)
        tone_bias = persona["state"].recent_sentiment_bias if "state" in persona else 0.0
        tone_target = float((base_leniency - 3.0) / 2.0 + tone_bias)  # approx mapping to [-1,1]
        tone_target = max(-1.0, min(1.0, tone_target))

        # Length target: near mean but within platform cap
        length_target = int(round(self.length_target_mean))
        length_target = max(1, min(length_target, int(platform_policy.get("max_length_sentences", 8))))

        self.state.planned_aspects = planned_aspects
        self.state.length_target = length_target
        self.state.tone_targets = {"sentiment": tone_target}

        plan = {
            "aspects": planned_aspects,
            "tone_targets": self.state.tone_targets,
            "length_target": length_target,
            "planning_rules": "rating-text consistency templates",
        }
        return plan


# ------------------------------ ReviewAuthor (LLM) ------------------------------


class ReviewAuthor:
    """
    Generate a review text using an LLM, conditioned on a structured plan and contexts.
    """

    def __init__(
        self,
        generation_guidelines: str = "Be platform-safe, concise, and specific.",
        style_alignment_weight: float = 0.7,
        llm_temperature: float = 0.5,
        max_revision_loops: int = 1,
        llm_model_name: str = "gpt-5",
        llm_max_output_tokens: int = 512,
        offline_mode: bool = False,
    ):
        """
        Initialize ReviewAuthor.

        Args:
            generation_guidelines (str): Overall generation constraints.
            style_alignment_weight (float): Weight of persona/style alignment in prompt.
            llm_temperature (float): Target temperature (hinted via prompt).
            max_revision_loops (int): Max auto-revision loops allowed.
            llm_model_name (str): OpenAI model name.
            llm_max_output_tokens (int): Max output tokens for LLM.
            offline_mode (bool): If True, do not call LLM; produce a stub review instead.
        """
        self.generation_guidelines = generation_guidelines
        self.style_alignment_weight = float(style_alignment_weight)
        self.llm_temperature = float(llm_temperature)
        self.max_revision_loops = int(max_revision_loops)
        self.llm_model_name = llm_model_name
        self.llm_max_output_tokens = int(llm_max_output_tokens)
        self.last_generated_text_quality_score = 0.0
        self.revision_count = 0
        self.offline_mode = bool(offline_mode)

    def _build_prompt(
        self,
        user_context: Dict[str, Any],
        item_context: Dict[str, Any],
        plan: Dict[str, Any],
        platform_policy: Dict[str, Any],
    ) -> str:
        """
        Construct the LLM prompt combining user context, item context, plan, and policy.

        Args:
            user_context (Dict[str, Any]): Persona data for the user.
            item_context (Dict[str, Any]): Item profile data.
            plan (Dict[str, Any]): Plan including aspects, tone targets, length target.
            platform_policy (Dict[str, Any]): Platform constraints.

        Returns:
            str: Prompt to feed to the LLM.
        """
        style = user_context.get("tone_style_prior", {})
        style_str = ", ".join([f"{k}:{v:.2f}" for k, v in style.items()]) if isinstance(style, dict) else str(style)
        aspects = plan.get("aspects", [])
        tone_target = plan.get("tone_targets", {}).get("sentiment", 0.0)
        length_target = plan.get("length_target", 3)
        domain_tags = item_context.get("domain_tags", ["general"])
        item_quality = item_context.get("quality_prior", 3.0)

        prompt = (
            f"You are writing a user review on an e-commerce platform.\n"
            f"Platform policy: name={platform_policy.get('name','amazon/product')}, "
            f"style_weight={platform_policy.get('style_weight',0.8)}, profanity_allowed={platform_policy.get('profanity',False)}.\n"
            f"Guidelines: {self.generation_guidelines}\n"
            f"User persona: baseline_leniency={user_context.get('baseline_leniency',3.0):.2f}, "
            f"verbosity_prior={user_context.get('verbosity_prior',1.0):.2f}, style_vector={style_str}.\n"
            f"Item context: quality_prior={item_quality:.2f}, domain_tags={', '.join(domain_tags)}.\n"
            f"Plan:\n"
            f"- Aspects to cover (in natural flow): {', '.join(aspects) if aspects else 'general impressions'}.\n"
            f"- Target tone sentiment (in [-1,1]): {tone_target:.2f} (positive means more favorable tone).\n"
            f"- Target number of sentences: {length_target}.\n"
            f"Please produce a coherent review of about {length_target} sentences that addresses the aspects, "
            f"aligns with the target sentiment, and adheres to the platform policy. "
            f"Keep it respectful and specific. "
            f"LLM temperature hint: {self.llm_temperature:.2f}."
        )
        return prompt

    def _offline_stub(self, plan: Dict[str, Any]) -> str:
        aspects = plan.get("aspects", [])
        length_target = int(plan.get("length_target", 3))
        tone = plan.get("tone_targets", {}).get("sentiment", 0.0)
        tone_word = "positive" if tone > 0.2 else "neutral" if tone > -0.2 else "critical"
        if not aspects:
            aspects = ["overall", "quality", "value"]
        sentences = []
        for i in range(length_target):
            a = aspects[i % len(aspects)]
            sentences.append(f"My {tone_word} take on the {a} is that it met expectations for the price.")
        return " ".join(sentences)

    def generate(
        self,
        user_context: Dict[str, Any],
        item_context: Dict[str, Any],
        plan: Dict[str, Any],
        platform_policy: Dict[str, Any],
    ) -> str:
        """
        Generate a review using the OpenAI Responses API based on the assembled prompt or a stub in offline mode.

        Args:
            user_context (Dict[str, Any]): Persona profile.
            item_context (Dict[str, Any]): Item profile.
            plan (Dict[str, Any]): Structured plan.
            platform_policy (Dict[str, Any]): Platform constraints.

        Returns:
            str: Generated review text.

        Raises:
            Exception: If the OpenAI call fails or API key is missing and not offline.
        """
        prompt = self._build_prompt(user_context, item_context, plan, platform_policy)
        if self.offline_mode:
            response = self._offline_stub(plan)
        else:
            response = call_gpt5_with_responses_api(
                prompt=prompt,
                model=self.llm_model_name,
                max_output_tokens=self.llm_max_output_tokens,
                llm_temperature=self.llm_temperature,
            )
        # Trim to target number of sentences
        length_target = int(plan.get("length_target", 3))
        text = trim_to_sentences(response, length_target)
        # Update state heuristically
        self.last_generated_text_quality_score = max(0.0, min(1.0, 0.5 + 0.1 * sentiment_score(text)))
        return text


# ------------------------------ StarRater ------------------------------


class StarRater:
    """
    Map sentiment to star rating with user/item priors and calibrated parameters.
    """

    def __init__(
        self,
        mapping_slope: float = 4.0,
        mapping_intercept: float = 0.0,
        user_bias_weight: float = 0.3,
        item_bias_weight: float = 0.3,
        uncertainty_scale: float = 0.2,
    ):
        """
        Initialize StarRater.

        Args:
            mapping_slope (float): Steepness for logistic mapping.
            mapping_intercept (float): Intercept for logistic mapping.
            user_bias_weight (float): Weight of user leniency prior in final rating.
            item_bias_weight (float): Weight of item reputation prior in final rating.
            uncertainty_scale (float): Gaussian noise scale before rounding.
        """
        self.mapping_slope = float(mapping_slope)
        self.mapping_intercept = float(mapping_intercept)
        self.user_bias_weight = float(user_bias_weight)
        self.item_bias_weight = float(item_bias_weight)
        self.uncertainty_scale = float(uncertainty_scale)

    @staticmethod
    def _sigmoid(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))

    def rate(
        self,
        text_sentiment: float,
        user_leniency_prior: float,
        item_reputation_prior: float,
    ) -> int:
        """
        Compute a star rating from sentiment and priors.

        Args:
            text_sentiment (float): Sentiment score in [-1, 1].
            user_leniency_prior (float): User's prior rating mean in [1, 5].
            item_reputation_prior (float): Item's prior rating mean in [1, 5].

        Returns:
            int: Stars rounded to 1..5.
        """
        base_sig = self._sigmoid(self.mapping_slope * float(text_sentiment) + self.mapping_intercept)
        base_rating = 1.0 + 4.0 * base_sig
        # Blend priors with normalized weights
        w_u = max(0.0, min(1.0, self.user_bias_weight))
        w_i = max(0.0, min(1.0, self.item_bias_weight))
        w_base = max(0.0, 1.0 - (w_u + w_i))
        tot = w_u + w_i + w_base
        if tot > 0:
            w_u /= tot
            w_i /= tot
            w_base /= tot
        pred = w_base * base_rating + w_u * float(user_leniency_prior) + w_i * float(item_reputation_prior)
        # Add noise
        pred += float(np.random.normal(loc=0.0, scale=max(1e-8, self.uncertainty_scale)))
        pred = max(1.0, min(5.0, pred))
        # Deterministic half-up rounding
        stars = int(math.floor(pred + 0.5))
        stars = max(1, min(5, stars))
        return stars


# ------------------------------ QAConsistency ------------------------------


class QAConsistency:
    """
    Check and optionally revise for rating-text consistency and style violations.
    """

    def __init__(
        self,
        consistency_threshold: float = 0.75,
        max_auto_fix_attempts: int = 1,
        penalty_weight_style_violations: float = 0.2,
    ):
        """
        Initialize QAConsistency.

        Args:
            consistency_threshold (float): Threshold for accepting consistency.
            max_auto_fix_attempts (int): Max attempts to auto-revise.
            penalty_weight_style_violations (float): Style violation penalty weight.
        """
        self.consistency_threshold = float(consistency_threshold)
        self.max_auto_fix_attempts = int(max_auto_fix_attempts)
        self.penalty_weight_style_violations = float(penalty_weight_style_violations)
        self.consistency_score = 0.0
        self.auto_fix_attempts = 0

    def _style_penalty(self, text: str, platform_policy: Dict[str, Any]) -> float:
        """
        Compute a simple style penalty based on platform policy.

        Args:
            text (str): Generated text.
            platform_policy (Dict[str, Any]): Policy.

        Returns:
            float: Penalty in [0, 1].
        """
        penalty = 0.0
        if not isinstance(text, str):
            return 1.0
        if platform_policy.get("profanity", False) is False:
            # crude profanity list
            profanity = {"damn", "hell", "shit", "crap"}
            toks = set(tokenize(text))
            if toks & profanity:
                penalty += 0.5
        max_len = int(platform_policy.get("max_length_sentences", 8))
        if sentence_count(text) > max_len:
            penalty += 0.3
        penalty = max(0.0, min(1.0, penalty))
        return penalty

    def assess(
        self,
        text: str,
        stars: int,
        plan: Dict[str, Any],
        platform_policy: Dict[str, Any],
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Assess consistency and style; return score and diagnostics.

        Args:
            text (str): Generated review.
            stars (int): Proposed stars.
            plan (Dict[str, Any]): Plan used.
            platform_policy (Dict[str, Any]): Policy constraints.

        Returns:
            Tuple[float, Dict[str, Any]]: (consistency_score, diagnostics)
        """
        s = sentiment_score(text)
        # Map stars to a rough [-1,1] target: 1->-1, 3->0, 5->1
        star_norm = (float(stars) - 3.0) / 2.0
        consistency = 1.0 - min(1.0, abs(s - star_norm))  # closer better
        style_pen = self._style_penalty(text, platform_policy)
        score = max(0.0, min(1.0, consistency - self.penalty_weight_style_violations * style_pen))
        self.consistency_score = score
        diagnostics = {
            "sentiment": s,
            "star_norm": star_norm,
            "style_penalty": style_pen,
            "consistency_raw": consistency,
            "final_score": score,
            "auto_fix_attempts": self.auto_fix_attempts,
        }
        return score, diagnostics


# ------------------------------ Evaluator ------------------------------


class Evaluator:
    """
    Compute metrics for validation or training evaluation.
    """

    def __init__(self, text_metric_selection: str = "cosine", segment_definitions: Optional[Dict[str, Any]] = None):
        """
        Initialize Evaluator.

        Args:
            text_metric_selection (str): "cosine" similarity on TF vectors.
            segment_definitions (Optional[Dict[str, Any]]): Optional segment definitions; unused for simplicity.
        """
        self.text_metric_selection = text_metric_selection
        self.segment_definitions = segment_definitions or {}
        self.last_eval_metrics: Dict[str, Any] = {}
        self.by_segment_metrics: Dict[str, Any] = {}

    def _text_similarity(self, ref_text: str, gen_text: str, tf_cache: Dict[str, Dict[str, float]]) -> float:
        """Compute semantic similarity via TF cosine with simple cache."""
        def get_tf(text: str) -> Dict[str, float]:
            if text not in tf_cache:
                tf_cache[text] = build_tf(tokenize(text or ""))
            return tf_cache[text]
        tf_ref = get_tf(ref_text or "")
        tf_gen = get_tf(gen_text or "")
        return cosine_sim_from_tfs(tf_ref, tf_gen)

    def compute_metrics(self, df_results: pd.DataFrame, interactions: pd.DataFrame) -> Dict[str, Any]:
        """
        Compute metrics defined in the blueprint.

        Args:
            df_results (pd.DataFrame): DataFrame of simulation results with columns:
                user_id, item_id, gen_text, gen_stars, plan_aspects, plan_length_target,
                consistency_score, optionally: gt_stars, gt_review
            interactions (pd.DataFrame): Original data for reference.

        Returns:
            Dict[str, Any]: Aggregated metrics.
        """
        df = df_results.copy()

        # Normalize numeric columns and mask valid rows
        df["gen_stars"] = pd.to_numeric(df.get("gen_stars", np.nan), errors="coerce")
        gt_stars_series = pd.to_numeric(df.get("gt_stars", np.nan), errors="coerce")
        star_mask = df["gen_stars"].notna() & gt_stars_series.notna()

        # Stars metrics
        if star_mask.any():
            diffs = (df.loc[star_mask, "gen_stars"].astype(float) - gt_stars_series.loc[star_mask].astype(float)).values
            rmse = float(np.sqrt(np.mean(np.square(diffs)))) if len(diffs) > 0 else float("nan")
            mae = float(np.mean(np.abs(diffs))) if len(diffs) > 0 else float("nan")
        else:
            rmse = float("nan")
            mae = float("nan")

        # Text similarity
        tf_cache: Dict[str, Dict[str, float]] = {}
        text_sims = []
        gt_reviews = df.get("gt_review", pd.Series([""] * len(df)))
        for r, g in zip(gt_reviews.tolist(), df.get("gen_text", pd.Series([""] * len(df))).tolist()):
            if isinstance(r, str) and r.strip():
                text_sims.append(self._text_similarity(str(r), str(g), tf_cache))
        text_similarity = float(np.mean(text_sims)) if text_sims else float("nan")

        # Sentiment agreement
        sent_agree = []
        for gtext, gstars in zip(df.get("gen_text", pd.Series([])).tolist(), df.get("gen_stars", pd.Series([])).tolist()):
            try:
                s = sentiment_score(str(gtext))
                star_norm = (float(gstars) - 3.0) / 2.0
                agree = 1.0 - min(1.0, abs(s - star_norm))
                sent_agree.append(agree)
            except Exception:
                continue
        sentiment_agreement = float(np.mean(sent_agree)) if sent_agree else float("nan")

        # Aspect coverage
        coverage = []
        for aspects, gtext in zip(df.get("plan_aspects", pd.Series([[]] * len(df))).tolist(), df.get("gen_text", pd.Series([""] * len(df))).tolist()):
            if not isinstance(aspects, list) or not aspects:
                coverage.append(0.0)
            else:
                toks = set(tokenize(str(gtext)))
                matched = sum(1 for a in aspects if a in toks)
                coverage.append(matched / float(len(aspects)))
        aspect_coverage = float(np.mean(coverage)) if coverage else float("nan")

        # Consistency score
        consistency_score = float(df.get("consistency_score", pd.Series([])).mean()) if len(df) > 0 else float("nan")

        # Length deviation
        len_dev = []
        for tgt, gtext in zip(df.get("plan_length_target", pd.Series([])).tolist(), df.get("gen_text", pd.Series([])).tolist()):
            try:
                dev = abs(int(tgt) - sentence_count(str(gtext)))
                len_dev.append(dev)
            except Exception:
                continue
        length_deviation = float(np.mean(len_dev)) if len_dev else float("nan")

        metrics = {
            "RMSE_stars": rmse,
            "MAE_stars": mae,
            "Text_Similarity": text_similarity,
            "Sentiment_Agreement": sentiment_agreement,
            "Aspect_Coverage": aspect_coverage,
            "Consistency_Score": consistency_score,
            "Length_Deviation": length_deviation,
            "count": int(len(df)),
        }
        self.last_eval_metrics = metrics
        # By-segment metrics (user/item frequency tertiles)
        self.by_segment_metrics = self._compute_segments(df)
        metrics["by_segments"] = self.by_segment_metrics
        return metrics

    @staticmethod
    def _compute_segments(df_results: pd.DataFrame) -> Dict[str, Any]:
        """Compute metrics by user/item frequency tertiles with NaN-safe handling."""
        segs = {}
        if len(df_results) == 0:
            return segs
        # Build frequencies
        user_counts = df_results.groupby("user_id").size()
        item_counts = df_results.groupby("item_id").size()

        def tertile_label(count: int, series: pd.Series) -> str:
            if series is None or len(series) == 0:
                return "all"
            q1 = series.quantile(1 / 3.0)
            q2 = series.quantile(2 / 3.0)
            try:
                if count <= q1:
                    return "low"
                elif count <= q2:
                    return "mid"
                else:
                    return "high"
            except Exception:
                return "all"

        df = df_results.copy()
        df["user_seg"] = df["user_id"].map(lambda u: tertile_label(user_counts.get(u, 0), user_counts))
        df["item_seg"] = df["item_id"].map(lambda i: tertile_label(item_counts.get(i, 0), item_counts))

        def agg(group: pd.DataFrame) -> Dict[str, float]:
            out = {}
            if len(group) == 0:
                return out
            # Safe numeric coercion
            gen = pd.to_numeric(group.get("gen_stars", np.nan), errors="coerce")
            gt = pd.to_numeric(group.get("gt_stars", np.nan), errors="coerce")
            mask = gen.notna() & gt.notna()
            if mask.any():
                diffs = (gen[mask].astype(float) - gt[mask].astype(float)).values
                out["RMSE_stars"] = float(np.sqrt(np.mean(np.square(diffs))))
                out["MAE_stars"] = float(np.mean(np.abs(diffs)))
            else:
                out["RMSE_stars"] = float("nan")
                out["MAE_stars"] = float("nan")
            # Text similarity with simple cache
            tf_cache: Dict[str, Dict[str, float]] = {}
            sims = []
            for r, g in zip(group.get("gt_review", pd.Series([])).tolist(), group.get("gen_text", pd.Series([])).tolist()):
                if isinstance(r, str) and r.strip():
                    tf_r = build_tf(tokenize(str(r)))
                    tf_g = build_tf(tokenize(str(g)))
                    sims.append(cosine_sim_from_tfs(tf_r, tf_g))
            out["Text_Similarity"] = float(np.mean(sims)) if sims else float("nan")
            out["Consistency_Score"] = float(group.get("consistency_score", pd.Series([])).mean())
            out["n"] = int(len(group))
            return out

        for seg in ["user_seg", "item_seg"]:
            for lab in ["low", "mid", "high", "all"]:
                g = df[df[seg] == lab] if lab != "all" else df
                segs[f"{seg}:{lab}"] = agg(g)
        return segs


# ------------------------------ Simulator ------------------------------


class Simulator:
    """
    Orchestrates per-record simulation using the agents.
    """

    def __init__(
        self,
        data_indexer: DataIndexer,
        persona_profiler: PersonaProfiler,
        item_profiler: ItemProfiler,
        plan_composer: PlanComposer,
        review_author: ReviewAuthor,
        star_rater: StarRater,
        qa_consistency: QAConsistency,
        evaluator: Evaluator,
        platform_policy: Dict[str, Any],
    ):
        """
        Initialize Simulator with agents.

        Args:
            data_indexer (DataIndexer): Data indexer.
            persona_profiler (PersonaProfiler): Persona profiler.
            item_profiler (ItemProfiler): Item profiler.
            plan_composer (PlanComposer): Planner.
            review_author (ReviewAuthor): Review author (LLM).
            star_rater (StarRater): Star rater.
            qa_consistency (QAConsistency): Consistency checker.
            evaluator (Evaluator): Evaluator.
            platform_policy (Dict[str, Any]): Platform policy dict.
        """
        self.idx = data_indexer
        self.persona = persona_profiler
        self.itemp = item_profiler
        self.planner = plan_composer
        self.author = review_author
        self.rater = star_rater
        self.qa = qa_consistency
        self.evaluator = evaluator
        self.platform_policy = platform_policy

    def rollout(self, df_subset: pd.DataFrame, keep_ground_truth: bool = True) -> pd.DataFrame:
        """
        Run forward simulation on a subset of interactions.

        Args:
            df_subset (pd.DataFrame): The interactions subset to simulate.
            keep_ground_truth (bool): If True, include GT text and stars for evaluation.

        Returns:
            pd.DataFrame: Results including generated text, stars, and diagnostics.
        """
        results = []
        # To avoid leakage, determine iteration order by timestamp if available else index
        if "timestamp" in df_subset.columns and pd.api.types.is_datetime64_any_dtype(df_subset["timestamp"]):
            df_iter = df_subset.sort_values("timestamp").copy()
            use_time = True
        else:
            df_iter = df_subset.copy()
            use_time = False

        for idx, row in df_iter.iterrows():
            user_id = str(row["user_id"])
            item_id = str(row["item_id"])
            cutoff_idx = idx if not use_time else None
            cutoff_ts = row["timestamp"] if use_time else None

            # Build contexts
            persona_profile = self.persona.build_persona(user_id=user_id, cutoff_idx=cutoff_idx, cutoff_timestamp=cutoff_ts)
            item_profile = self.itemp.build_item_profile(item_id=item_id, cutoff_idx=cutoff_idx, cutoff_timestamp=cutoff_ts)

            # Compose plan
            plan = self.planner.compose(persona_profile, item_profile, self.platform_policy)

            # Author review (LLM)
            text = self.author.generate(user_context=persona_profile, item_context=item_profile, plan=plan, platform_policy=self.platform_policy)

            # Map to stars
            s = sentiment_score(text)
            stars = self.rater.rate(
                text_sentiment=s,
                user_leniency_prior=persona_profile.get("baseline_leniency", 3.0),
                item_reputation_prior=item_profile.get("quality_prior", 3.0),
            )

            # QA Consistency and optional revisions
            score, diag = self.qa.assess(text, stars, plan, self.platform_policy)
            attempts = 0
            self.qa.auto_fix_attempts = 0
            max_attempts = min(self.qa.max_auto_fix_attempts, self.author.max_revision_loops)
            while score < self.qa.consistency_threshold and attempts < max_attempts:
                # Adjust tone target to better align with star_norm
                desired = (stars - 3.0) / 2.0
                plan["tone_targets"]["sentiment"] = float(desired)
                # Regenerate
                text = self.author.generate(user_context=persona_profile, item_context=item_profile, plan=plan, platform_policy=self.platform_policy)
                s = sentiment_score(text)
                stars = self.rater.rate(
                    text_sentiment=s,
                    user_leniency_prior=persona_profile.get("baseline_leniency", 3.0),
                    item_reputation_prior=item_profile.get("quality_prior", 3.0),
                )
                attempts += 1
                self.qa.auto_fix_attempts = attempts
                score, diag = self.qa.assess(text, stars, plan, self.platform_policy)

            trace = {
                "user_id": user_id,
                "item_id": item_id,
                "gen_text": text,
                "gen_stars": stars,
                "plan_aspects": plan.get("aspects", []),
                "plan_length_target": plan.get("length_target", 3),
                "consistency_score": score,
                "consistency_diag": diag,
                "tone_target": plan.get("tone_targets", {}).get("sentiment", 0.0),
            }
            if keep_ground_truth:
                trace["gt_stars"] = int(row["stars"]) if not pd.isna(row["stars"]) else None
                trace["gt_review"] = str(row["review"]) if "review" in row and isinstance(row["review"], (str,)) else ""
            results.append(trace)

        df_res = pd.DataFrame(results)
        return df_res


# ------------------------------ ParameterTuner (Calibration) ------------------------------


class ParameterTuner:
    """
    Random search tuner with early stopping to calibrate simulator parameters.
    """

    def __init__(
        self,
        simulator_builder,
        data_indexer: DataIndexer,
        num_trials: int = 5,
        early_stop_patience: int = 2,
        objective_weights: Optional[Dict[str, float]] = None,
        train_sample_size: int = 30,
        seed: int = 42,
        llm_model_name: str = "gpt-5",
        llm_max_output_tokens: int = 512,
    ):
        """
        Initialize ParameterTuner.

        Args:
            simulator_builder (callable): Function accepting a params dict and returning a Simulator.
            data_indexer (DataIndexer): Data indexer.
            num_trials (int): Maximum number of trials.
            early_stop_patience (int): Early stopping patience.
            objective_weights (Optional[Dict[str, float]]): Weights for objective components; normalized if provided.
            train_sample_size (int): Number of training records sampled per trial.
            seed (int): Random seed.
            llm_model_name (str): OpenAI model to use (kept for compatibility).
            llm_max_output_tokens (int): Max tokens for generation (kept for compatibility).
        """
        self.simulator_builder = simulator_builder
        self.idx = data_indexer
        self.num_trials = int(num_trials)
        self.early_stop_patience = int(early_stop_patience)
        self.objective_weights = self._normalize_weights(objective_weights or {"stars": 0.6, "text": 0.3, "consistency": 0.1})
        self.train_sample_size = int(train_sample_size)
        self.seed = int(seed)
        self.llm_model_name = llm_model_name
        self.llm_max_output_tokens = int(llm_max_output_tokens)

        self.current_params: Dict[str, Any] = {}
        self.best_params: Dict[str, Any] = {}
        self.history_of_trials: List[Dict[str, Any]] = []

    @staticmethod
    def _normalize_weights(w: Dict[str, float]) -> Dict[str, float]:
        tot = float(sum(max(0.0, v) for v in w.values()))
        if tot <= 0:
            return {"stars": 0.6, "text": 0.3, "consistency": 0.1}
        return {k: float(v) / tot for k, v in w.items()}

    def _sample_param_space(self) -> Dict[str, Any]:
        """
        Sample a parameter configuration within specified bounds.

        Returns:
            Dict[str, Any]: Parameter dict.
        """
        # Calibratable parameters with bounds
        p = {}
        p["neighbor_weight"] = float(np.random.uniform(0.0, 0.5))
        p["ctx_merge_weight"] = float(np.random.uniform(0.2, 0.8))
        p["aspect_topk"] = int(round(np.random.uniform(3, 6)))
        p["length_target_mean"] = int(round(np.random.uniform(2, 6)))
        p["plan_diversity_temp"] = float(np.random.uniform(0.3, 1.2))
        p["llm_temperature"] = float(np.random.uniform(0.2, 0.9))
        p["style_alignment_weight"] = float(np.random.uniform(0.3, 1.0))
        p["mapping_slope"] = float(np.random.uniform(2.0, 8.0))
        p["mapping_intercept"] = float(np.random.uniform(-2.0, 2.0))
        p["user_bias_weight"] = float(np.random.uniform(0.0, 1.0))
        p["item_bias_weight"] = float(np.random.uniform(0.0, 1.0))
        p["uncertainty_scale"] = float(np.random.uniform(0.1, 1.0))
        p["consistency_threshold"] = float(np.random.uniform(0.6, 0.9))
        p["max_auto_fix_attempts"] = int(round(np.random.uniform(0, 2)))
        # Objective weights with normalization
        stars_w = float(np.random.uniform(0.4, 0.8))
        text_w = float(np.random.uniform(0.2, 0.6))
        cons_w = float(np.random.uniform(0.0, 0.3))
        tot = stars_w + text_w + cons_w
        p["objective_weights"] = {"stars": stars_w / tot, "text": text_w / tot, "consistency": cons_w / tot}
        return p

    def _objective(self, df_sim: pd.DataFrame, interactions: pd.DataFrame, weights: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
        """
        Compute objective value (lower is better) combining star error, text loss, and 1 - consistency.

        Args:
            df_sim (pd.DataFrame): Simulation results on train subset.
            interactions (pd.DataFrame): Reference interactions dataframe.
            weights (Dict[str, float]): Component weights.

        Returns:
            Tuple[float, Dict[str, float]]: (objective_value, components)
        """
        evaluator = Evaluator()
        metrics = evaluator.compute_metrics(df_sim, interactions)
        # Components: we turn metrics into losses (lower is better)
        stars_loss = metrics.get("RMSE_stars", 0.0)
        text_loss = 1.0 - metrics.get("Text_Similarity", 0.0)
        cons_loss = 1.0 - metrics.get("Consistency_Score", 0.0)
        obj = weights.get("stars", 0.6) * stars_loss + weights.get("text", 0.3) * text_loss + weights.get("consistency", 0.1) * cons_loss
        comps = {"stars_loss": stars_loss, "text_loss": text_loss, "consistency_loss": cons_loss}
        return obj, comps

    def fit(self, train_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Run calibration over the training split.

        Args:
            train_df (pd.DataFrame): Training interactions.

        Returns:
            Dict[str, Any]: Best found parameters.
        """
        # Sample a subset of training to control LLM cost
        if len(train_df) == 0:
            raise ValueError("Training DataFrame is empty; cannot calibrate.")
        sample_n = min(self.train_sample_size, len(train_df))
        train_sample = train_df.sample(n=sample_n, random_state=self.seed).copy()  # type: ignore

        best_obj = float("inf")
        best_params = None
        no_improve_rounds = 0
        trial = 0

        while trial < self.num_trials:
            params = self._sample_param_space()
            self.current_params = params
            # Build simulator with current params
            sim = self.simulator_builder(params)
            # Rollout on sampled training subset
            df_sim = sim.rollout(train_sample, keep_ground_truth=True)
            # Compute objective
            weights = params.get("objective_weights", self.objective_weights)
            obj, comps = self._objective(df_sim, self.idx.interactions, weights)
            self.history_of_trials.append({"trial": trial, "params": params, "objective": obj, "components": comps, "count": len(train_sample)})

            if obj < best_obj:
                best_obj = obj
                best_params = params
                no_improve_rounds = 0
            else:
                no_improve_rounds += 1

            if no_improve_rounds >= self.early_stop_patience:
                break
            trial += 1

        if best_params is None:
            # Fallback to defaults if not improved
            best_params = self._sample_param_space()
        self.best_params = best_params
        return best_params


# ------------------------------ Data I/O ------------------------------


def parse_cli() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Multi-agent review simulation and calibration.")
    parser.add_argument("--max_trials", type=int, default=3, help="Maximum calibration trials.")
    parser.add_argument("--early_stop_patience", type=int, default=2, help="Early stopping patience.")
    parser.add_argument("--train_sample_size", type=int, default=20, help="Number of training examples per trial.")
    parser.add_argument("--seed", type=int, default=42, help="Global random seed.")
    parser.add_argument("--llm_model", type=str, default="gpt-5", help="OpenAI model name for LLM calls.")
    parser.add_argument("--llm_max_output_tokens", type=int, default=512, help="Max output tokens for LLM.")
    parser.add_argument("--skip_ablation", action="store_true", help="Skip optional ablation report.")
    parser.add_argument("--offline", action="store_true", help="Run without calling LLM (stub generator).")
    parser.add_argument("--data_dir", type=str, default=None, help="Data directory override (contains interactions.csv).")
    args = parser.parse_args()
    return args


def load_data(data_dir: Optional[str] = None) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame], str]:
    """
    Load interactions and optional user/item files from data_dir.

    Expected files:
      - interactions.csv with columns: user_id, item_id, stars, review
        Optional: datatype (train/test), timestamp (ISO8601 or sortable), source/type

      - users.csv (optional)
      - items.csv (optional)

    Args:
        data_dir (Optional[str]): Directory override.

    Returns:
        Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame], str]:
            (interactions, users, items, resolved_data_dir)

    Raises:
        ValueError: If data_dir invalid or interactions.csv is missing/invalid.
    """
    resolved = ensure_data_dir_valid(data_dir)
    interactions_file = os.path.join(resolved, "interactions.csv")
    users_file = os.path.join(resolved, "users.csv")
    items_file = os.path.join(resolved, "items.csv")

    if not os.path.isfile(interactions_file):
        raise ValueError(f"Required file not found: {interactions_file}")

    interactions = pd.read_csv(interactions_file)
    # Validate required columns
    required_cols = {"user_id", "item_id", "stars"}
    missing = required_cols - set(interactions.columns)
    if missing:
        raise ValueError(f"interactions.csv missing required columns: {missing}")

    # Normalize types
    interactions["user_id"] = interactions["user_id"].astype(str)
    interactions["item_id"] = interactions["item_id"].astype(str)
    interactions["stars"] = pd.to_numeric(interactions["stars"], errors="coerce").astype(float)
    if "review" not in interactions.columns:
        interactions["review"] = ""
    else:
        interactions["review"] = interactions["review"].fillna("").astype(str)

    # Timestamps handling: ensure sortable
    if "timestamp" in interactions.columns:
        try:
            interactions["timestamp"] = pd.to_datetime(interactions["timestamp"])
        except Exception:
            pass

    users = pd.read_csv(users_file) if os.path.isfile(users_file) else None
    items = pd.read_csv(items_file) if os.path.isfile(items_file) else None

    return interactions, users, items, resolved


def build_network_and_agents(
    interactions: pd.DataFrame,
    users: Optional[pd.DataFrame],
    items: Optional[pd.DataFrame],
    params: Optional[Dict[str, Any]] = None,
    llm_model_name: str = "gpt-5",
    llm_max_output_tokens: int = 512,
    prior_indices: Optional[Iterable[int]] = None,
    offline_mode: bool = False,
) -> Tuple[DataIndexer, Simulator]:
    """
    Build the data indexer and all agents; return a configured Simulator.

    Args:
        interactions (pd.DataFrame): Interactions data.
        users (Optional[pd.DataFrame]): Users data.
        items (Optional[pd.DataFrame]): Items data.
        params (Optional[Dict[str, Any]]): Calibrated parameter overrides.
        llm_model_name (str): LLM model name for ReviewAuthor.
        llm_max_output_tokens (int): LLM max output tokens.
        prior_indices (Optional[Iterable[int]]): Indices for computing priors (train split).
        offline_mode (bool): If True, author will not call LLM.

    Returns:
        Tuple[DataIndexer, Simulator]: (data_indexer, simulator)
    """
    params = params or {}

    # DataIndexer with train-only priors to avoid leakage
    idx = DataIndexer(interactions, users, items, prior_indices=prior_indices)

    # PersonaProfiler
    persona = PersonaProfiler(
        data_indexer=idx,
        neighbor_weight=float(params.get("neighbor_weight", 0.0)),
        leniency_drift_rate=0.05,
        verbosity_scale=1.0,
        aspect_weight_decay=0.9,
    )

    # ItemProfiler
    itemp = ItemProfiler(
        data_indexer=idx,
        aspect_smoothing_alpha=0.6,
        reputation_inertia=0.8,
        min_reviews_for_confidence=3,
    )

    # PlanComposer
    planner = PlanComposer(
        aspect_vocab=idx.aspect_vocab,
        aspect_topk=int(params.get("aspect_topk", 4)),
        length_target_mean=int(params.get("length_target_mean", 3)),
        ctx_merge_weight=float(params.get("ctx_merge_weight", 0.5)),
        plan_diversity_temp=float(params.get("plan_diversity_temp", 0.7)),
    )

    # ReviewAuthor (LLM integration)
    author = ReviewAuthor(
        generation_guidelines="Be platform-safe, concise, specific, and avoid profanity.",
        style_alignment_weight=float(params.get("style_alignment_weight", 0.7)),
        llm_temperature=float(params.get("llm_temperature", 0.5)),
        max_revision_loops=int(params.get("max_auto_fix_attempts", 1)),
        llm_model_name=llm_model_name,
        llm_max_output_tokens=int(llm_max_output_tokens),
        offline_mode=offline_mode,
    )

    # StarRater
    rater = StarRater(
        mapping_slope=float(params.get("mapping_slope", 4.0)),
        mapping_intercept=float(params.get("mapping_intercept", 0.0)),
        user_bias_weight=float(params.get("user_bias_weight", 0.3)),
        item_bias_weight=float(params.get("item_bias_weight", 0.3)),
        uncertainty_scale=float(params.get("uncertainty_scale", 0.2)),
    )

    # QAConsistency
    qa = QAConsistency(
        consistency_threshold=float(params.get("consistency_threshold", 0.75)),
        max_auto_fix_attempts=int(params.get("max_auto_fix_attempts", 1)),
        penalty_weight_style_violations=0.2,
    )

    # Evaluator
    evaluator = Evaluator(text_metric_selection="cosine")

    simulator = Simulator(
        data_indexer=idx,
        persona_profiler=persona,
        item_profiler=itemp,
        plan_composer=planner,
        review_author=author,
        star_rater=rater,
        qa_consistency=qa,
        evaluator=evaluator,
        platform_policy=idx.platform_policy,
    )
    return idx, simulator


def holdout_split(interactions: pd.DataFrame, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Partition the dataset into train and validation sets following the blueprint:
    - If 'datatype' labels exist, use datatype == 'train' vs 'test' (case-insensitive).
    - Else, if timestamps exist, use temporal split: first 80% by date as train, last 20% as validation.
    - Otherwise, random 80/20 split with fixed seed.

    Args:
        interactions (pd.DataFrame): Interactions DataFrame.
        seed (int): Random seed for random split.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (train_df, val_df)
    """
    df = interactions.copy()
    if "datatype" in df.columns:
        dtype = df["datatype"].astype(str).str.lower()
        train_df = df[dtype == "train"].copy()
        test_df = df[dtype.isin(["test", "val", "validation"])].copy()
        if len(train_df) > 0 and len(test_df) > 0:
            return train_df, test_df
        # Fallback if labels incomplete: random
    if "timestamp" in df.columns and pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df_sorted = df.sort_values("timestamp")
        n = len(df_sorted)
        cut = int(0.8 * n)
        return df_sorted.iloc[:cut].copy(), df_sorted.iloc[cut:].copy()
    # Random split
    np.random.seed(seed)
    msk = np.random.rand(len(df)) < 0.8
    return df[msk].copy(), df[~msk].copy()


def save_results(
    calibrated_params: Dict[str, Any],
    data_indexer: DataIndexer,
    df_val_results: pd.DataFrame,
    eval_metrics: Dict[str, Any],
    data_dir: Optional[str] = None,
    ablation_report: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save outputs to data_dir:
    - calibrated_parameters.json
    - simulation_traces.jsonl (validation results)
    - evaluation_metrics.json
    - ablation_report.json (optional)

    Args:
        calibrated_params (Dict[str, Any]): Best parameter set.
        data_indexer (DataIndexer): Data indexer to extract priors.
        df_val_results (pd.DataFrame): Validation results dataframe.
        eval_metrics (Dict[str, Any]): Evaluation metrics.
        data_dir (Optional[str]): Destination directory override.
        ablation_report (Optional[Dict[str, Any]]): Ablation results.
    """
    resolved_dir = ensure_data_dir_valid(data_dir)

    # Prepare priors summary to avoid writing extremely large files
    user_priors_sample = dict(list(data_indexer.user_priors.items())[:100])  # limit to 100 users
    item_priors_sample = dict(list(data_indexer.item_priors.items())[:100])  # limit to 100 items
    calib_out = {
        "calibrated_parameters": calibrated_params,
        "global_mean_stars": data_indexer.global_mean_stars,
        "aspect_vocab": data_indexer.aspect_vocab,
        "user_priors_sample": user_priors_sample,
        "item_priors_sample": item_priors_sample,
        "timestamp": int(time.time()),
    }
    calibrated_params_file = os.path.join(resolved_dir, "calibrated_parameters.json")
    safe_json_dump(calib_out, calibrated_params_file)

    # Traces (validation)
    sim_traces_file = os.path.join(resolved_dir, "simulation_traces.jsonl")
    safe_jsonl_dump(df_val_results.to_dict(orient="records"), sim_traces_file)

    # Metrics
    metrics_file = os.path.join(resolved_dir, "evaluation_metrics.json")
    safe_json_dump(eval_metrics, metrics_file)

    # Ablation (optional)
    if ablation_report is not None:
        ablation_file = os.path.join(resolved_dir, "ablation_report.json")
        safe_json_dump(ablation_report, ablation_file)


def _preflight_openai(offline: bool = False) -> None:
    """
    Early validation for OpenAI availability and API key to fail fast, unless offline mode.
    """
    if offline:
        return
    if not _OPENAI_AVAILABLE or OpenAI is None:
        raise RuntimeError(
            "OpenAI SDK not available. Install with: pip install openai. "
            f"Original import error: {_OPENAI_IMPORT_ERROR}"
        )
    # Validate API key presence
    api_key = get_openai_api_key()
    # Ensure env var and instantiate client
    if os.environ.get("OPENAI_API_KEY") != api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    _ = OpenAI()


# ------------------------------ Orchestrator (main) ------------------------------


def main() -> None:
    """
    Orchestrate the full pipeline:
    parse_cli() -> load_data() -> holdout_split() -> build_network_and_agents(train priors) ->
    calibrator.fit() -> simulator.rollout() -> evaluator.compute_metrics() -> save_results()
    """
    try:
        args = parse_cli()
        set_global_seed(args.seed)

        # Preflight OpenAI unless offline mode
        _preflight_openai(offline=args.offline)

        # Load data
        interactions, users, items, resolved_dir = load_data(args.data_dir)

        # Holdout split first to avoid leakage when computing priors
        train_df, val_df = holdout_split(interactions, seed=args.seed)
        if len(val_df) == 0:
            raise ValueError("Validation split is empty; please provide datatype labels or sufficient data for split.")

        # Build initial network and agents with train-only priors
        idx, simulator = build_network_and_agents(
            interactions=interactions,
            users=users,
            items=items,
            params=None,
            llm_model_name=args.llm_model,
            llm_max_output_tokens=args.llm_max_output_tokens,
            prior_indices=train_df.index,
            offline_mode=args.offline,
        )

        # Calibrator: builder to re-create simulator with different params (always using train priors)
        def simulator_builder(p: Dict[str, Any]) -> Simulator:
            _, sim = build_network_and_agents(
                interactions=interactions,
                users=users,
                items=items,
                params=p,
                llm_model_name=args.llm_model,
                llm_max_output_tokens=args.llm_max_output_tokens,
                prior_indices=train_df.index,
                offline_mode=args.offline,
            )
            return sim

        calibrator = ParameterTuner(
            simulator_builder=simulator_builder,
            data_indexer=idx,
            num_trials=args.max_trials,
            early_stop_patience=args.early_stop_patience,
            objective_weights=None,  # use per-sample params
            train_sample_size=args.train_sample_size,
            seed=args.seed,
            llm_model_name=args.llm_model,
            llm_max_output_tokens=args.llm_max_output_tokens,
        )

        # Fit/calibrate on training split
        best_params = calibrator.fit(train_df=train_df)

        # Freeze parameters and run simulator on validation split
        _, tuned_simulator = build_network_and_agents(
            interactions=interactions,
            users=users,
            items=items,
            params=best_params,
            llm_model_name=args.llm_model,
            llm_max_output_tokens=args.llm_max_output_tokens,
            prior_indices=train_df.index,
            offline_mode=args.offline,
        )
        df_val_results = tuned_simulator.rollout(val_df, keep_ground_truth=True)

        # Evaluate
        evaluator = Evaluator()
        metrics = evaluator.compute_metrics(df_val_results, interactions)

        # Optional ablation: compare with a different plan diversity temperature
        ablation_report = None
        if not args.skip_ablation:
            ablation_params = dict(best_params)
            # Simple ablation: reduce plan diversity temp
            ablation_params["plan_diversity_temp"] = max(0.3, float(best_params.get("plan_diversity_temp", 0.7)) * 0.7)
            _, ablation_sim = build_network_and_agents(
                interactions=interactions,
                users=users,
                items=items,
                params=ablation_params,
                llm_model_name=args.llm_model,
                llm_max_output_tokens=args.llm_max_output_tokens,
                prior_indices=train_df.index,
                offline_mode=args.offline,
            )
            df_ablation = ablation_sim.rollout(val_df, keep_ground_truth=True)
            ablation_metrics = evaluator.compute_metrics(df_ablation, interactions)
            ablation_report = {
                "baseline_params": best_params,
                "ablation_params": ablation_params,
                "baseline_metrics": metrics,
                "ablation_metrics": ablation_metrics,
            }

        # Save results
        save_results(
            calibrated_params=best_params,
            data_indexer=idx,
            df_val_results=df_val_results,
            eval_metrics=metrics,
            data_dir=resolved_dir,
            ablation_report=ablation_report,
        )

        print("Simulation complete.")
        print(f"- Calibrated parameters saved to: {os.path.join(resolved_dir, 'calibrated_parameters.json')}")
        print(f"- Validation traces saved to: {os.path.join(resolved_dir, 'simulation_traces.jsonl')}")
        print(f"- Evaluation metrics saved to: {os.path.join(resolved_dir, 'evaluation_metrics.json')}")
        if ablation_report is not None:
            print(f"- Ablation report saved to: {os.path.join(resolved_dir, 'ablation_report.json')}")

    except Exception as e:
        # Provide actionable message
        sys.stderr.write(f"ERROR: {e}\n")
        sys.exit(1)



# Execute main for both direct execution and sandbox wrapper invocation
main()