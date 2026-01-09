#!/usr/bin/env python3
"""
simulate_star_rating.py

Specialized star rating prediction system with LLM + post-processing calibration.
Focuses solely on optimizing star rating accuracy using existing review text.

Architecture:
1. LLM Prediction Phase (Fixed):
   - Uses fixed temperature (0.2) and max tokens (64)
   - Fixed prompt structure with product context and few-shot examples
   - Returns raw integer prediction (1-5)
   - LLM results are cached for reproducibility

2. Post-Processing Calibration Phase (Tunable):
   - linear_bias: Additive bias correction
   - sentiment_weight: Weight for sentiment-based adjustment
   - user_bias_weight: Weight for user leniency prior
   - item_bias_weight: Weight for item reputation prior
   - uncertainty_scale: Gaussian noise scale for stochastic exploration
   - These parameters are calibrated to minimize MAE on training data

Key features:
- Data ingestion from environment-configured absolute paths (PROJECT_ROOT/DATA_PATH).
- Deterministic behavior via global random seed.
- Temporal/random holdout splitting with training/validation sets.
- Parameter calibration for post-processing layer only (LLM prompt unchanged).
- Pluggable calibrators, including random search and evolutionary strategy.
- Metrics: MAE and RMSE for star rating accuracy.
- OpenAI LLM integration using the Responses API for rating prediction.

Run:
  python simulate_star_rating.py --seed 42 --num-trials 20 --max-records 200

Environment variables:
  - PROJECT_ROOT: Absolute path to project root directory.
  - DATA_PATH: Path relative to project root for data directory.
  - OPENAI_API_KEY: Required (unless OFFLINE_MODE=1) to call OpenAI Responses API.
  - OPENAI_MODEL: Optional, default "gpt-4.1-mini".
  - OFFLINE_MODE: If "1", uses deterministic offline stubs for rating.

Outputs (written under DATA_DIR):
  - calibrated_parameters.json (post-processing parameters only)
  - evaluation_metrics.json (MAE/RMSE metrics only)
"""

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import random
import re
import statistics
import sys
import time
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple
from pathlib import Path

import numpy as np

# Path handling per instructions
PROJECT_ROOT = os.environ.get("PROJECT_ROOT") or os.getcwd()
DATA_PATH = os.environ.get("DATA_PATH") or "data"
DATA_DIR = os.path.join(PROJECT_ROOT, DATA_PATH)

# Optional OpenAI import with guard
try:
    from openai import OpenAI  # type: ignore
except Exception:
    OpenAI = None  # Will be validated if LLM is requested

# Optional imports for evaluation metrics
try:
    from transformers import pipeline as transformers_pipeline  # type: ignore
    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False

try:
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer  # type: ignore
    # Download required NLTK data if not available
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        try:
            nltk.download('punkt', quiet=True)
        except Exception:
            pass
    try:
        nltk.data.find('vader_lexicon')
    except LookupError:
        try:
            nltk.download('vader_lexicon', quiet=True)
        except Exception:
            pass
    NLTK_AVAILABLE = True
except Exception:
    NLTK_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except Exception:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Optional torch / EvoTorch imports for evolutionary calibration
try:
    import torch
except Exception:
    torch = None  # type: ignore

try:
    from evotorch import Problem
    from evotorch.algorithms import GeneticAlgorithm  # type: ignore
    from evotorch.operators import GaussianMutation, SimulatedBinaryCrossOver  # type: ignore
    EVOTORCH_AVAILABLE = True
except Exception:
    EVOTORCH_AVAILABLE = False

# Optional BoTorch / GPyTorch imports for Bayesian optimization
try:
    import botorch  # type: ignore
    from botorch.models import SingleTaskGP  # type: ignore
    from botorch.fit import fit_gpytorch_mll  # type: ignore
    from botorch.acquisition import LogExpectedImprovement  # type: ignore
    from botorch.optim import optimize_acqf  # type: ignore
    from botorch.utils.sampling import draw_sobol_samples  # type: ignore
    from gpytorch.mlls import ExactMarginalLogLikelihood  # type: ignore
    BOTORCH_AVAILABLE = True
except Exception:
    BOTORCH_AVAILABLE = False

# Global constants and seed
GLOBAL_SEED = 42
random.seed(GLOBAL_SEED)

# Files (consistent path format)
INTERACTIONS_FILE = os.path.join(DATA_DIR, "interactions.csv")
USER_PROFILES_FILE = os.path.join(DATA_DIR, "user_profiles.csv")
ITEM_METADATA_FILE = os.path.join(DATA_DIR, "item_metadata.csv")
CALIBRATED_PARAMS_FILE = os.path.join(DATA_DIR, "calibrated_parameters.json")
SIM_TRACES_FILE = os.path.join(DATA_DIR, "simulation_traces.jsonl")
EVAL_METRICS_FILE = os.path.join(DATA_DIR, "evaluation_metrics.json")
ABLATION_REPORT_FILE = os.path.join(DATA_DIR, "ablation_report.json")
LLM_CACHE_FILE = os.path.join(DATA_DIR, "llm_cache.json")

# Global LLM cache dictionary (loaded from file, updated during execution, saved at end)
llm_cache_dict: Dict[str, Any] = {}

# Preferred JSON data sources (if available)
AMAZON_TRAIN_JSON = os.path.join(DATA_DIR, "amazon_train_sample.json")
AMAZON_TEST_JSON = os.path.join(DATA_DIR, "amazon_test_sample.json")
USER_SAMPLE_JSON = os.path.join(DATA_DIR, "user_sample.json")
ITEM_SAMPLE_JSON = os.path.join(DATA_DIR, "item_sample.json")
REVIEW_SAMPLE_JSON = os.path.join(DATA_DIR, "review_sample.json")

# Default aspect vocabulary
DEFAULT_ASPECT_VOCAB = [
    "quality", "price", "delivery", "packaging", "usability", "durability",
    "customer service", "value", "features", "design"
]

# Aspect alias dictionary for lightweight normalization
ASPECT_ALIASES: Dict[str, List[str]] = {
    "quality": ["quality", "build", "craftsmanship", "materials", "finish"],
    "price": ["price", "pricing", "cost", "expensive", "cheap"],
    "delivery": ["delivery", "shipping", "ship", "arrival", "arrived", "courier"],
    "packaging": ["packaging", "package", "box", "wrapping", "packed"],
    "usability": ["usability", "ease", "easy", "user-friendly", "interface", "setup", "install"],
    "durability": ["durability", "durable", "sturdy", "rugged", "last", "lasting", "broke", "broken"],
    "customer service": ["customer service", "support", "service", "helpdesk", "seller", "cs"],
    "value": ["value", "bang", "worth", "deal", "bargain"],
    "features": ["features", "feature", "function", "functions", "capability", "options"],
    "design": ["design", "style", "look", "appearance", "aesthetic"],
}

# Profanity lexicon (small)
PROFANITY = {"damn", "hell", "shit", "crap", "sucks", "bastard"}


def is_offline_mode() -> bool:
    return os.environ.get("OFFLINE_MODE", "0") == "1"


def get_openai_api_key() -> str:
    """
    Retrieve the OpenAI API key from the environment variable OPENAI_API_KEY.

    Returns:
        str: The API key string.

    Raises:
        ValueError: If OPENAI_API_KEY is not set in the environment.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        return api_key
    raise ValueError("OpenAI API key not found in environment. Set OPENAI_API_KEY, or set OFFLINE_MODE=1 to run offline.")


def call_gpt5_with_responses_api(
    prompt: str,
    model: str = None,
    max_output_tokens: int = 4000,
    temperature: float = 0.6,
    retries: int = 2,
    backoff_base: float = 1.5
) -> str:
    """
    Call the OpenAI Responses API with a given prompt.

    Args:
        prompt (str): The prompt text for the LLM.
        model (str): Model name; default from env OPENAI_MODEL or "gpt-4.1-mini".
        max_output_tokens (int): Max tokens to generate.
        temperature (float): Sampling temperature.
        retries (int): Number of retries on transient errors.
        backoff_base (float): Exponential backoff base.

    Returns:
        str: The response text extracted from the Response object.

    Raises:
        RuntimeError: If OpenAI SDK is not available or API call fails.
        ValueError: If API key is missing.
    """
    if is_offline_mode():
        # Should not be called in offline mode
        raise RuntimeError("LLM call attempted in OFFLINE_MODE.")
    if OpenAI is None:
        raise RuntimeError("OpenAI SDK not available. Install the 'openai' package to enable LLM calls.")

    api_key = get_openai_api_key()
    client = OpenAI(api_key=api_key)

    if not model:
        model = os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")

    responses_kwargs = {
        "model": model,
        "input": [
            {"role": "user", "content": [{"type": "input_text", "text": prompt}]}
        ],
        "max_output_tokens": max_output_tokens,
        "temperature": temperature,
    }

    def extract_response(resp_obj: Any) -> str:
        if hasattr(resp_obj, "output_text") and isinstance(resp_obj.output_text, str):
            return resp_obj.output_text
        try:
            output = getattr(resp_obj, "output", None)
            if output and isinstance(output, list):
                content = output[0].get("content") if isinstance(output[0], dict) else None
                if content and isinstance(content, list) and len(content) > 0:
                    text = content[0].get("text")
                    if isinstance(text, str):
                        return text
        except Exception:
            pass
        return str(resp_obj)

    last_err = None
    for attempt in range(retries + 1):
        try:
            resp = client.responses.create(**responses_kwargs)
            return extract_response(resp)
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep((backoff_base ** attempt) + random.random() * 0.2)
            else:
                raise RuntimeError(f"OpenAI Responses API call failed after {retries+1} attempts: {e}") from e
    # Should not reach here
    raise RuntimeError(f"OpenAI Responses API call failed: {last_err}")


def ensure_data_dir() -> None:
    """
    Ensure the data directory exists; create if missing.
    """
    os.makedirs(DATA_DIR, exist_ok=True)


def _cache_key_to_string(cache_key: Tuple) -> str:
    """
    Convert a cache key tuple to a JSON-serializable string.
    
    Args:
        cache_key: Tuple containing cache key components
    
    Returns:
        JSON-serializable string representation of the cache key
    """
    # Convert tuple to list, handling nested tuples
    def _normalize(obj):
        if isinstance(obj, tuple):
            return list(obj)
        elif isinstance(obj, list):
            return [_normalize(item) for item in obj]
        else:
            return obj
    
    normalized = _normalize(cache_key)
    return json.dumps(normalized, sort_keys=True, ensure_ascii=False)


def load_llm_cache() -> Dict[str, Any]:
    """
    Load global LLM cache from file.
    
    Returns:
        Dictionary containing cached LLM responses (empty dict if file doesn't exist)
    """
    global llm_cache_dict
    cache_file = LLM_CACHE_FILE
    if os.path.isfile(cache_file):
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                llm_cache_dict = json.load(f)
                if not isinstance(llm_cache_dict, dict):
                    llm_cache_dict = {}
                print(f"Loaded LLM cache from {cache_file}: {len(llm_cache_dict)} entries")
        except Exception as e:
            print(f"Warning: Failed to load LLM cache from {cache_file}: {e}")
            llm_cache_dict = {}
    else:
        llm_cache_dict = {}
        print(f"LLM cache file not found at {cache_file}, starting with empty cache")
    return llm_cache_dict


def save_llm_cache() -> None:
    """Save global LLM cache to file."""
    global llm_cache_dict
    ensure_data_dir()
    cache_file = LLM_CACHE_FILE
    try:
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(llm_cache_dict, f, indent=2, ensure_ascii=False)
        print(f"Saved LLM cache to {cache_file}: {len(llm_cache_dict)} entries")
    except Exception as e:
        print(f"Warning: Failed to save LLM cache to {cache_file}: {e}")


def parse_cli() -> argparse.Namespace:
    """
    Parse command-line arguments for simulation configuration.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Multi-agent review simulation with calibration and evaluation.")
    parser.add_argument("--seed", type=int, default=GLOBAL_SEED, help="Global random seed.")
    parser.add_argument("--num-trials", type=int, default=20, help="Number of calibration trials.")
    parser.add_argument("--early-stop-patience", type=int, default=5, help="Early stopping patience for calibration.")
    parser.add_argument("--max-records", type=int, default=200, help="Max number of records to use (subset for speed).")
    parser.add_argument("--max-validation-records", type=int, default=30, help="Max records in validation rollout (LLM calls).")
    parser.add_argument("--use-llm", type=int, default=1, help="If 1, use OpenAI LLM for generation and rating; if 0, require OFFLINE_MODE=1.")
    parser.add_argument("--model-name", type=str, default=os.environ.get("OPENAI_MODEL", "gpt-4.1-mini"), help="OpenAI model name (Responses API).")
    parser.add_argument("--max-output-tokens", type=int, default=600, help="Max tokens for LLM outputs.")
    parser.add_argument("--llm-retries", type=int, default=2, help="Retries for LLM calls.")
    parser.add_argument("--ablation", type=int, default=1, help="If 1, run a small ablation study.")
    parser.add_argument("--offline", type=int, default=int(os.environ.get("OFFLINE_MODE", "0")), help="If 1, run offline stubs only.")
    parser.add_argument("--test", action="store_true", help="Test mode: skip calibration, load params from checkpoint and run on test set.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint folder name (relative to DATA_PATH). Used with --test to load calibrated parameters.")
    args = parser.parse_args()
    return args


def set_global_seed(seed: int) -> None:
    """
    Set global random seeds for deterministic behavior.

    Args:
        seed (int): Seed value.
    """
    random.seed(seed)


def read_csv_if_exists(path: str) -> Optional[List[Dict[str, str]]]:
    """
    Read a CSV file into a list of dictionaries if it exists.

    Args:
        path (str): Path to the CSV file.

    Returns:
        Optional[List[Dict[str, str]]]: List of rows as dicts, or None if file not found.
    """
    if not os.path.isfile(path):
        return None
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader]
    return rows


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    """
    Write a list of dictionaries to CSV.

    Args:
        path (str): Output file path.
        rows (List[Dict[str, Any]]): Rows to write.

    Raises:
        ValueError: If rows is empty or inconsistent.
    """
    if not rows:
        raise ValueError("Cannot write empty CSV: no rows provided.")
    fieldnames = sorted(set().union(*(row.keys() for row in rows)))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})


def read_json_or_jsonl(path: str) -> Optional[List[Dict[str, Any]]]:
    if not os.path.isfile(path):
        return None
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    if not content:
        return None
    content = content.lstrip("\ufeff").lstrip()  # strip BOM and leading whitespace
    # Try parse as JSON array or object first
    try:
        data = json.loads(content)
        if isinstance(data, list):
            rows = [r for r in data if isinstance(r, dict)]
            return rows or None
        elif isinstance(data, dict):
            # If dict values are all dicts (records), extract values; otherwise treat as single record
            values = list(data.values())
            if values and all(isinstance(v, dict) for v in values):
                return values  # Extract records from dict values
            else:
                return [data]  # Treat entire dict as single record
    except Exception:
        pass
    # Fallback: JSONL
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                rows.append(obj)
        except Exception:
            continue
    return rows or None


def synthesize_dataset(n_users: int = 20, n_items: int = 30, n_interactions: int = 200, seed: int = 42) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Create a synthetic dataset of interactions, user profiles, and item metadata.

    Args:
        n_users (int): Number of unique users.
        n_items (int): Number of unique items.
        n_interactions (int): Number of interactions.
        seed (int): RNG seed.

    Returns:
        Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
            interactions, user_profiles, item_metadata
    """
    rng = random.Random(seed)
    users = [f"U{u:03d}" for u in range(1, n_users + 1)]
    items = [f"I{i:03d}" for i in range(1, n_items + 1)]
    item_cats = ["electronics", "home", "toys", "books", "kitchen", "garden"]
    interactions: List[Dict[str, Any]] = []
    user_profiles: List[Dict[str, Any]] = []
    item_metadata: List[Dict[str, Any]] = []

    user_leniency = {u: rng.uniform(2.5, 3.5) for u in users}
    item_quality = {it: rng.uniform(2.5, 4.2) for it in items}

    # user profiles
    for u in users:
        user_profiles.append({
            "user_id": u,
            "avg_stars": round(user_leniency[u], 2),
            "friends": "",
            "review_count": rng.randint(1, 50),
            "tone_hint": rng.choice(["formal", "casual", "enthusiastic", "concise"])
        })

    # item metadata
    for it in items:
        cat = rng.choice(item_cats)
        tags = rng.sample(DEFAULT_ASPECT_VOCAB, k=rng.randint(2, 5))
        item_metadata.append({
            "item_id": it,
            "category": cat,
            "keywords": ";".join(tags)
        })

    # interactions
    base_date = dt.date(2023, 1, 1)
    for idx in range(n_interactions):
        u = rng.choice(users)
        it = rng.choice(items)
        date = base_date + dt.timedelta(days=idx % 90)
        sentiment = (item_quality[it] - 3.0) * 0.6 + (user_leniency[u] - 3.0) * 0.4 + rng.gauss(0, 0.2)
        stars = min(5, max(1, int(round(3 + 1.2 * sentiment))))
        aspects = rng.sample(DEFAULT_ASPECT_VOCAB, k=rng.randint(2, 4))
        tone = rng.choice(["positive", "neutral", "negative"]) if stars != 3 else "neutral"
        review = f"This {rng.choice(['product', 'item'])} has {rng.choice(['good', 'decent', 'average', 'poor'])} {aspects[0]} and {rng.choice(['solid', 'okay', 'weak'])} {aspects[1]}. Overall {tone} experience."
        interactions.append({
            "user_id": u,
            "item_id": it,
            "stars": stars,
            "review": review,
            "timestamp": date.isoformat(),
            "datatype": ""  # fill later
        })

    interactions_sorted = sorted(interactions, key=lambda r: r["timestamp"])
    split_idx = int(0.8 * len(interactions_sorted))
    for i, rec in enumerate(interactions_sorted):
        rec["datatype"] = "train" if i < split_idx else "test"

    return interactions_sorted, user_profiles, item_metadata


def coerce_interaction_row(row: Dict[str, Any]) -> Dict[str, Any]:
    r = dict(row)
    r["user_id"] = str(r.get("user_id", r.get("user", "")) or "").strip()
    r["item_id"] = str(r.get("item_id", r.get("item", "")) or "").strip()
    try:
        r["stars"] = int(float(r.get("stars", r.get("rating", 3))))
    except Exception:
        r["stars"] = 3
    r["review"] = str(r.get("review", r.get("text", "")) or "").strip()
    r["timestamp"] = str(r.get("timestamp", r.get("time", dt.date(2023, 1, 1).isoformat())))
    r["datatype"] = r.get("datatype", "")
    return r


def load_data(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Load data from JSON preferred files or CSV files or synthesize if missing. Returns a data bundle.

    Args:
        args (argparse.Namespace): CLI arguments.

    Returns:
        Dict[str, Any]: Data bundle including interactions, user_profiles, item_metadata, history.
    """
    ensure_data_dir()

    # Preferred: JSON sources if they exist
    train_json = read_json_or_jsonl(AMAZON_TRAIN_JSON)
    test_json = read_json_or_jsonl(AMAZON_TEST_JSON)
    user_json = read_json_or_jsonl(USER_SAMPLE_JSON)
    item_json = read_json_or_jsonl(ITEM_SAMPLE_JSON)
    review_json = read_json_or_jsonl(REVIEW_SAMPLE_JSON)

    interactions: Optional[List[Dict[str, Any]]] = None
    user_profiles: Optional[List[Dict[str, Any]]] = None
    item_metadata: Optional[List[Dict[str, Any]]] = None

    if train_json or test_json:
        interactions = []
        if train_json:
            for row in train_json:
                rec = coerce_interaction_row(row)
                rec["datatype"] = "train"
                interactions.append(rec)
        if test_json:
            for row in test_json:
                rec = coerce_interaction_row(row)
                rec["datatype"] = "test"
                interactions.append(rec)

        user_profiles = user_json or []
        item_metadata = item_json or []
        # Optionally, could merge review_json as additional historical reviews if provided.
    else:
        # CSV or synth
        interactions = read_csv_if_exists(INTERACTIONS_FILE)
        user_profiles = read_csv_if_exists(USER_PROFILES_FILE)
        item_metadata = read_csv_if_exists(ITEM_METADATA_FILE)

        if interactions is None:
            synth_inter, synth_users, synth_items = synthesize_dataset(
                n_users=30, n_items=50, n_interactions=max(100, args.max_records), seed=args.seed
            )
            write_csv(INTERACTIONS_FILE, synth_inter)
            write_csv(USER_PROFILES_FILE, synth_users)
            write_csv(ITEM_METADATA_FILE, synth_items)
            interactions = read_csv_if_exists(INTERACTIONS_FILE)
            user_profiles = read_csv_if_exists(USER_PROFILES_FILE)
            item_metadata = read_csv_if_exists(ITEM_METADATA_FILE)

    if interactions is None:
        raise ValueError("Failed to load interactions from JSON or CSV sources.")

    required_fields = {"user_id", "item_id", "stars", "review"}
    for row in interactions:
        if not required_fields.issubset(row.keys()):
            raise ValueError(f"Missing required fields in interactions. Found keys: {list(row.keys())}")
        try:
            row["stars"] = int(float(row["stars"]))
        except Exception:
            row["stars"] = int(row["stars"]) if isinstance(row["stars"], int) else 3
        row["review"] = str(row.get("review", "")).strip()
        if "timestamp" not in row or not row["timestamp"]:
            row["timestamp"] = dt.date(2023, 1, 1).isoformat()
        if "datatype" not in row or row["datatype"] not in {"train", "test"}:
            row["datatype"] = ""

    if args.max_records and len(interactions) > args.max_records:
        interactions = interactions[: args.max_records]

    if all(not r["datatype"] for r in interactions):
        try:
            sorted_rows = sorted(interactions, key=lambda r: r["timestamp"])
        except Exception:
            sorted_rows = interactions[:]
        split_idx = int(0.8 * len(sorted_rows))
        for i, rec in enumerate(sorted_rows):
            rec["datatype"] = "train" if i < split_idx else "test"
        interactions = sorted_rows

    user_profiles = user_profiles or []
    item_metadata = item_metadata or []

    # Attach historical reviews if available
    history_reviews = review_json or []

    data_bundle = {
        "interactions": interactions,
        "user_profiles": user_profiles,
        "item_metadata": item_metadata,
        "history_reviews": history_reviews
    }
    return data_bundle


def tokenize(text: str) -> List[str]:
    text = text.lower()
    tokens = []
    curr = []
    for ch in text:
        if ch.isalnum():
            curr.append(ch)
        else:
            if curr:
                tokens.append("".join(curr))
                curr = []
    if curr:
        tokens.append("".join(curr))
    return tokens


def normalize_token(tok: str) -> str:
    t = tok.lower()
    # simple suffix stripping for plural/tense
    for suf in ("ing", "ed", "ly", "s"):
        if t.endswith(suf) and len(t) > len(suf) + 2:
            t = t[: -len(suf)]
            break
    return t


def normalized_tokens(text: str) -> List[str]:
    return [normalize_token(t) for t in tokenize(text)]


def build_aspect_index() -> Dict[str, set]:
    idx: Dict[str, set] = {}
    for a, aliases in ASPECT_ALIASES.items():
        aset = set()
        for al in aliases + [a]:
            for tok in al.split():
                aset.add(normalize_token(tok))
        idx[a] = aset
    return idx


ASPECT_INDEX = build_aspect_index()


def detect_aspects_in_tokens(toks: List[str]) -> set:
    tset = set(toks)
    hits = set()
    for a, alias_tokens in ASPECT_INDEX.items():
        if alias_tokens & tset:
            hits.add(a)
    return hits


def sentence_split(text: str) -> List[str]:
    parts = []
    buff = []
    for ch in text:
        buff.append(ch)
        if ch in [".", "!", "?"]:
            parts.append("".join(buff).strip())
            buff = []
    if buff:
        parts.append("".join(buff).strip())
    return [s for s in parts if s]


POS_WORDS = {
    "good", "great", "excellent", "amazing", "love", "loved", "like", "liked",
    "awesome", "fantastic", "superb", "satisfied", "happy", "positive", "recommend",
    "durable", "reliable", "value", "fast", "comfortable", "nice", "perfect"
}
NEG_WORDS = {
    "bad", "terrible", "awful", "hate", "hated", "dislike", "disliked", "poor",
    "disappointed", "broken", "slow", "uncomfortable", "worse", "worst", "problem",
    "issue", "buggy", "fragile", "cheap", "expensive", "negative"
}


def sentiment_score(text: str) -> float:
    toks = tokenize(text)
    if not toks:
        return 0.0
    pos = sum(1 for t in toks if t in POS_WORDS)
    neg = sum(1 for t in toks if t in NEG_WORDS)
    # simple phrase heuristic
    if "not" in toks and "recommend" in toks:
        neg += 1
    score = (pos - neg) / max(1, pos + neg)
    if pos + neg == 0:
        if any(t in {"good", "nice", "decent"} for t in toks):
            score = 0.2
        elif any(t in {"poor", "bad", "awful"} for t in toks):
            score = -0.2
        else:
            score = 0.0
    return max(-1.0, min(1.0, score))


def cosine_similarity(a: Dict[str, float], b: Dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(a.get(k, 0.0) * v for k, v in b.items())
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0.0 or nb == 0.0:
        return 0.0
    sim = dot / (na * nb)
    return max(0.0, min(1.0, sim))


# Global models for evaluation metrics (lazy initialization)
_emotion_classifier = None
_sentiment_analyzer = None
_sentence_transformer = None


def _get_emotion_classifier():
    """Lazy initialization of emotion classifier."""
    global _emotion_classifier
    if _emotion_classifier is None and TRANSFORMERS_AVAILABLE:
        try:
            _emotion_classifier = transformers_pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                return_all_scores=True,
                device=-1  # CPU
            )
        except Exception as e:
            print(f"Warning: Failed to load emotion classifier: {e}")
    return _emotion_classifier


def _get_sentiment_analyzer():
    """Lazy initialization of NLTK sentiment analyzer."""
    global _sentiment_analyzer
    if _sentiment_analyzer is None and NLTK_AVAILABLE:
        try:
            _sentiment_analyzer = SentimentIntensityAnalyzer()
        except Exception as e:
            print(f"Warning: Failed to load NLTK sentiment analyzer: {e}")
    return _sentiment_analyzer


def _get_sentence_transformer():
    """Lazy initialization of Sentence-BERT model."""
    global _sentence_transformer
    if _sentence_transformer is None and SENTENCE_TRANSFORMERS_AVAILABLE:
        try:
            _sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            print(f"Warning: Failed to load Sentence-BERT: {e}")
    return _sentence_transformer


def get_emotion_vector(text: str) -> List[float]:
    """
    Get emotion scores for top 5 emotions using emotion classifier.
    Returns normalized vector (values in [0, 1]).
    The emotion classifier outputs probabilities, so we take top 5 and normalize to [0, 1].
    """
    classifier = _get_emotion_classifier()
    if classifier is None or not text:
        return [0.0] * 5
    
    try:
        results = classifier(text)[0]  # Get first (and only) result
        # Sort by score descending and take top 5
        sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)[:5]
        # Extract scores (these are already probabilities from the classifier)
        scores = [r['score'] for r in sorted_results]
        # Pad to exactly 5 elements if needed
        while len(scores) < 5:
            scores.append(0.0)
        scores = scores[:5]
        # Scores from the classifier are already probabilities (sum to 1), 
        # but we normalize to [0, 1] range where max is 1.0 for consistency
        max_score = max(scores) if scores else 1.0
        if max_score > 0:
            scores = [s / max_score for s in scores]
        return scores
    except Exception as e:
        # Return default neutral vector on error
        return [0.2] * 5  # Uniform distribution as fallback


def get_sentiment_score_normalized(text: str) -> float:
    """
    Get sentiment score using NLTK SentimentIntensityAnalyzer.
    Returns normalized value in [0, 1] where 0 = very negative, 1 = very positive.
    """
    analyzer = _get_sentiment_analyzer()
    if analyzer is None or not text:
        return 0.5  # Neutral
    
    try:
        scores = analyzer.polarity_scores(text)
        # compound score ranges from -1 to 1, normalize to [0, 1]
        compound = scores.get('compound', 0.0)
        normalized = (compound + 1.0) / 2.0  # Map [-1, 1] to [0, 1]
        return max(0.0, min(1.0, normalized))
    except Exception:
        return 0.5


def get_topic_embedding(text: str) -> Optional[List[float]]:
    """
    Get text embedding using Sentence-BERT.
    Returns embedding vector or None if unavailable.
    """
    model = _get_sentence_transformer()
    if model is None or not text:
        return None
    
    try:
        embedding = model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
    except Exception:
        return None


def cosine_similarity_vectors(a: Optional[List[float]], b: Optional[List[float]]) -> float:
    """
    Compute cosine similarity between two embedding vectors.
    Returns similarity in [0, 1].
    """
    if a is None or b is None or len(a) == 0 or len(b) == 0:
        return 0.0
    if len(a) != len(b):
        return 0.0
    
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    sim = dot / (na * nb)
    return max(0.0, min(1.0, sim))


def stable_hash_bucket(s: str, dim: int) -> int:
    h = hashlib.blake2b(s.encode("utf-8"), digest_size=8).hexdigest()
    return int(h, 16) % dim


def hashed_ngram_vector(text: str, n: int = 3, dim: int = 2048) -> Dict[str, float]:
    s = text.lower()
    grams = [s[i: i + n] for i in range(0, max(0, len(s) - n + 1))]
    vec: Dict[str, float] = {}
    for g in grams:
        key = str(stable_hash_bucket(g, dim))
        vec[key] = vec.get(key, 0.0) + 1.0
    norm = math.sqrt(sum(v * v for v in vec.values()))
    if norm > 0:
        for k in list(vec.keys()):
            vec[k] /= norm
    return vec


@dataclass
class DataIndexer:
    """
    DataIndexer parses interaction records, builds indices, and precomputes priors.
    """
    interactions: List[Dict[str, Any]]
    user_profiles: List[Dict[str, Any]]
    item_metadata: List[Dict[str, Any]]
    history_reviews: List[Dict[str, Any]] = field(default_factory=list)
    aspect_vocab: List[str] = field(default_factory=lambda: DEFAULT_ASPECT_VOCAB.copy())
    platform_policy_weight: float = 0.5

    user_index: Dict[str, List[int]] = field(init=False, default_factory=dict)
    item_index: Dict[str, List[int]] = field(init=False, default_factory=dict)
    split_index: Dict[str, List[int]] = field(init=False, default_factory=dict)
    user_priors: Dict[str, Dict[str, float]] = field(init=False, default_factory=dict)
    item_priors: Dict[str, Dict[str, float]] = field(init=False, default_factory=dict)
    item_tags: Dict[str, List[str]] = field(init=False, default_factory=dict)
    item_metadata_map: Dict[str, Dict[str, Any]] = field(init=False, default_factory=dict)  # Store item name, category, etc.
    profile_map: Dict[str, Dict[str, Any]] = field(init=False, default_factory=dict)
    global_mean: float = field(init=False, default=3.0)
    user_history_texts: Dict[str, List[str]] = field(init=False, default_factory=dict)
    item_history_texts: Dict[str, List[str]] = field(init=False, default_factory=dict)
    user_history_reviews_with_stars: Dict[str, List[Tuple[str, int]]] = field(init=False, default_factory=dict)  # For few-shot in rating
    item_history_reviews_with_stars: Dict[str, List[Tuple[str, int]]] = field(init=False, default_factory=dict)  # For few-shot in rating

    def __post_init__(self) -> None:
        self._build_indices()
        self._build_profiles()
        self._build_item_tags()
        self._build_item_metadata_map()
        self._compute_priors()
        self._build_histories()

    def _build_indices(self) -> None:
        self.user_index = defaultdict(list)
        self.item_index = defaultdict(list)
        self.split_index = defaultdict(list)
        for idx, rec in enumerate(self.interactions):
            self.user_index[rec["user_id"]].append(idx)
            self.item_index[rec["item_id"]].append(idx)
            split = rec.get("datatype", "train")
            self.split_index[split].append(idx)

    def _build_profiles(self) -> None:
        self.profile_map = {}
        for p in self.user_profiles:
            uid = str(p.get("user_id", "")).strip()
            if uid:
                self.profile_map[uid] = p

    def _build_item_tags(self) -> None:
        self.item_tags = {}
        for it in self.item_metadata:
            item_id = str(it.get("item_id", "")).strip()
            if not item_id:
                continue
            tags = str(it.get("keywords", it.get("tags", ""))).split(";")
            self.item_tags[item_id] = [t.strip().lower() for t in tags if t.strip()]
        for item_id in self.item_index.keys():
            if item_id not in self.item_tags:
                self.item_tags[item_id] = random.sample(self.aspect_vocab, k=min(3, len(self.aspect_vocab)))

    def _build_item_metadata_map(self) -> None:
        """Build a map from item_id to item metadata (name, category, type, etc.)"""
        self.item_metadata_map = {}
        for it in self.item_metadata:
            item_id = str(it.get("item_id", "")).strip()
            if not item_id:
                continue
            self.item_metadata_map[item_id] = {
                "name": it.get("name", ""),
                "category": it.get("category", ""),
                "categories": it.get("categories", ""),
                "type": it.get("type", ""),
                "description": it.get("description", ""),
            }
        # For items without metadata, try to extract product name from first review
        for item_id in self.item_index.keys():
            if item_id not in self.item_metadata_map:
                # Try to extract product name from first review about this item
                item_idxs = self.item_index.get(item_id, [])
                if item_idxs:
                    first_review = str(self.interactions[item_idxs[0]].get("review", "")).strip()
                    # Heuristic: extract first noun phrase (often product name)
                    # For Amazon reviews, product name often appears at the start
                    product_name = ""
                    if first_review:
                        # Look for patterns like "The [Product Name] is..." or "[Product Name] is..."
                        sentences = first_review.split(".")[:1]
                        if sentences:
                            first_sent = sentences[0].strip()
                            # Try to extract product name (first 5-10 words before "is" or comma)
                            words = first_sent.split()
                            if len(words) > 2:
                                # Common pattern: "The [Product Name] is..."
                                if words[0].lower() == "the" and len(words) > 3:
                                    product_name = " ".join(words[1:min(6, len(words))])
                                else:
                                    product_name = " ".join(words[:min(5, len(words))])
                    self.item_metadata_map[item_id] = {
                        "name": product_name,
                        "category": "",
                        "categories": "",
                        "type": "",
                        "description": "",
                    }

    def _compute_priors(self) -> None:
        # global mean based on train if available
        train_idxs = self.split_index.get("train", [])
        all_idxs = train_idxs if train_idxs else list(range(len(self.interactions)))
        stars_all = [int(self.interactions[i]["stars"]) for i in all_idxs] or [3]
        self.global_mean = float(statistics.mean(stars_all))

        train_set = set(train_idxs)

        self.user_priors = {}
        for user_id, idxs in self.user_index.items():
            idxs_train = [i for i in idxs if i in train_set]
            stars = [int(self.interactions[i]["stars"]) for i in idxs_train]
            if stars:
                mean = statistics.mean(stars)
                var = statistics.pvariance(stars) if len(stars) > 1 else 1.0
            else:
                mean = self.global_mean
                var = 1.0
            self.user_priors[user_id] = {"mean": float(mean), "var": float(var)}

        self.item_priors = {}
        for item_id, idxs in self.item_index.items():
            idxs_train = [i for i in idxs if i in train_set]
            stars = [int(self.interactions[i]["stars"]) for i in idxs_train]
            if stars:
                mean = statistics.mean(stars)
                var = statistics.pvariance(stars) if len(stars) > 1 else 1.0
            else:
                mean = self.global_mean
                var = 1.0
            self.item_priors[item_id] = {"mean": float(mean), "var": float(var)}

    def _build_histories(self) -> None:
        self.user_history_texts = defaultdict(list)
        self.item_history_texts = defaultdict(list)
        self.user_history_reviews_with_stars = defaultdict(list)  # List of (review_text, stars) tuples
        self.item_history_reviews_with_stars = defaultdict(list)  # List of (review_text, stars) tuples
        # Use training set texts for few-shot hints
        train_idxs = self.split_index.get("train", [])
        for i in train_idxs:
            rec = self.interactions[i]
            uid, iid = rec["user_id"], rec["item_id"]
            txt = str(rec.get("review", "")).strip()
            stars = int(rec.get("stars", 3))
            if txt:
                if len(self.user_history_texts[uid]) < 3:
                    self.user_history_texts[uid].append(txt[:240])
                if len(self.item_history_texts[iid]) < 3:
                    self.item_history_texts[iid].append(txt[:240])
                # Store review+star pairs for few-shot examples in rating
                if len(self.user_history_reviews_with_stars[uid]) < 3:
                    self.user_history_reviews_with_stars[uid].append((txt[:240], stars))
                if len(self.item_history_reviews_with_stars[iid]) < 3:
                    self.item_history_reviews_with_stars[iid].append((txt[:240], stars))
        # supplement with history_reviews if available
        for obj in self.history_reviews:
            uid = str(obj.get("user_id", "")).strip()
            iid = str(obj.get("item_id", "")).strip()
            txt = str(obj.get("review", obj.get("text", ""))).strip()
            stars = int(obj.get("stars", obj.get("rating", 3)))
            if uid and txt and len(self.user_history_texts[uid]) < 3:
                self.user_history_texts[uid].append(txt[:240])
            if iid and txt and len(self.item_history_texts[iid]) < 3:
                self.item_history_texts[iid].append(txt[:240])
            # Store review+star pairs
            if uid and txt and len(self.user_history_reviews_with_stars[uid]) < 3:
                self.user_history_reviews_with_stars[uid].append((txt[:240], stars))
            if iid and txt and len(self.item_history_reviews_with_stars[iid]) < 3:
                self.item_history_reviews_with_stars[iid].append((txt[:240], stars))

    def get_user_context(self, user_id: str) -> Dict[str, Any]:
        prior = self.user_priors.get(user_id, {"mean": self.global_mean, "var": 1.0})
        profile = self.profile_map.get(user_id, {})
        return {
            "user_id": user_id,
            "leniency_prior": float(prior["mean"]),
            "profile": profile,
            "history_texts": self.user_history_texts.get(user_id, []),
            "history_reviews_with_stars": self.user_history_reviews_with_stars.get(user_id, [])  # For few-shot in rating
        }

    def get_item_context(self, item_id: str) -> Dict[str, Any]:
        prior = self.item_priors.get(item_id, {"mean": self.global_mean, "var": 1.0})
        tags = self.item_tags.get(item_id, [])
        metadata = self.item_metadata_map.get(item_id, {})
        
        # Helper to normalize category value (handle both string and list)
        def normalize_category(cat_val):
            if not cat_val:
                return ""
            if isinstance(cat_val, list):
                return ", ".join(str(c).strip() for c in cat_val if c)
            return str(cat_val).strip()
        
        # Get category (prefer category, fallback to categories)
        category_raw = metadata.get("category") or metadata.get("categories") or ""
        item_category = normalize_category(category_raw)
        
        # Normalize other fields to ensure they're strings
        item_name = str(metadata.get("name", "")).strip()
        item_type = str(metadata.get("type", "")).strip()
        item_description = str(metadata.get("description", "")).strip()
        
        return {
            "item_id": item_id,
            "reputation_prior": float(prior["mean"]),
            "tags": tags,
            "history_texts": self.item_history_texts.get(item_id, []),
            "history_reviews_with_stars": self.item_history_reviews_with_stars.get(item_id, []),  # For few-shot in rating
            "item_name": item_name,
            "item_category": item_category,
            "item_type": item_type,
            "item_description": item_description,
        }

    def refresh(self) -> None:
        self._compute_priors()


@dataclass
class PersonaProfiler:
    neighbor_weight: float = 0.0
    leniency_drift_rate: float = 0.05
    verbosity_scale: float = 1.0
    aspect_weight_decay: float = 0.9

    user_state: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    neighbors: Dict[str, List[str]] = field(default_factory=dict)

    def construct(self, data: DataIndexer) -> None:
        self.user_state = {}
        self.neighbors = {uid: [] for uid in data.user_index.keys()}
        for uid, prof in data.profile_map.items():
            friends_raw = str(prof.get("friends", "")).strip()
            if friends_raw:
                friends = [f.strip() for f in friends_raw.split(";") if f.strip()]
                self.neighbors[uid] = [f for f in friends if f in data.user_index]
        train_set = set(data.split_index.get("train", []))
        for uid in data.user_index:
            prior = data.user_priors.get(uid, {"mean": data.global_mean})
            tone_hint = data.profile_map.get(uid, {}).get("tone_hint", "neutral")
            history_idxs_all = data.user_index.get(uid, [])
            history_idxs = [i for i in history_idxs_all if i in train_set]
            recent_reviews = [data.interactions[i]["review"] for i in history_idxs[-5:]]
            style_vector = self._infer_style_vector(recent_reviews, tone_hint)
            aspect_weights = self._infer_aspect_weights(recent_reviews)
            self.user_state[uid] = {
                "baseline_leniency": float(prior["mean"]),
                "verbosity_prior": float(min(1.5, 0.5 + 0.02 * len(history_idxs))),
                "tone_style_prior": tone_hint,
                "domain_familiarity": float(min(1.0, 0.2 + 0.1 * math.log(1 + len(history_idxs)))),
                "leniency_drift": 0.0,
                "aspect_preference_weights": aspect_weights,
                "style_vector": style_vector,
                "recent_sentiment_bias": 0.0,
            }

    def _infer_style_vector(self, reviews: List[str], tone_hint: str) -> List[float]:
        formality = 0.6 if tone_hint == "formal" else 0.4
        enthusiasm = 0.7 if tone_hint in ("enthusiastic",) else 0.4
        conciseness = 0.6 if tone_hint in ("concise",) else 0.4
        avg_len = statistics.mean(len(r) for r in reviews) if reviews else 80.0
        conciseness = max(0.1, min(1.0, 200.0 / (50.0 + avg_len)))
        return [formality, enthusiasm, conciseness]

    def _infer_aspect_weights(self, reviews: List[str]) -> Dict[str, float]:
        counts = Counter()
        for r in reviews:
            toks = normalized_tokens(r)
            hits = detect_aspects_in_tokens(toks)
            for a in hits:
                counts[a] += 1
        total = sum(counts.values()) or 1
        weights = {a: (counts[a] / total) for a in DEFAULT_ASPECT_VOCAB}
        if sum(weights.values()) == 0:
            weights = {a: 1.0 / len(DEFAULT_ASPECT_VOCAB) for a in DEFAULT_ASPECT_VOCAB}
        return weights

    def update_from_recent(self, user_id: str, new_review: str, new_stars: int) -> None:
        state = self.user_state.get(user_id)
        if not state:
            return
        err = new_stars - state["baseline_leniency"]
        state["leniency_drift"] = (1 - self.leniency_drift_rate) * state["leniency_drift"] + self.leniency_drift_rate * err

        toks = normalized_tokens(new_review)
        hits = detect_aspects_in_tokens(toks)
        for a in DEFAULT_ASPECT_VOCAB:
            hit = a in hits
            prev = state["aspect_preference_weights"].get(a, 1.0 / len(DEFAULT_ASPECT_VOCAB))
            target = 1.0 if hit else 0.0
            state["aspect_preference_weights"][a] = self.aspect_weight_decay * prev + (1 - self.aspect_weight_decay) * target

        s = sentiment_score(new_review)
        state["recent_sentiment_bias"] = 0.8 * state["recent_sentiment_bias"] + 0.2 * s

    def get_persona(self, user_id: str, data: DataIndexer) -> Dict[str, Any]:
        base = self.user_state.get(user_id)
        if not base:
            prior = data.user_priors.get(user_id, {"mean": data.global_mean})
            base = {
                "baseline_leniency": float(prior["mean"]),
                "verbosity_prior": 1.0,
                "tone_style_prior": "neutral",
                "domain_familiarity": 0.5,
                "leniency_drift": 0.0,
                "aspect_preference_weights": {a: 1.0 / len(DEFAULT_ASPECT_VOCAB) for a in DEFAULT_ASPECT_VOCAB},
                "style_vector": [0.5, 0.5, 0.5],
                "recent_sentiment_bias": 0.0,
            }
            self.user_state[user_id] = base

        neighs = self.neighbors.get(user_id, [])
        if neighs and self.neighbor_weight > 0.0:
            leniencies = [self.user_state.get(n, {}).get("baseline_leniency", data.user_priors.get(n, {}).get("mean", data.global_mean)) for n in neighs]
            if leniencies:
                avg_leniency = statistics.mean(leniencies)
                base["baseline_leniency"] = (1 - self.neighbor_weight) * base["baseline_leniency"] + self.neighbor_weight * avg_leniency

            agg = Counter()
            for n in neighs:
                pref = self.user_state.get(n, {}).get("aspect_preference_weights", {})
                for a, w in pref.items():
                    agg[a] += w
            total = sum(agg.values()) or 1
            neigh_aspects = {a: (agg[a] / total) for a in DEFAULT_ASPECT_VOCAB}
            for a in DEFAULT_ASPECT_VOCAB:
                base["aspect_preference_weights"][a] = (1 - self.neighbor_weight) * base["aspect_preference_weights"].get(a, 1.0 / len(DEFAULT_ASPECT_VOCAB)) + self.neighbor_weight * neigh_aspects.get(a, 1.0 / len(DEFAULT_ASPECT_VOCAB))

        return base


@dataclass
class ItemProfiler:
    reputation_inertia: float = 0.7
    aspect_smoothing_alpha: float = 0.6
    min_reviews_for_confidence: int = 5

    item_state: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def construct(self, data: DataIndexer) -> None:
        self.item_state = {}
        train_set = set(data.split_index.get("train", []))
        for item_id, idxs in data.item_index.items():
            idxs_train = [i for i in idxs if i in train_set]
            reviews = [data.interactions[i]["review"] for i in idxs_train]
            stars = [int(data.interactions[i]["stars"]) for i in idxs_train]
            aspects = self._extract_aspects(reviews)
            item_mean = statistics.mean(stars) if stars else data.global_mean
            prior_mean = self.reputation_inertia * item_mean + (1 - self.reputation_inertia) * data.global_mean
            var = statistics.pvariance(stars) if len(stars) > 1 else 1.0
            self.item_state[item_id] = {
                "quality_prior": prior_mean,
                "variance": var,
                "aspect_summary": aspects,
                "domain_tags": data.item_tags.get(item_id, []),
                "freshness_score": min(1.0, len(stars) / 10.0),
                "controversy": min(1.0, var / 2.0),
                "aspect_confidence": min(1.0, len(reviews) / 10.0),
            }

    def _extract_aspects(self, reviews: List[str]) -> Dict[str, float]:
        counts = Counter()
        for r in reviews:
            toks = normalized_tokens(r)
            hits = detect_aspects_in_tokens(toks)
            for a in hits:
                counts[a] += 1
        total = sum(counts.values()) or 1
        aspects = {a: (counts[a] / total) for a in DEFAULT_ASPECT_VOCAB}
        if sum(aspects.values()) == 0:
            aspects = {a: 1.0 / len(DEFAULT_ASPECT_VOCAB) for a in DEFAULT_ASPECT_VOCAB}
        return aspects

    def update_from_observation(self, item_id: str, new_review: str, new_stars: int) -> None:
        st = self.item_state.get(item_id)
        if not st:
            st = {
                "quality_prior": 3.0,
                "variance": 1.0,
                "aspect_summary": {a: 1.0 / len(DEFAULT_ASPECT_VOCAB) for a in DEFAULT_ASPECT_VOCAB},
                "domain_tags": [],
                "freshness_score": 0.5,
                "controversy": 0.5,
                "aspect_confidence": 0.1,
            }
            self.item_state[item_id] = st
        st["quality_prior"] = self.reputation_inertia * st["quality_prior"] + (1 - self.reputation_inertia) * new_stars
        new_aspects = self._extract_aspects([new_review])
        for a in DEFAULT_ASPECT_VOCAB:
            st["aspect_summary"][a] = self.aspect_smoothing_alpha * st["aspect_summary"].get(a, 0.0) + (1 - self.aspect_smoothing_alpha) * new_aspects.get(a, 0.0)
        st["freshness_score"] = min(1.0, 0.9 * st["freshness_score"] + 0.1)
        st["aspect_confidence"] = min(1.0, st["aspect_confidence"] + 0.1)


@dataclass
class PlanComposer:
    aspect_topk: int = 4
    length_target_mean: int = 4
    ctx_merge_weight: float = 0.5
    plan_diversity_temp: float = 0.7
    length_policy: Tuple[int, int] = (2, 6)

    def compose(self, persona: Dict[str, Any], item: Dict[str, Any], platform_policy_weight: float) -> Dict[str, Any]:
        p_weights = persona.get("aspect_preference_weights", {})
        i_aspects = item.get("aspect_summary", {a: 1.0 / len(DEFAULT_ASPECT_VOCAB) for a in DEFAULT_ASPECT_VOCAB})
        merged: Dict[str, float] = {}
        for a in DEFAULT_ASPECT_VOCAB:
            merged[a] = self.ctx_merge_weight * p_weights.get(a, 1.0 / len(DEFAULT_ASPECT_VOCAB)) + (1 - self.ctx_merge_weight) * i_aspects.get(a, 1.0 / len(DEFAULT_ASPECT_VOCAB))

        aspects_sorted = sorted(merged.items(), key=lambda kv: kv[1], reverse=True)
        top_candidates = [a for a, _ in aspects_sorted[: max(self.aspect_topk * 2, self.aspect_topk + 2)]]

        def softmax(vals: List[float], temp: float) -> List[float]:
            mx = max(vals) if vals else 0.0
            exps = [math.exp((v - mx) / max(1e-6, temp)) for v in vals]
            s = sum(exps) or 1.0
            return [e / s for e in exps]

        cand_scores = [merged[a] for a in top_candidates]
        probs = softmax(cand_scores, self.plan_diversity_temp)
        aspects = []
        available = list(top_candidates)
        for _ in range(self.aspect_topk):
            if not available:
                break
            r = random.random()
            csum = 0.0
            idx = 0
            for i, p in enumerate(probs):
                csum += p
                if r <= csum:
                    idx = i
                    break
            chosen = available.pop(idx)
            aspects.append(chosen)
            if available:
                cand_scores = [merged[a] for a in available]
                probs = softmax(cand_scores, self.plan_diversity_temp)

        target_leniency = persona["baseline_leniency"] + persona.get("leniency_drift", 0.0) + 0.5 * persona.get("recent_sentiment_bias", 0.0)
        tone = "positive" if target_leniency >= 3.2 else ("negative" if target_leniency <= 2.8 else "neutral")

        min_len, max_len = self.length_policy
        conciseness = persona.get("style_vector", [0.5, 0.5, 0.5])[2]
        length_factor = max(0.5, min(1.5, 1.5 - conciseness))
        length_target = int(round(self.length_target_mean * length_factor))
        length_target = max(min_len, min(max_len, length_target))

        plan = {
            "planned_aspects": aspects,
            "tone_target": tone,
            "length_target": length_target,
            "platform_policy_weight": platform_policy_weight
        }
        return plan


@dataclass
class ReviewAuthor:
    style_alignment_weight: float = 0.7
    llm_temperature: float = 0.6
    max_revision_loops: int = 1
    generation_guidelines: str = "Be concise, specific, and comply with platform policy (no profanity)."
    cache: Dict[Tuple[str, str, Tuple[str, ...], str, int, int], str] = field(default_factory=dict)

    def _build_prompt(self, user_ctx: Dict[str, Any], item_ctx: Dict[str, Any], plan: Dict[str, Any]) -> str:
        style_vec = user_ctx.get("style_vector", [0.5, 0.5, 0.5])
        tone_hint = user_ctx.get("tone_style_prior", "neutral")
        persona_desc = (
            f"- Baseline leniency: {user_ctx.get('baseline_leniency', 3.0):.2f}\n"
            f"- Style/tone: {tone_hint}, style_vector(formality,enthusiasm,conciseness)={style_vec}\n"
            f"- Domain familiarity: {user_ctx.get('domain_familiarity', 0.5):.2f}\n"
        )
        
        # Enhanced item description with product name, category, and features
        # Normalize to strings (handle both string and list types)
        def safe_str(value, default=""):
            if not value:
                return default
            if isinstance(value, list):
                return ", ".join(str(v).strip() for v in value if v)
            return str(value).strip()
        
        item_name = safe_str(item_ctx.get("item_name", ""))
        item_category = safe_str(item_ctx.get("item_category", ""))
        item_type = safe_str(item_ctx.get("item_type", ""))
        item_description = safe_str(item_ctx.get("item_description", ""))
        
        item_desc_parts = []
        if item_name:
            item_desc_parts.append(f"- Product name: {item_name}")
        if item_category:
            item_desc_parts.append(f"- Product category: {item_category}")
        if item_type:
            item_desc_parts.append(f"- Product type: {item_type}")
        if item_description:
            item_desc_parts.append(f"- Product description: {item_description}")
        item_desc_parts.append(f"- Item reputation prior: {item_ctx.get('reputation_prior', item_ctx.get('quality_prior', 3.0)):.2f}")
        item_desc_parts.append(f"- Item tags: {', '.join(item_ctx.get('tags', item_ctx.get('domain_tags', []))) if item_ctx.get('tags', item_ctx.get('domain_tags', [])) else 'N/A'}")
        item_desc = "\n".join(item_desc_parts)
        
        plan_desc = (
            f"- Planned aspects: {', '.join(plan.get('planned_aspects', []))}\n"
            f"- Tone target: {plan.get('tone_target', 'neutral')}\n"
            f"- Target sentence count: {plan.get('length_target', 3)}\n"
            f"- Platform policy weight (0-1): {plan.get('platform_policy_weight', 0.5):.2f}\n"
        )
        
        # Enhanced few-shot examples: use 3 recent historical reviews from user as style guidance
        few_user = user_ctx.get("history_texts", [])[:3]  # Get 3 most recent reviews from this user for style guidance
        few_item = item_ctx.get("history_texts", [])[:3]  # 2-3 examples from this item to show product specifics
        
        few_shot = ""
        if few_user:
            few_shot += "\nSTYLE GUIDANCE - This user's recent review history (use these as style references):\n"
            few_shot += "Please carefully study the following reviews written by this user, and match their writing style, "
            few_shot += "tone, level of detail, vocabulary choice, and overall review structure when generating the new review.\n\n"
            for i, example in enumerate(few_user, 1):
                few_shot += f"User's Review Example {i}:\n---\n{example}\n---\n"
        if few_item:
            few_shot += f"\nExamples of reviews about this product/item:\n"
            for i, example in enumerate(few_item, 1):
                few_shot += f"Example {i}:\n---\n{example}\n---\n"
        
        guidelines = self.generation_guidelines
        
        # Enhanced task description with specific requirements
        product_info_note = ""
        if item_name:
            product_info_note = f"\nIMPORTANT: You are reviewing the product '{item_name}'. "
            if item_category:
                product_info_note += f"It is in the category: {item_category}. "
        else:
            product_info_note = "\nIMPORTANT: You are reviewing a specific product. "
        
        prompt = (
            "You are a helpful assistant that writes product reviews for an e-commerce platform.\n"
            "Follow the plan, align with the user's persona, and respect platform policies.\n\n"
            "User persona:\n"
            f"{persona_desc}\n"
            "Product/Item context:\n"
            f"{item_desc}\n"
            "Plan:\n"
            f"{plan_desc}\n"
            f"Guidelines: {guidelines}\n"
            f"{few_shot}\n"
            "Task:\n"
            f"{product_info_note}"
            "Write a natural, coherent product review that:\n"
            "- **Matches this user's writing style** - Carefully analyze the user's recent review history provided above, "
            "and mirror their style, including: vocabulary level, sentence structure, detail depth, formality level, "
            "use of technical terms, emotional expression, and overall tone consistency. This is the PRIMARY style guidance.\n"
            "- **Uses the product name/type** (mention the specific product you are reviewing)\n"
            "- **Mentions specific features, functions, and characteristics** of the product (not generic attributes)\n"
            "- **Includes usage scenarios and real-world applications** where relevant\n"
            "- **Covers the planned aspects** in separate sentences\n"
            "- **Matches the tone target** (positive/neutral/negative)\n"
            "- **Approximately meets the target sentence count**\n"
            "- **Is specific and concrete** (learn from the example reviews above about the level of detail and specificity)\n"
            "- Avoids profanity and does not include star ratings in the text\n"
            "\n"
            "IMPORTANT: The user's recent review history shown above serves as your PRIMARY style reference. "
            "Ensure your generated review matches the writing patterns, style, and tone demonstrated in those examples.\n"
            "\n"
            "Write the review now:\n"
        )
        return prompt

    def _offline_template(self, aspects: List[str], tone: str, length: int) -> str:
        verb = {"positive": "appreciate", "neutral": "note", "negative": "dislike"}.get(tone, "note")
        sentences = []
        for i, a in enumerate(aspects[:length]):
            modifier = ["clearly", "notably", "generally", "mostly", "somewhat"][i % 5]
            polarity = {"positive": "good", "neutral": "average", "negative": "poor"}.get(tone, "average")
            sentences.append(f"I {verb} the {a}; it is {modifier} {polarity}.")
        while len(sentences) < length:
            sentences.append("Overall, the experience matches expectations.")
        return " ".join(sentences)

    def generate(self, user_ctx: Dict[str, Any], item_ctx: Dict[str, Any], plan: Dict[str, Any],
                 use_llm: bool, model_name: str, max_output_tokens: int, llm_retries: int = 2) -> str:
        aspects = plan.get("planned_aspects", [])
        tone = plan.get("tone_target", "neutral")
        length = plan.get("length_target", 3)

        # Build cache key (same as before for instance cache)
        cache_key_tuple = (
            str(user_ctx.get("user_id", "")),
            str(item_ctx.get("item_id", "")),
            tuple(sorted(aspects)),
            tone,
            length,
            int(self.llm_temperature * 100),
        )
        
        # Convert to JSON-serializable string for global cache
        cache_key_str = f"ReviewAuthor:{_cache_key_to_string(cache_key_tuple)}"
        
        # Check global cache first
        if cache_key_str in llm_cache_dict:
            text = llm_cache_dict[cache_key_str]
            # Also update instance cache for quick access
            self.cache[cache_key_tuple] = text
        # Check instance cache second (for backward compatibility)
        elif cache_key_tuple in self.cache:
            text = self.cache[cache_key_tuple]
        else:
            # Cache miss: generate text
            if use_llm and not is_offline_mode():
                prompt = self._build_prompt(user_ctx, item_ctx, plan)
                text = call_gpt5_with_responses_api(
                    prompt=prompt,
                    model=model_name,
                    max_output_tokens=max_output_tokens,
                    temperature=self.llm_temperature,
                    retries=llm_retries
                ).strip()
            else:
                # OFFLINE_MODE or use_llm=False fallback
                text = self._offline_template(aspects, tone, length)
            
            # Update both caches
            self.cache[cache_key_tuple] = text
            llm_cache_dict[cache_key_str] = text

        sentences = sentence_split(text)
        target = plan.get("length_target", 3)
        if len(sentences) > target:
            sentences = sentences[:target]
        elif len(sentences) < target:
            sentences += ["Additionally, it performs as expected."] * (target - len(sentences))
        final_text = " ".join(sentences).strip()
        return final_text


@dataclass
class StarRater:
    """
    Star rating predictor with LLM + post-processing calibration.
    
    Post-processing parameters (tunable for calibration):
    - linear_bias: Linear bias correction applied to LLM output
    - sentiment_weight: Weight for sentiment-based adjustment
    - user_bias_weight: Weight for user leniency prior
    - item_bias_weight: Weight for item reputation prior
    - uncertainty_scale: Gaussian noise scale for stochastic predictions
    """
    # Post-processing calibration parameters (tunable)
    linear_bias: float = 0.0            # Additive bias correction
    sentiment_weight: float = 0.2       # Sentiment adjustment weight
    user_bias_weight: float = 0.3       # User prior weight
    item_bias_weight: float = 0.3       # Item prior weight
    uncertainty_scale: float = 0.0      # Noise scale (0 = deterministic)
    
    # Fixed LLM parameters (not tunable)
    LLM_TEMPERATURE: float = 0.2        # Fixed temperature for LLM calls
    LLM_MAX_TOKENS: int = 64            # Fixed max tokens
    
    # Internal state
    global_mean: float = 3.0            # Updated during fit
    cache: Dict[Tuple[str, float, float, Tuple[str, ...], str, int], int] = field(default_factory=dict)

    def fit_global(self, data: DataIndexer, train_idxs: List[int]) -> None:
        """
        Initialize global statistics from training data.
        Only sets global_mean; post-processing parameters are calibrated separately.
        """
        self.global_mean = data.global_mean

    def predict(self, text: str, user_prior: float, item_prior: float, rng: random.Random) -> int:
        """
        Heuristic-based prediction using sentiment analysis.
        Used as fallback when LLM is unavailable.
        """
        # Simple heuristic: sentiment-based rating with priors
        s = sentiment_score(text)  # Range: typically [-1, 1]
        
        # Base score from sentiment (map [-1,1] to [1,5])
        base_score = 3.0 + 2.0 * s
        
        # Apply post-processing adjustments
        base_score += self.linear_bias
        base_score += self.sentiment_weight * s
        
        # User/item prior adjustments
        user_adjustment = self.user_bias_weight * (user_prior - self.global_mean) / 2.0
        item_adjustment = self.item_bias_weight * (item_prior - self.global_mean) / 2.0
        base_score += user_adjustment + item_adjustment
        
        # Add noise
        if self.uncertainty_scale > 0:
            base_score += rng.gauss(0, self.uncertainty_scale)
        
        # Clip and round
        star = int(round(base_score))
        return min(5, max(1, star))

    def predict_via_llm(self, review_text: str, user_ctx: Dict[str, Any], item_ctx: Dict[str, Any],
                         model_name: str, llm_retries: int = 2) -> int:
        """
        Get raw LLM prediction using fixed temperature and prompt.
        Returns integer 1-5 WITHOUT post-processing (post-processing applied externally).
        """
        if is_offline_mode():
            # fall back to heuristic
            return self.predict(review_text, user_prior=user_ctx.get("baseline_leniency", self.global_mean),
                                item_prior=item_ctx.get("quality_prior", item_ctx.get("reputation_prior", self.global_mean)),
                                rng=random.Random(GLOBAL_SEED))
        
        # Build cache key with FIXED LLM parameters
        user_leniency = round(user_ctx.get("baseline_leniency", self.global_mean), 2)
        item_reputation = round(item_ctx.get("quality_prior", item_ctx.get("reputation_prior", self.global_mean)), 2)
        item_tags = tuple(sorted(item_ctx.get("tags", item_ctx.get("domain_tags", []))))
        
        cache_key_tuple = (
            review_text,
            user_leniency,
            item_reputation,
            item_tags,
            model_name,
            self.LLM_MAX_TOKENS  # Use fixed constant
        )
        
        cache_key_str = f"StarRater:{_cache_key_to_string(cache_key_tuple)}"
        
        # Check global cache first
        if cache_key_str in llm_cache_dict:
            parsed = llm_cache_dict[cache_key_str]
            if isinstance(parsed, (int, float)):
                parsed = int(parsed)
            self.cache[cache_key_tuple] = parsed
            return parsed
        elif cache_key_tuple in self.cache:
            return self.cache[cache_key_tuple]
        
        # Cache miss: call LLM with FIXED parameters
        def safe_str(value, default=""):
            if not value:
                return default
            if isinstance(value, list):
                return ", ".join(str(v).strip() for v in value if v)
            return str(value).strip()
        
        item_name = safe_str(item_ctx.get("item_name", ""))
        item_category = safe_str(item_ctx.get("item_category", ""))
        item_type = safe_str(item_ctx.get("item_type", ""))
        
        product_context_parts = []
        if item_name:
            product_context_parts.append(f"Product name: {item_name}")
        if item_category:
            product_context_parts.append(f"Product category: {item_category}")
        if item_type:
            product_context_parts.append(f"Product type: {item_type}")
        product_context = "\n".join(product_context_parts) if product_context_parts else "Product: (unspecified)"
        
        # Few-shot examples
        few_shot_examples = []
        user_history_reviews = user_ctx.get("history_reviews_with_stars", [])[:1]
        for rev_text, stars_val in user_history_reviews:
            few_shot_examples.append(f"Review: {rev_text}\nRating: {stars_val} stars")
        
        item_history_reviews = item_ctx.get("history_reviews_with_stars", [])[:3]
        for rev_text, stars_val in item_history_reviews:
            few_shot_examples.append(f"Review: {rev_text}\nRating: {stars_val} stars")
        
        few_shot_section = ""
        if few_shot_examples:
            few_shot_section = "\n\nExamples of reviews and their ratings:\n"
            for i, example in enumerate(few_shot_examples, 1):
                few_shot_section += f"\nExample {i}:\n---\n{example}\n---\n"
        
        # Fixed prompt structure
        prompt = (
            "You are a rating assistant. Given the following product review and context, "
            "assign an overall star rating from 1 to 5 as an integer. Consider user leniency and item reputation if helpful. "
            "Respond with only the integer 1, 2, 3, 4, or 5.\n\n"
            f"Product context:\n{product_context}\n\n"
            f"User leniency prior: {user_leniency:.2f}\n"
            f"Item reputation prior: {item_reputation:.2f}\n"
            f"Item tags: {', '.join(item_tags) if item_tags else 'N/A'}\n"
            f"{few_shot_section}\n"
            "Review to rate:\n---\n"
            f"{review_text}\n---\n"
            "Answer (only the integer 1, 2, 3, 4, or 5):"
        )
        
        # Call LLM with FIXED temperature
        text = call_gpt5_with_responses_api(
            prompt=prompt,
            model=model_name,
            max_output_tokens=self.LLM_MAX_TOKENS,
            temperature=self.LLM_TEMPERATURE,  # Fixed temperature
            retries=llm_retries
        ).strip()
        
        # Extract integer
        parsed = None
        m = re.search(r'(?<!\d)([1-5])(?!\d)', text)
        if m:
            try:
                parsed = int(m.group(1))
            except Exception:
                parsed = None
        if parsed is None:
            parsed = self.predict(review_text, user_prior=user_leniency,
                                  item_prior=item_reputation,
                                  rng=random.Random(GLOBAL_SEED))
        
        # Store in both caches
        self.cache[cache_key_tuple] = parsed
        llm_cache_dict[cache_key_str] = parsed
        
        return parsed
    
    def apply_post_processing(self, llm_rating: int, review_text: str, 
                             user_prior: float, item_prior: float, rng: random.Random) -> int:
        """
        Apply calibrated post-processing to LLM output.
        This is the differentiable/tunable correction layer.
        
        Args:
            llm_rating: Raw LLM prediction (1-5)
            review_text: Review text for sentiment analysis
            user_prior: User leniency prior
            item_prior: Item reputation prior
            rng: Random number generator for noise
            
        Returns:
            Final calibrated rating (1-5)
        """
        # Start with LLM prediction as continuous value
        score = float(llm_rating)
        
        # 1. Linear bias correction
        score += self.linear_bias
        
        # 2. Sentiment-based adjustment
        sentiment = sentiment_score(review_text)  # Range: typically [-1, 1]
        sentiment_adjustment = self.sentiment_weight * sentiment
        score += sentiment_adjustment
        
        # 3. User/item prior adjustments
        user_adjustment = self.user_bias_weight * (user_prior - self.global_mean) / 2.0
        item_adjustment = self.item_bias_weight * (item_prior - self.global_mean) / 2.0
        score += user_adjustment + item_adjustment
        
        # 4. Stochastic noise (for exploration during calibration)
        if self.uncertainty_scale > 0:
            noise = rng.gauss(0, self.uncertainty_scale)
            score += noise
        
        # 5. Clip to valid range and round
        final_rating = int(round(score))
        return min(5, max(1, final_rating))


@dataclass
class QAConsistency:
    consistency_threshold: float = 0.75
    max_auto_fix_attempts: int = 1
    penalty_weight_style_violations: float = 0.5

    def _style_compliance(self, text: str, target_len: int, policy_weight: float) -> float:
        dev = abs(len(sentence_split(text)) - target_len)
        style_len_score = math.exp(-0.5 * dev)
        # profanity penalty increases with policy weight
        toks = set(tokenize(text))
        has_profanity = any(p in toks for p in PROFANITY)
        if has_profanity:
            style_len_score *= max(0.0, 1.0 - 0.5 * max(0.0, min(1.0, policy_weight)))
        return max(0.0, min(1.0, style_len_score))

    def score(self, text: str, proposed_stars: int, plan: Dict[str, Any]) -> float:
        s = sentiment_score(text)
        star_polarity = 1 if proposed_stars >= 4 else (-1 if proposed_stars <= 2 else 0)
        sent_polarity = 1 if s > 0.15 else (-1 if s < -0.15 else 0)
        agreement = 1.0 if star_polarity == sent_polarity else (0.6 if (star_polarity == 0 or sent_polarity == 0) else 0.0)
        style_score = self._style_compliance(text, plan.get("length_target", 3), plan.get("platform_policy_weight", 0.5))
        score = (agreement + self.penalty_weight_style_violations * style_score) / (1.0 + self.penalty_weight_style_violations)
        return max(0.0, min(1.0, score))

    def maybe_revise(self, author: ReviewAuthor, user_ctx: Dict[str, Any], item_ctx: Dict[str, Any],
                     plan: Dict[str, Any], proposed_stars: int, text: str, use_llm: bool,
                     model_name: str, max_output_tokens: int, llm_retries: int = 2) -> Tuple[str, int, float, int]:
        score0 = self.score(text, proposed_stars, plan)
        if score0 >= self.consistency_threshold or self.max_auto_fix_attempts <= 0:
            return text, proposed_stars, score0, 0

        revision_count = 0
        final_text = text
        final_stars = proposed_stars
        best_score = score0
        original_guidelines = author.generation_guidelines
        try:
            for _ in range(self.max_auto_fix_attempts):
                revision_count += 1
                if proposed_stars >= 4:
                    plan["tone_target"] = "positive"
                elif proposed_stars <= 2:
                    plan["tone_target"] = "negative"
                else:
                    plan["tone_target"] = "neutral"

                temp_guidelines = "Revise to better match the target tone and maintain factuality. Avoid profanity."
                author.generation_guidelines = temp_guidelines
                try:
                    new_text = author.generate(user_ctx, item_ctx, plan, use_llm=use_llm, model_name=model_name, max_output_tokens=max_output_tokens, llm_retries=llm_retries)
                except Exception:
                    new_text = final_text
                new_score = self.score(new_text, proposed_stars, plan)
                if new_score > best_score:
                    final_text = new_text
                    best_score = new_score
                    if best_score >= self.consistency_threshold:
                        break
        finally:
            author.generation_guidelines = original_guidelines

        return final_text, final_stars, best_score, revision_count


@dataclass
class FittedParams:
    """Container holding calibrated parameter values and metadata."""
    param_values: Dict[str, Any]
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Flat representation of calibrated parameters plus metadata."""
        return {**self.param_values, "meta": self.meta}


def _sample_param_dict(rng: random.Random) -> Dict[str, Any]:
    """
    Sample post-processing calibration parameters for star rating.
    These parameters adjust LLM predictions without affecting the prompt.
    """
    def uniform(a: float, b: float) -> float:
        return a + (b - a) * rng.random()

    params = {
        # Post-processing calibration parameters (applied after LLM call)
        "linear_bias": uniform(-1.0, 1.0),          # Linear bias correction
        "sentiment_weight": uniform(0.0, 0.5),      # Sentiment adjustment weight
        "user_bias_weight": uniform(0.0, 0.6),      # User prior weight
        "item_bias_weight": uniform(0.0, 0.6),      # Item prior weight
        "uncertainty_scale": uniform(0.0, 0.3),     # Stochastic noise scale
    }
    return params


def _evaluate_candidate(
    simulator: "Simulator",
    train_idxs: List[int],
    params: Dict[str, Any],
    use_llm: bool,
) -> Tuple[float, Dict[str, float]]:
    """Apply star rating parameters, run rollout, and compute MAE objective."""
    simulator.reset_agents()
    simulator.apply_params(params)
    metrics, _ = simulator.rollout(
        idxs=train_idxs,
        use_llm=use_llm,
        max_records=len(train_idxs),
        collect_traces=False,
    )
    # Use MAE_stars as objective (lower is better)
    obj = metrics.get("MAE_stars", 1.0)
    return obj, metrics


class Calibrator(ABC):
    """Pluggable calibrator interface with shared utilities."""

    def __init__(self, name: str):
        self.name = name
        self.calibration_logs: List[str] = []
        self.history_of_trials: List[Dict[str, Any]] = []
        self.best_objective: float = float("inf")
        self.best_params: Optional[Dict[str, Any]] = None
        self.best_metrics: Dict[str, Any] = {}

    def _log(self, message: str) -> None:
        print(message)
        self.calibration_logs.append(message)

    def get_logs(self) -> List[str]:
        return list(self.calibration_logs)

    def get_history(self) -> List[Dict[str, Any]]:
        return list(self.history_of_trials)

    @abstractmethod
    def fit(
        self,
        data: DataIndexer,
        train_idxs: List[int],
        simulator: "Simulator",
        rng: random.Random,
        use_llm: bool = True,
    ) -> FittedParams:
        """Return calibrated parameters for downstream simulation."""


class RandomSearchCalibrator(Calibrator):
    """Black-box random search over star rating post-processing parameters."""

    def __init__(self, num_trials: int = 20, early_stop_patience: int = 5):
        super().__init__("random_search")
        self.num_trials = num_trials
        self.early_stop_patience = early_stop_patience

    def fit(
        self,
        data: DataIndexer,
        train_idxs: List[int],
        simulator: "Simulator",
        rng: random.Random,
        use_llm: bool = True,
    ) -> FittedParams:
        no_improve_rounds = 0
        self.calibration_logs.clear()
        self.history_of_trials.clear()
        self._log(f"\n{'='*60}")
        self._log(f"RandomSearchCalibrator (Star Rating): {self.num_trials} trials, {len(train_idxs)} training records")
        self._log(f"{'='*60}")
        for trial in range(self.num_trials):
            params = _sample_param_dict(rng)
            self._log(f"\n[RandomSearch] Trial {trial+1}/{self.num_trials}")
            obj, metrics = _evaluate_candidate(simulator, train_idxs, params, use_llm)
            entry = {"trial": trial, "params": params, "metrics": metrics, "objective": obj}
            self.history_of_trials.append(entry)

            status = "✓ NEW BEST" if obj < self.best_objective else "×"
            self._log(
                f"  MAE_stars={metrics.get('MAE_stars', 0.0):.4f} "
                f"| RMSE_stars={metrics.get('RMSE_stars', 0.0):.4f} {status}"
            )
            if obj < self.best_objective:
                self.best_objective = obj
                self.best_params = params
                self.best_metrics = metrics
                no_improve_rounds = 0
                self._log("  → new best parameters found")
            else:
                no_improve_rounds += 1
                if no_improve_rounds >= self.early_stop_patience:
                    self._log(
                        f"Early stopping: {no_improve_rounds} trials without improvement "
                        f"(patience={self.early_stop_patience})"
                    )
                    break

        if self.best_params is None:
            self.best_params = _sample_param_dict(rng)

        self._log(f"\nBest MAE_stars: {self.best_objective:.6f}")
        meta = {
            "calibrator_name": self.name,
            "num_trials": self.num_trials,
            "early_stop_patience": self.early_stop_patience,
            "best_objective": self.best_objective,
            "n_evaluations": len(self.history_of_trials),
        }
        return FittedParams(param_values=self.best_params, meta=meta)


class LogitHeadCalibrator(Calibrator):
    """Placeholder for future logistic-head calibrator."""

    def __init__(self):
        super().__init__("logit_head")

    def fit(
        self,
        data: DataIndexer,
        train_idxs: List[int],
        simulator: "Simulator",
        rng: random.Random,
        use_llm: bool = True,
    ) -> FittedParams:
        raise NotImplementedError("LogitHeadCalibrator is not implemented yet.")


class SBICalibrator(Calibrator):
    """Placeholder for Simulation-Based Inference calibrator."""

    def __init__(self):
        super().__init__("sbi")

    def fit(
        self,
        data: DataIndexer,
        train_idxs: List[int],
        simulator: "Simulator",
        rng: random.Random,
        use_llm: bool = True,
    ) -> FittedParams:
        raise NotImplementedError("SBICalibrator is not implemented yet.")


class BoCalibrator(Calibrator):
    """Bayesian Optimization calibrator using BoTorch + optional TuRBO trust region."""

    PARAM_BOUNDS = {
        "linear_bias": (-1.0, 1.0),
        "sentiment_weight": (0.0, 0.5),
        "user_bias_weight": (0.0, 0.6),
        "item_bias_weight": (0.0, 0.6),
        "uncertainty_scale": (0.0, 0.3),
    }

    def __init__(
        self,
        n_trials: int = 40,
        acquisition_function: str = "ei",
        use_turbo: bool = True,
        turbo_config: Optional[Dict[str, Any]] = None,
    ):
        if not BOTORCH_AVAILABLE or torch is None:
            raise ImportError(
                "BoTorch + torch are required for BoCalibrator. "
                "Install botorch[default] and gpytorch to enable Bayesian optimization."
            )
        super().__init__("bo")
        self.n_trials = max(5, n_trials)
        self.acquisition_function = acquisition_function.lower()
        self.use_turbo = use_turbo
        self.param_names = list(self.PARAM_BOUNDS.keys())
        bounds_list = [[self.PARAM_BOUNDS[name][0], self.PARAM_BOUNDS[name][1]] for name in self.param_names]
        self.base_bounds = torch.tensor(bounds_list, dtype=torch.double).T  # actual bounds (2, d)
        self.base_lows = self.base_bounds[0].clone()
        self.base_highs = self.base_bounds[1].clone()
        self.current_bounds_actual = self.base_bounds.clone()
        default_turbo = {
            "trust_region_size": 0.8,
            "success_tolerance": 3,
            "failure_tolerance": 10,
            "expansion_factor": 2.0,
            "contraction_factor": 0.5,
            "min_trust_region": 1e-4,
            "max_trust_region": 1.0,
        }
        self.turbo_config = default_turbo if turbo_config is None else {**default_turbo, **turbo_config}
        self.turbo_state: Optional[Dict[str, Any]] = None
        if self.use_turbo:
            self.turbo_state = {
                "center": None,
                "trust_region_size": self.turbo_config["trust_region_size"],
                "successes": 0,
                "failures": 0,
                "best_value": float("inf"),
            }
        self.X_train: Optional[torch.Tensor] = None
        self.Y_train: Optional[torch.Tensor] = None
        self.bounds = self._actual_to_unit_bounds(self.current_bounds_actual)

    def _actual_to_unit_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        return (tensor - self.base_lows) / (self.base_highs - self.base_lows)

    def _unit_to_actual_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor * (self.base_highs - self.base_lows) + self.base_lows

    def _actual_to_unit_bounds(self, actual_bounds: torch.Tensor) -> torch.Tensor:
        lower = self._actual_to_unit_tensor(actual_bounds[0])
        upper = self._actual_to_unit_tensor(actual_bounds[1])
        return torch.stack([lower, upper], dim=0)

    def _tensor_to_params(self, tensor: torch.Tensor) -> Dict[str, float]:
        params: Dict[str, float] = {}
        for idx, name in enumerate(self.param_names):
            low, high = self.PARAM_BOUNDS[name]
            val = float(torch.clamp(tensor[idx], min=low, max=high).item())
            params[name] = val
        return params

    def _params_to_tensor(self, params: Dict[str, float]) -> torch.Tensor:
        return torch.tensor([params[name] for name in self.param_names], dtype=torch.double)

    def _unit_to_params(self, unit_tensor: torch.Tensor) -> Dict[str, float]:
        actual_tensor = self._unit_to_actual_tensor(unit_tensor)
        return self._tensor_to_params(actual_tensor)

    def _sample_initial_points(self, n_init: int, seed: int) -> torch.Tensor:
        return draw_sobol_samples(bounds=self.bounds, n=1, q=n_init, seed=seed).squeeze(0).to(dtype=torch.double)

    def _fit_gp(self, X: torch.Tensor, Y: torch.Tensor) -> SingleTaskGP:
        gp = SingleTaskGP(X, Y)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)
        return gp

    def _get_acquisition(self, gp: SingleTaskGP, best_f: float) -> LogExpectedImprovement:
        # Maximize log-EI to avoid numerical issues with standard EI
        return LogExpectedImprovement(gp, best_f=best_f)

    def _optimize_acquisition(self, acq_func: LogExpectedImprovement) -> torch.Tensor:
        candidates, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=self.bounds,
            q=1,
            num_restarts=20,
            raw_samples=64,
        )
        return candidates[0]

    def _update_turbo_state(self, params_tensor: torch.Tensor, objective_value: float) -> None:
        if not self.use_turbo or self.turbo_state is None:
            return
        state = self.turbo_state
        candidate = params_tensor.detach().cpu().numpy()
        if state["center"] is None:
            state["center"] = candidate
            state["best_value"] = objective_value
            return
        if objective_value < state["best_value"]:
            state["best_value"] = objective_value
            state["center"] = candidate
            state["successes"] += 1
            state["failures"] = 0
            if state["successes"] >= self.turbo_config["success_tolerance"]:
                state["trust_region_size"] = min(
                    state["trust_region_size"] * self.turbo_config["expansion_factor"],
                    self.turbo_config["max_trust_region"],
                )
                state["successes"] = 0
                self._log(f"  [TuRBO] Expanded trust region to {state['trust_region_size']:.4f}")
        else:
            state["failures"] += 1
            state["successes"] = 0
            if state["failures"] >= self.turbo_config["failure_tolerance"]:
                state["trust_region_size"] = max(
                    state["trust_region_size"] * self.turbo_config["contraction_factor"],
                    self.turbo_config["min_trust_region"],
                )
                state["failures"] = 0
                self._log(f"  [TuRBO] Contracted trust region to {state['trust_region_size']:.4f}")

    def _refresh_bounds(self) -> None:
        if not self.use_turbo or self.turbo_state is None or self.turbo_state["center"] is None:
            self.current_bounds_actual = self.base_bounds.clone()
        else:
            center = self.turbo_state["center"]
            tr_size = self.turbo_state["trust_region_size"]
            bounds_list = []
            for idx, name in enumerate(self.param_names):
                low, high = self.PARAM_BOUNDS[name]
                span = high - low
                tr_half = tr_size * span / 2.0
                tr_low = max(low, center[idx] - tr_half)
                tr_high = min(high, center[idx] + tr_half)
                bounds_list.append([tr_low, tr_high])
            self.current_bounds_actual = torch.tensor(bounds_list, dtype=torch.double).T
        self.bounds = self._actual_to_unit_bounds(self.current_bounds_actual)

    def fit(
        self,
        data: DataIndexer,
        train_idxs: List[int],
        simulator: "Simulator",
        rng: random.Random,
        use_llm: bool = True,
    ) -> FittedParams:
        self.calibration_logs.clear()
        self.history_of_trials.clear()
        self.best_objective = float("inf")
        self.best_params = None
        self.best_metrics = {}
        torch.manual_seed(rng.randint(0, 2**31 - 1))
        self._log(
            f"\n{'='*60}\nBoCalibrator (BoTorch BO): {self.n_trials} trials, TuRBO={self.use_turbo}\n{'='*60}"
        )

        n_init = min(10, max(4, self.n_trials // 3))
        X_init_unit = self._sample_initial_points(n_init, rng.randint(0, 2**31 - 1))
        Y_init_list = []

        for idx in range(n_init):
            unit_point = X_init_unit[idx]
            params = self._unit_to_params(unit_point)
            obj, metrics = _evaluate_candidate(simulator, train_idxs, params, use_llm)
            self.history_of_trials.append(
                {"iteration": idx, "params": params, "metrics": metrics, "objective": obj}
            )
            self._log(
                f"[BO] Init {idx+1}/{n_init} → MAE_stars={metrics.get('MAE_stars', 0.0):.4f}, "
                f"RMSE_stars={metrics.get('RMSE_stars', 0.0):.4f}"
            )
            if obj < self.best_objective:
                self.best_objective = obj
                self.best_params = params
                self.best_metrics = metrics
            Y_init_list.append(-obj)  # negate to convert to maximization

        self.X_train = X_init_unit.clone()
        self.Y_train = torch.tensor(Y_init_list, dtype=torch.double).unsqueeze(-1)

        if self.use_turbo and self.turbo_state is not None and self.best_params is not None:
            self.turbo_state["center"] = self._params_to_tensor(self.best_params).cpu().numpy()
            self.turbo_state["best_value"] = self.best_objective
            self._refresh_bounds()

        # Main BO loop
        for iteration in range(n_init, self.n_trials):
            self._log(f"\n[BO] Iteration {iteration+1}/{self.n_trials}")
            gp_model = self._fit_gp(self.X_train, self.Y_train)
            best_f = torch.max(self.Y_train).item()
            acq_func = self._get_acquisition(gp_model, best_f)
            candidate_unit = self._optimize_acquisition(acq_func)
            params = self._unit_to_params(candidate_unit)
            obj, metrics = _evaluate_candidate(simulator, train_idxs, params, use_llm)
            self._log(
                f"  Candidate MAE_stars={metrics.get('MAE_stars', 0.0):.4f}, "
                f"RMSE_stars={metrics.get('RMSE_stars', 0.0):.4f}"
            )

            # Append training data
            self.X_train = torch.cat([self.X_train, candidate_unit.unsqueeze(0)], dim=0)
            self.Y_train = torch.cat([self.Y_train, torch.tensor([[-obj]], dtype=torch.double)], dim=0)

            if obj < self.best_objective or self.best_params is None:
                self.best_objective = obj
                self.best_params = params
                self.best_metrics = metrics
                self._log(f"  → New best MAE_stars={self.best_objective:.6f}")

            if self.use_turbo:
                candidate_actual = self._unit_to_actual_tensor(candidate_unit)
                self._update_turbo_state(candidate_actual, obj)
                self._refresh_bounds()

            self.history_of_trials.append(
                {
                    "iteration": iteration,
                    "params": params,
                    "metrics": metrics,
                    "objective": obj,
                    "best_objective": self.best_objective,
                }
            )

        meta = {
            "calibrator_name": self.name,
            "n_trials": self.n_trials,
            "acquisition_function": self.acquisition_function,
            "use_turbo": self.use_turbo,
            "best_objective": self.best_objective,
            "n_evaluations": len(self.history_of_trials),
        }
        self._log(f"\nBoCalibrator best MAE_stars: {self.best_objective:.6f}")
        return FittedParams(param_values=self.best_params or _sample_param_dict(rng), meta=meta)


class EvoCalibrator(Calibrator):
    """Evolutionary calibrator powered by EvoTorch GeneticAlgorithm."""

    PARAM_BOUNDS = {
        "linear_bias": (-1.0, 1.0),
        "sentiment_weight": (0.0, 0.5),
        "user_bias_weight": (0.0, 0.6),
        "item_bias_weight": (0.0, 0.6),
        "uncertainty_scale": (0.0, 0.3),
    }

    def __init__(
        self,
        n_generations: int = 15,
        population_size: int = 64,
        mutation_sigma: float = 0.05,
        crossover_eta: float = 8.0,
        tournament_size: int = 3,
        device: Optional[str] = None,
    ):
        if not EVOTORCH_AVAILABLE or torch is None:
            raise ImportError(
                "EvoTorch (and torch) are required for EvoCalibrator. "
                "Please install evotorch==0.4.* and torch."
            )
        super().__init__("evo")
        self.n_generations = n_generations
        self.population_size = population_size
        self.mutation_sigma = mutation_sigma
        self.crossover_eta = crossover_eta
        self.tournament_size = tournament_size
        self.param_names = list(self.PARAM_BOUNDS.keys())
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.problem: Optional[Problem] = None
        self.algorithm: Optional[GeneticAlgorithm] = None

    def _clamp(self, name: str, value: float) -> float:
        low, high = self.PARAM_BOUNDS[name]
        return max(low, min(high, value))

    def _tensor_to_params(self, tensor_values: torch.Tensor) -> Dict[str, float]:
        params = {}
        for idx, name in enumerate(self.param_names):
            params[name] = self._clamp(name, float(tensor_values[idx].item()))
        return params

    def _fitness_single(
        self,
        tensor_values: torch.Tensor,
        simulator: "Simulator",
        train_idxs: List[int],
        use_llm: bool,
    ) -> float:
        params = self._tensor_to_params(tensor_values)
        obj, metrics = _evaluate_candidate(simulator, train_idxs, params, use_llm)
        entry = {
            "params": params,
            "metrics": metrics,
            "objective": obj,
            "generation": len(self.history_of_trials) // max(1, self.population_size),
            "individual": len(self.history_of_trials) % max(1, self.population_size),
        }
        self.history_of_trials.append(entry)
        if obj < self.best_objective:
            self.best_objective = obj
            self.best_params = params
            self.best_metrics = metrics
            self._log(
                f"[Evo] New best MAE_stars={metrics.get('MAE_stars', 0.0):.4f} "
                f"| RMSE_stars={metrics.get('RMSE_stars', 0.0):.4f}"
            )
        return -obj  # maximize negative MAE

    def _create_fitness_function(
        self,
        simulator: "Simulator",
        train_idxs: List[int],
        use_llm: bool,
    ):
        def fitness_fn(solution: torch.Tensor) -> torch.Tensor:
            if solution.dim() == 2:
                scores = [
                    self._fitness_single(sol, simulator, train_idxs, use_llm)
                    for sol in solution
                ]
                return torch.tensor(scores, dtype=torch.float32, device=solution.device)
            return torch.tensor(
                self._fitness_single(solution, simulator, train_idxs, use_llm),
                dtype=torch.float32,
                device=solution.device,
            )

        return fitness_fn

    def _create_problem(
        self,
        simulator: "Simulator",
        train_idxs: List[int],
        use_llm: bool,
    ) -> Problem:
        bounds_lower = torch.tensor(
            [self.PARAM_BOUNDS[name][0] for name in self.param_names],
            dtype=torch.float32,
            device=self.device,
        )
        bounds_upper = torch.tensor(
            [self.PARAM_BOUNDS[name][1] for name in self.param_names],
            dtype=torch.float32,
            device=self.device,
        )
        fitness_fn = self._create_fitness_function(simulator, train_idxs, use_llm)
        problem = Problem(
            objective_sense="max",
            solution_length=len(self.param_names),
            bounds=(bounds_lower, bounds_upper),
            objective_func=fitness_fn,
            dtype=torch.float32,
            device=self.device,
        )
        return problem

    def _create_algorithm(self, problem: Problem) -> GeneticAlgorithm:
        mutation = GaussianMutation(problem, stdev=self.mutation_sigma)
        crossover = SimulatedBinaryCrossOver(
            problem,
            tournament_size=self.tournament_size,
            eta=self.crossover_eta,
        )
        algorithm = GeneticAlgorithm(
            problem,
            popsize=self.population_size,
            operators=[crossover, mutation],
        )
        return algorithm

    def fit(
        self,
        data: DataIndexer,
        train_idxs: List[int],
        simulator: "Simulator",
        rng: random.Random,
        use_llm: bool = True,
    ) -> FittedParams:
        self.calibration_logs.clear()
        self.history_of_trials.clear()
        self.best_objective = float("inf")
        self.best_params = None
        self.best_metrics = {}

        torch.manual_seed(rng.randint(0, 2**31 - 1))
        self._log(
            f"\n{'='*60}\nEvoCalibrator (EvoTorch GA): "
            f"{self.n_generations} generations x {self.population_size} pop\n{'='*60}"
        )

        self.problem = self._create_problem(simulator, train_idxs, use_llm)
        self.algorithm = self._create_algorithm(self.problem)

        best_history: List[float] = []
        mean_history: List[float] = []

        for gen in range(self.n_generations):
            self.algorithm.step()
            population = self.algorithm.population
            fitness_vals = population.evals
            best_fitness = float(torch.max(fitness_vals))
            mean_fitness = float(torch.mean(fitness_vals))
            best_history.append(best_fitness)
            mean_history.append(mean_fitness)
            best_mae = -best_fitness
            mean_mae = -mean_fitness
            self._log(
                f"[Evo] Generation {gen+1:02d}/{self.n_generations}: "
                f"best={best_mae:.6f} MAE, mean={mean_mae:.6f}"
            )

        population = self.algorithm.population
        best_idx = torch.argmax(population.evals)
        best_solution = population.values[best_idx]
        final_params = self._tensor_to_params(best_solution)
        if self.best_params is None:
            self.best_params = final_params
            self.best_objective = -float(population.evals[best_idx])

        meta = {
            "calibrator_name": self.name,
            "n_generations": self.n_generations,
            "population_size": self.population_size,
            "mutation_sigma": self.mutation_sigma,
            "crossover_eta": self.crossover_eta,
            "tournament_size": self.tournament_size,
            "best_objective": self.best_objective,
            "n_evaluations": len(self.history_of_trials),
        }
        self._log(f"\nEvoCalibrator best MAE_stars: {self.best_objective:.6f}")
        return FittedParams(param_values=self.best_params or final_params, meta=meta)


CALIBRATOR_REGISTRY = {
    "logit_head": LogitHeadCalibrator,
    "random_search": RandomSearchCalibrator,
    "sbi": SBICalibrator,
    "bo": BoCalibrator,
    "evo": EvoCalibrator,
}


def get_calibrator(name: str, **kwargs) -> Calibrator:
    """Factory for pluggable calibrators."""
    if name not in CALIBRATOR_REGISTRY:
        raise ValueError(f"Unknown calibrator '{name}'. Available: {list(CALIBRATOR_REGISTRY.keys())}")
    return CALIBRATOR_REGISTRY[name](**kwargs)


@dataclass
class Evaluator:
    """Simplified evaluator focusing only on star rating metrics."""
    metric_definitions: List[str] = field(default_factory=lambda: ["MAE_stars", "RMSE_stars"])
    last_eval_metrics: Dict[str, float] = field(default_factory=dict)
    by_segment_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def compute_overall_metrics(self, records: List[Dict[str, Any]], preds: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Compute star rating metrics only:
        - MAE_stars: Mean Absolute Error of star ratings
        - RMSE_stars: Root Mean Squared Error of star ratings
        """
        star_abs_errors: List[float] = []
        star_squared_errors: List[float] = []

        for gt, pr in zip(records, preds):
            y = int(gt.get("stars", 3))
            yhat = int(pr.get("stars", 3))
            error = y - yhat
            star_abs_errors.append(abs(error))
            star_squared_errors.append(error ** 2)

        n = len(records) or 1
        
        mae_stars = sum(star_abs_errors) / n if star_abs_errors else 0.0
        rmse_stars = math.sqrt(sum(star_squared_errors) / n) if star_squared_errors else 0.0

        metrics = {
            "MAE_stars": mae_stars,
            "RMSE_stars": rmse_stars,
        }
        return metrics

    def compute_metrics(self, records: List[Dict[str, Any]], preds: List[Dict[str, Any]], data: DataIndexer) -> Dict[str, Any]:
        if len(records) != len(preds):
            raise ValueError("Records and predictions length mismatch in evaluator.")

        overall_metrics = self.compute_overall_metrics(records, preds)
        self.last_eval_metrics = overall_metrics

        # Segment metrics by user frequency tertiles
        user_counts = Counter([gt["user_id"] for gt in records])
        freqs = list(user_counts.values())
        if freqs:
            sorted_freqs = sorted(freqs)
            t1 = sorted_freqs[len(sorted_freqs) // 3]
            t2 = sorted_freqs[2 * len(sorted_freqs) // 3]
        else:
            t1 = t2 = 0

        segs = {"low": [], "mid": [], "high": []}
        for i, gt in enumerate(records):
            c = user_counts[gt["user_id"]]
            seg = "low" if c <= t1 else ("mid" if c <= t2 else "high")
            segs[seg].append(i)

        self.by_segment_metrics = {}
        for seg, idxs in segs.items():
            if not idxs:
                continue
            seg_records = [records[i] for i in idxs]
            seg_preds = [preds[i] for i in idxs]
            seg_metrics = self.compute_overall_metrics(seg_records, seg_preds)
            self.by_segment_metrics[seg] = seg_metrics

        return {"overall": overall_metrics, "by_segment": self.by_segment_metrics}


@dataclass
class Simulator:
    data: DataIndexer
    persona: PersonaProfiler
    item_profiler: ItemProfiler
    planner: PlanComposer
    author: ReviewAuthor
    rater: StarRater
    qa: QAConsistency
    evaluator: Evaluator
    platform_policy_weight: float = 0.5
    rng: random.Random = field(default_factory=lambda: random.Random(GLOBAL_SEED))
    model_name: str = field(default_factory=lambda: os.environ.get("OPENAI_MODEL", "gpt-4.1-mini"))
    max_output_tokens: int = 600
    llm_retries: int = 2
    base_seed: int = GLOBAL_SEED

    def apply_params(self, params: Dict[str, Any]) -> None:
        """Apply post-processing calibration parameters to star rater."""
        self.rater.linear_bias = float(params.get("linear_bias", self.rater.linear_bias))
        self.rater.sentiment_weight = float(params.get("sentiment_weight", self.rater.sentiment_weight))
        self.rater.user_bias_weight = float(params.get("user_bias_weight", self.rater.user_bias_weight))
        self.rater.item_bias_weight = float(params.get("item_bias_weight", self.rater.item_bias_weight))
        self.rater.uncertainty_scale = float(params.get("uncertainty_scale", self.rater.uncertainty_scale))

    def reset_agents(self) -> None:
        # Reset RNG and agent internal mutable state; rebuild persona/item priors from training data only
        # NOTE: We do NOT clear LLM caches here because:
        # 1. LLM calls are pure functions (same input → same output)
        # 2. Cache keys already include all relevant parameters (temperature, model_name, etc.)
        # 3. Sharing caches across trials does not affect trial fairness, but saves significant LLM API calls
        self.rng = random.Random(self.base_seed)
        self.persona.construct(self.data)
        self.item_profiler.construct(self.data)
        # Do NOT clear caches - keep them global across trials for efficiency
        # self.author.cache.clear()  # REMOVED: Keep cache for efficiency
        # self.rater.cache.clear()   # REMOVED: Keep cache for efficiency
        train_idxs = self.data.split_index.get("train", [])
        self.rater.fit_global(self.data, train_idxs)

    def rollout(self, idxs: Optional[List[int]] = None, use_llm: bool = True, max_records: Optional[int] = None, collect_traces: bool = False,
                model_name: Optional[str] = None, max_output_tokens: Optional[int] = None, traces_file_path: Optional[str] = None) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
        """
        Simplified rollout for star rating prediction only.
        Uses existing review text from data and predicts stars using LLM + post-processing.
        """
        if idxs is None:
            idxs = list(range(len(self.data.interactions)))
        if max_records is not None:
            idxs = idxs[:max_records]

        preds: List[Dict[str, Any]] = []
        records: List[Dict[str, Any]] = []
        model = model_name or self.model_name

        total_records = len(idxs)
        progress_interval = max(1, min(10, total_records // 10))
        
        for idx_pos, i in enumerate(idxs):
            # Print progress
            if idx_pos % progress_interval == 0 or idx_pos == total_records - 1:
                progress_pct = 100.0 * (idx_pos + 1) / total_records
                print(f"  Processing record {idx_pos+1}/{total_records} ({progress_pct:.1f}%)", end="\r")
            
            rec = self.data.interactions[i]
            user_id = rec["user_id"]
            item_id = rec["item_id"]
            # Use existing review text from data
            review_text = str(rec.get("review", ""))

            # Get user and item context for priors
            user_ctx_full = self.persona.get_persona(user_id, self.data)
            item_ctx_full = self.data.get_item_context(item_id)
            
            # LLM-based star prediction
            user_prior = user_ctx_full.get("baseline_leniency", self.data.global_mean)
            item_prior = item_ctx_full.get("quality_prior", item_ctx_full.get("reputation_prior", self.data.global_mean))
            
            if use_llm and not is_offline_mode():
                # Primary path: LLM prediction + post-processing
                stars_llm = self.rater.predict_via_llm(
                    review_text, 
                    user_ctx=user_ctx_full, 
                    item_ctx=item_ctx_full,
                    model_name=model, 
                    llm_retries=self.llm_retries
                )
                # Apply calibrated post-processing layer
                final_stars = self.rater.apply_post_processing(
                    llm_rating=stars_llm,
                    review_text=review_text,
                    user_prior=user_prior,
                    item_prior=item_prior,
                    rng=self.rng
                )
            else:
                # Fallback: heuristic-based prediction
                final_stars = self.rater.predict(review_text, user_prior=user_prior, item_prior=item_prior, rng=self.rng)

            pred = {
                "user_id": user_id,
                "item_id": item_id,
                "stars": final_stars,
            }
            preds.append(pred)
            records.append(rec)
        
        print()  # New line after progress

        results = self.evaluator.compute_metrics(records, preds, self.data)
        overall = results.get("overall", {})
        return overall, preds


def build_network_and_agents(data_bundle: Dict[str, Any], seed: int) -> Tuple[DataIndexer, PersonaProfiler, ItemProfiler, PlanComposer, ReviewAuthor, StarRater, QAConsistency, Evaluator, Simulator]:
    rng = random.Random(seed)
    interactions = data_bundle["interactions"]
    user_profiles = data_bundle["user_profiles"]
    item_metadata = data_bundle["item_metadata"]
    history_reviews = data_bundle.get("history_reviews", [])

    data_indexer = DataIndexer(interactions=interactions, user_profiles=user_profiles, item_metadata=item_metadata, history_reviews=history_reviews)
    persona = PersonaProfiler()
    persona.construct(data_indexer)
    item_profiler = ItemProfiler()
    item_profiler.construct(data_indexer)
    planner = PlanComposer()
    author = ReviewAuthor()
    rater = StarRater()
    train_idxs = data_indexer.split_index.get("train", [])
    rater.fit_global(data_indexer, train_idxs)
    qa = QAConsistency()
    evaluator = Evaluator()
    sim = Simulator(
        data=data_indexer,
        persona=persona,
        item_profiler=item_profiler,
        planner=planner,
        author=author,
        rater=rater,
        qa=qa,
        evaluator=evaluator,
        platform_policy_weight=data_indexer.platform_policy_weight,
        rng=rng,
        model_name=os.environ.get("OPENAI_MODEL", "gpt-4.1-mini"),
        base_seed=seed
    )
    return data_indexer, persona, item_profiler, planner, author, rater, qa, evaluator, sim


def holdout_split(data_indexer: DataIndexer, seed: int = GLOBAL_SEED) -> Tuple[List[int], List[int], List[int]]:
    """
    Split data into train (for calibration), valid (for validation), and test sets.
    First, get train/test split from data_indexer.
    Then, randomly split train set into train (80%) and valid (20%) for calibration/validation.
    
    Args:
        data_indexer: DataIndexer instance with split_index
        seed: Random seed for reproducible splits
    
    Returns:
        Tuple of (train_idxs, valid_idxs, test_idxs)
    """
    # Get original train and test indices
    train_idxs = data_indexer.split_index.get("train", [])
    test_idxs = data_indexer.split_index.get("test", [])
    
    # If no train/test split exists, create one first
    if not train_idxs or not test_idxs:
        all_idxs = list(range(len(data_indexer.interactions)))
        rng_temp = random.Random(seed)
        rng_temp.shuffle(all_idxs)
        cut = int(0.8 * len(all_idxs))
        train_idxs, test_idxs = all_idxs[:cut], all_idxs[cut:]
    
    # Split train set into train (80%) and valid (20%) for calibration/validation
    # Use seeded random for reproducibility
    rng = random.Random(seed)
    train_list = train_idxs[:]  # Make a copy
    rng.shuffle(train_list)
    train_cut = int(0.8 * len(train_list))
    train_calibration_idxs = train_list[:train_cut]
    valid_validation_idxs = train_list[train_cut:]
    
    return train_calibration_idxs, valid_validation_idxs, test_idxs


def save_results(calibrated_params: Dict[str, Any], eval_metrics: Dict[str, Any], ablation: Optional[Dict[str, Any]] = None, output_folder: Optional[str] = None) -> None:
    ensure_data_dir()
    
    # Save to output_folder if provided, otherwise to DATA_DIR
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)
        eval_metrics_file = os.path.join(output_folder, "evaluation_metrics.json")
        ablation_report_file = os.path.join(output_folder, "ablation_report.json") if ablation is not None else None
    else:
        eval_metrics_file = EVAL_METRICS_FILE
        ablation_report_file = ABLATION_REPORT_FILE if ablation is not None else None
    
    # Always save calibrated_params to DATA_DIR (backward compatibility)
    with open(CALIBRATED_PARAMS_FILE, "w", encoding="utf-8") as f:
        json.dump(calibrated_params, f, indent=2)
    
    # Save evaluation metrics
    with open(eval_metrics_file, "w", encoding="utf-8") as f:
        json.dump(eval_metrics, f, indent=2)
    
    # Save ablation report if provided
    if ablation is not None and ablation_report_file:
        with open(ablation_report_file, "w", encoding="utf-8") as f:
            json.dump(ablation, f, indent=2)


def save_calibration_results(
    fitted_params: FittedParams,
    calibrator: Calibrator,
    args: argparse.Namespace,
    data_dir: str,
    train_calibration_idxs: List[int],
    valid_validation_idxs: List[int],
    eval_metrics: Dict[str, Any],
    output_folder: Optional[str] = None
) -> str:
    """
    Save calibration results (optimal parameters and config) to a persistent folder in DATA_PATH.
    
    Args:
        fitted_params: Calibrated parameters and metadata
        calibrator: Calibrator instance (provides logs/history metadata)
        args: Command-line arguments (configuration)
        data_dir: DATA_DIR path
        train_calibration_idxs: Training indices used for calibration
        valid_validation_idxs: Validation indices used for evaluation
        eval_metrics: Evaluation metrics on validation set
    
    Returns:
        Path to the created calibration output folder
    """
    # Use provided output_folder or create a new one with timestamp
    if output_folder is None:
        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_folder_name = f"outputs_calibration_{timestamp}"
        output_folder = os.path.join(data_dir, output_folder_name)
    os.makedirs(output_folder, exist_ok=True)
    output_folder_name = os.path.basename(output_folder)
    
    # Save config.json (calibration configuration)
    config_data = {
        "config": {
            "data_folder": DATA_PATH,
            "seed": args.seed,
            "num_trials": getattr(calibrator, "num_trials", args.num_trials),
            "early_stop_patience": getattr(calibrator, "early_stop_patience", args.early_stop_patience),
            "max_records": args.max_records,
            "max_validation_records": args.max_validation_records,
            "use_llm": args.use_llm,
            "model_name": args.model_name,
            "max_output_tokens": args.max_output_tokens,
            "llm_retries": args.llm_retries,
            "offline_mode": args.offline == 1,
            "output_folder": output_folder_name,
            "verbose": True
        },
        "data_split": {
            "train_calibration_count": len(train_calibration_idxs),
            "valid_validation_count": len(valid_validation_idxs),
            "train_calibration_ratio": 0.8,
            "valid_validation_ratio": 0.2
        },
        "seed": args.seed,
        "calibrator_type": fitted_params.meta.get("calibrator_name", calibrator.name)
    }
    
    config_file = os.path.join(output_folder, "config.json")
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2)
    
    # Save calibration training logs
    log_file = os.path.join(output_folder, "calibration_log.txt")
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("\n".join(calibrator.get_logs()))
    print(f"Calibration logs saved to: {log_file}")
    
    # Save calibrated_parameters.json (post-processing parameters only)
    best_params = fitted_params.param_values
    history = calibrator.get_history()
    calibrated_params_data = {
        # Optimal post-processing calibration parameters
        "rater_params": {
            "linear_bias": best_params.get("linear_bias", 0.0),
            "sentiment_weight": best_params.get("sentiment_weight", 0.2),
            "user_bias_weight": best_params.get("user_bias_weight", 0.3),
            "item_bias_weight": best_params.get("item_bias_weight", 0.3),
            "uncertainty_scale": best_params.get("uncertainty_scale", 0.0)
        },
        
        # Metadata
        "meta": {
            "seed": args.seed,
            "calibrator_name": fitted_params.meta.get("calibrator_name", calibrator.name),
            "optimal_objective_value": float(calibrator.best_objective) if calibrator.best_objective != float("inf") else None,
            "n_optimization_trials": len(history),
            "optimization_history_length": len(history),
            "best_trial_index": None,  # Will be set if found
            "validation_metrics": eval_metrics.get("overall", {}),
            "calibration_config": {
                "num_trials": getattr(calibrator, "num_trials", getattr(args, "num_trials", None)),
                "early_stop_patience": getattr(calibrator, "early_stop_patience", getattr(args, "early_stop_patience", None)),
                "objective_type": "MAE_stars",
                "objective_formula": "Mean Absolute Error of star ratings",
                "metric_components": [
                    "MAE_stars",
                    "RMSE_stars"
                ]
            }
        },
        
        # Optimization history (last 100 trials for space efficiency)
        "optimization_history": history[-100:] if len(history) > 100 else history
    }
    calibrated_params_data["meta"].update(fitted_params.meta)
    
    # Find best trial index
    best_trial_idx = None
    for i, trial in enumerate(history):
        if trial.get("params") == best_params:
            best_trial_idx = i
            break
    calibrated_params_data["meta"]["best_trial_index"] = best_trial_idx
    
    calibrated_params_file = os.path.join(output_folder, "calibrated_parameters.json")
    with open(calibrated_params_file, "w", encoding="utf-8") as f:
        json.dump(calibrated_params_data, f, indent=2)
    
    return output_folder


def run_ablation(sim: Simulator, test_idxs: List[int], base_params: Dict[str, Any], use_llm: bool, model_name: str, max_output_tokens: int, max_records: int) -> Dict[str, Any]:
    results = {}
    configs = {
        "base": base_params,
        "no_peer_influence": {**base_params, "neighbor_weight": 0.0},
        "persona_driven_planning": {**base_params, "ctx_merge_weight": min(0.8, max(0.2, (base_params.get("ctx_merge_weight", 0.5) + 0.2)))},
        "item_driven_planning": {**base_params, "ctx_merge_weight": min(0.8, max(0.2, (base_params.get("ctx_merge_weight", 0.5) - 0.2)))},
    }
    for name, cfg in configs.items():
        sim.reset_agents()
        sim.apply_params(cfg)
        metrics, _ = sim.rollout(idxs=test_idxs, use_llm=use_llm, max_records=max_records, collect_traces=False,
                                 model_name=model_name, max_output_tokens=max_output_tokens)
        results[name] = metrics
    return results


def load_calibrated_params(checkpoint_path: str) -> Dict[str, Any]:
    """
    Load calibrated star rating post-processing parameters from checkpoint folder.
    
    Args:
        checkpoint_path: Path to checkpoint folder containing calibrated_parameters.json
    
    Returns:
        Dict[str, Any]: Post-processing parameters dictionary for simulator.apply_params()
        Contains: linear_bias, sentiment_weight, user_bias_weight, item_bias_weight, uncertainty_scale
    
    Raises:
        FileNotFoundError: If calibrated_parameters.json is not found
        ValueError: If the file format is invalid
    """
    calibrated_params_file = os.path.join(checkpoint_path, "calibrated_parameters.json")
    if not os.path.exists(calibrated_params_file):
        raise FileNotFoundError(f"Calibrated parameters file not found: {calibrated_params_file}")
    
    with open(calibrated_params_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Extract post-processing calibration parameters
    params = {}
    rater_params = data.get("rater_params", {})
    
    # New parameter names (post-processing layer)
    params["linear_bias"] = rater_params.get("linear_bias", 0.0)
    params["sentiment_weight"] = rater_params.get("sentiment_weight", 0.2)
    params["user_bias_weight"] = rater_params.get("user_bias_weight", 0.3)
    params["item_bias_weight"] = rater_params.get("item_bias_weight", 0.3)
    params["uncertainty_scale"] = rater_params.get("uncertainty_scale", 0.0)
    
    # Backward compatibility: if old parameter names exist, convert them
    if "mapping_slope" in rater_params and "linear_bias" not in rater_params:
        print("Warning: Found old parameter format (mapping_slope/mapping_intercept). "
              "Converting to new format (linear_bias/sentiment_weight).")
        # Approximate conversion (old params were for sentiment-based mapping)
        params["linear_bias"] = rater_params.get("mapping_intercept", 0.0)
        # mapping_slope was used differently, set sentiment_weight to a reasonable default
        params["sentiment_weight"] = 0.2
        params["user_bias_weight"] = rater_params.get("user_bias_weight", 0.3)
        params["item_bias_weight"] = rater_params.get("item_bias_weight", 0.3)
        params["uncertainty_scale"] = rater_params.get("uncertainty_scale", 0.0)
    
    return params


def save_test_results(
    test_metrics: Dict[str, float],
    checkpoint_path: str,
    test_count: int,
    timestamp: str
) -> None:
    """
    Save star rating test results to test_result.txt in checkpoint folder.
    
    Args:
        test_metrics: Test evaluation metrics (star rating metrics only)
        checkpoint_path: Path to checkpoint folder
        test_count: Number of test records evaluated
        timestamp: Timestamp string for the test run
    """
    test_result_file = os.path.join(checkpoint_path, "test_result.txt")
    
    with open(test_result_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("STAR RATING PREDICTION - TEST SET EVALUATION RESULTS\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Test run timestamp: {timestamp}\n")
        f.write(f"Number of test records: {test_count}\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("STAR RATING METRICS\n")
        f.write("-" * 80 + "\n\n")
        
        # Write star rating metrics only
        f.write(f"MAE_stars:  {test_metrics.get('MAE_stars', 0.0):.6f}\n")
        f.write(f"RMSE_stars: {test_metrics.get('RMSE_stars', 0.0):.6f}\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("METRIC DESCRIPTIONS\n")
        f.write("-" * 80 + "\n\n")
        f.write("MAE_stars:  Mean Absolute Error of star rating predictions (1-5)\n")
        f.write("RMSE_stars: Root Mean Squared Error of star rating predictions\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("DETAILED JSON OUTPUT\n")
        f.write("-" * 80 + "\n\n")
        f.write(json.dumps(test_metrics, indent=2))
        f.write("\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("END OF TEST RESULTS\n")
        f.write("=" * 80 + "\n")
    
    print(f"Test results saved to: {test_result_file}")


def main() -> None:
    args = parse_cli()
    set_global_seed(args.seed)

    # Load global LLM cache from file (persistent across runs)
    load_llm_cache()

    if args.offline == 1:
        os.environ["OFFLINE_MODE"] = "1"
    else:
        os.environ["OFFLINE_MODE"] = "0"

    # Enforce LLM availability unless offline
    if args.use_llm == 1 and not is_offline_mode():
        try:
            _ = get_openai_api_key()
            if OpenAI is None:
                raise RuntimeError("OpenAI SDK not available.")
        except Exception as e:
            raise RuntimeError(f"LLM required but not available: {e}. Set OFFLINE_MODE=1 or use --offline=1 to run offline.") from e

    data_bundle = load_data(args)
    data_indexer, persona, item_profiler, planner, author, rater, qa, evaluator, simulator = build_network_and_agents(data_bundle, seed=args.seed)
    simulator.max_output_tokens = args.max_output_tokens
    simulator.model_name = args.model_name
    simulator.llm_retries = args.llm_retries

    # Split train dataset into train (80%) for calibration and valid (20%) for validation
    train_calibration_idxs, valid_validation_idxs, test_idxs = holdout_split(data_indexer, seed=args.seed)

    # Test mode: skip calibration, load params from checkpoint, and run on test set
    if args.test:
        if not args.checkpoint:
            raise ValueError("--checkpoint must be provided when using --test mode")
        
        # Build checkpoint path: DATA_PATH/checkpoint
        checkpoint_path = os.path.join(DATA_DIR, args.checkpoint)
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint folder not found: {checkpoint_path}")
        
        print(f"Test mode: Loading calibrated parameters from {checkpoint_path}")
        
        # Load calibrated parameters from checkpoint
        best_params = load_calibrated_params(checkpoint_path)
        print("Loaded calibrated parameters:")
        for key, value in best_params.items():
            if key != "objective_weights":
                print(f"  {key}: {value}")
        
        # Reset simulator and apply loaded parameters
        simulator.reset_agents()
        simulator.apply_params(best_params)
        
        use_llm_test = (args.use_llm == 1) and not is_offline_mode()
        # In test mode, use full test set (ignore max_validation_records limitation)
        max_test = None
        
        print(f"\nRunning test evaluation on full test set ({len(test_idxs)} records)...")
        
        # Run simulation on test set
        test_metrics_overall, test_preds = simulator.rollout(
            idxs=test_idxs,
            use_llm=use_llm_test,
            max_records=max_test,  # None means use all test_idxs
            collect_traces=False,  # Don't collect traces for test to save space
            model_name=args.model_name,
            max_output_tokens=args.max_output_tokens
        )
        
        test_eval_metrics = {"overall": test_metrics_overall, "by_segment": simulator.evaluator.by_segment_metrics}
        
        # Save test results
        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        test_count = len(test_preds) if test_preds else len(test_idxs)
        save_test_results(
            test_metrics=test_eval_metrics["overall"],
            checkpoint_path=checkpoint_path,
            test_count=test_count,
            timestamp=timestamp
        )
        
        # Save global LLM cache to file (persistent across runs)
        save_llm_cache()
        
        print("\n" + "=" * 80)
        print("TEST EVALUATION COMPLETE")
        print("=" * 80)
        print(f"Test records evaluated: {test_count}")
        print("Test metrics (overall):", json.dumps(test_eval_metrics.get("overall", {}), indent=2))
        print(f"Test results saved to: {os.path.join(checkpoint_path, 'test_result.txt')}")
        return

    # Normal calibration mode leveraging pluggable calibrators
    # Uncomment the desired calibrator configuration:
    
    # BoCalibrator (BoTorch Bayesian Optimization + TuRBO) - DEFAULT
    calibrator_config_repr = (
        f'calibrator = get_calibrator("bo", '
        f'n_trials={args.num_trials}, acquisition_function="ei", use_turbo=True)'
    )
    calibrator = get_calibrator(
        "bo",
        n_trials=args.num_trials,
        acquisition_function="ei",
        use_turbo=True,
    )
    
    # # RandomSearchCalibrator (Random Search)
    # calibrator_config_repr = (
    #     f'calibrator = get_calibrator("random_search", '
    #     f"num_trials={args.num_trials}, early_stop_patience={args.early_stop_patience})"
    # )
    # calibrator = get_calibrator("random_search", num_trials=args.num_trials, early_stop_patience=args.early_stop_patience)
    
    # # EvoCalibrator (EvoTorch Genetic Algorithm)
    # evo_population = max(32, args.num_trials * 4)
    # calibrator_config_repr = (
    #     f'calibrator = get_calibrator("evo", '
    #     f"n_generations={args.num_trials}, population_size={evo_population}, "
    #     "mutation_sigma=0.05, crossover_eta=8.0, tournament_size=3)"
    # )
    # calibrator = get_calibrator(
    #     "evo",
    #     n_generations=args.num_trials,
    #     population_size=evo_population,
    #     mutation_sigma=0.05,
    #     crossover_eta=8.0,
    #     tournament_size=3,
    # )
    
    # LogitHeadCalibrator (Logistic Regression - Not Implemented)
    # calibrator_config_repr = 'calibrator = get_calibrator("logit_head")'
    # calibrator = get_calibrator("logit_head")
    
    # SBICalibrator (Simulation-Based Inference - Not Implemented)
    # calibrator_config_repr = 'calibrator = get_calibrator("sbi")'
    # calibrator = get_calibrator("sbi")
    
    config_log_message = f"Selected calibrator configuration:\n  {calibrator_config_repr}"
    if hasattr(calibrator, "_log"):
        calibrator._log(config_log_message)
    else:
        print(config_log_message)

    # Use LLM during calibration unless offline
    # Use train_calibration_idxs (80% of original train set) for calibration
    use_llm_training = (args.use_llm == 1) and not is_offline_mode()
    fitted_params = calibrator.fit(
        data=data_indexer,
        train_idxs=train_calibration_idxs,
        simulator=simulator,
        rng=random.Random(args.seed),
        use_llm=use_llm_training,
    )
    best_params = fitted_params.param_values

    # Reset to clean training-only state and apply best params before validation
    simulator.reset_agents()
    simulator.apply_params(best_params)

    use_llm_validation = (args.use_llm == 1) and not is_offline_mode()

    max_val = max(0, args.max_validation_records)

    # Create output folder for calibration results (with timestamp)
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder_name = f"outputs_calibration_{timestamp}"
    calibration_output_folder = os.path.join(DATA_DIR, output_folder_name)
    os.makedirs(calibration_output_folder, exist_ok=True)

    # Set traces file path to output folder
    traces_file_path = os.path.join(calibration_output_folder, "simulation_traces.jsonl")

    # Use valid_validation_idxs (20% of original train set) for validation
    metrics_overall, preds = simulator.rollout(
        idxs=valid_validation_idxs,
        use_llm=use_llm_validation,
        max_records=max_val if max_val > 0 else None,
        collect_traces=True,
        model_name=args.model_name,
        max_output_tokens=args.max_output_tokens,
        traces_file_path=traces_file_path
    )

    eval_metrics = {"overall": metrics_overall, "by_segment": simulator.evaluator.by_segment_metrics}

    ablation_report = None
    if args.ablation == 1:
        # Use valid_validation_idxs (20% of original train set) for ablation study
        ablation_report = run_ablation(
            simulator, valid_validation_idxs, best_params, use_llm=use_llm_validation,
            model_name=args.model_name, max_output_tokens=args.max_output_tokens,
            max_records=max_val if max_val > 0 else None
        )

    # Save results to output folder (including evaluation_metrics.json and ablation_report.json)
    save_results(best_params, eval_metrics, ablation=ablation_report, output_folder=calibration_output_folder)

    # Save calibration results to persistent folder in DATA_PATH
    save_calibration_results(
        fitted_params=fitted_params,
        calibrator=calibrator,
        args=args,
        data_dir=DATA_DIR,
        train_calibration_idxs=train_calibration_idxs,
        valid_validation_idxs=valid_validation_idxs,
        eval_metrics=eval_metrics,
        output_folder=calibration_output_folder
    )

    # Save global LLM cache to file (persistent across runs)
    save_llm_cache()

    print("Calibration complete. Best objective:", calibrator.best_objective)
    print("Best parameters saved to:", CALIBRATED_PARAMS_FILE)
    print("Calibration results (persistent) saved to:", calibration_output_folder)
    print(f"  - Config: {os.path.join(calibration_output_folder, 'config.json')}")
    print(f"  - Calibrated parameters: {os.path.join(calibration_output_folder, 'calibrated_parameters.json')}")
    print(f"  - Calibration log: {os.path.join(calibration_output_folder, 'calibration_log.txt')}")
    print("Validation metrics (overall):", json.dumps(eval_metrics.get("overall", {}), indent=2))
    print(f"Simulation traces written to: {os.path.join(calibration_output_folder, 'simulation_traces.jsonl')}")
    print(f"Evaluation metrics saved to: {os.path.join(calibration_output_folder, 'evaluation_metrics.json')}")
    if ablation_report is not None:
        print(f"Ablation report saved to: {os.path.join(calibration_output_folder, 'ablation_report.json')}")



# Execute main for both direct execution and sandbox wrapper invocation
main()