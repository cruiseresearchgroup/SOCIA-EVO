#!/usr/bin/env python3
"""
simulate.py

Production-grade, end-to-end executable multi-agent simulator for e-commerce review generation
and star rating prediction, with automatic calibration and evaluation on a holdout split.

Key features:
- Data ingestion from environment-configured absolute paths (PROJECT_ROOT/DATA_PATH).
- Deterministic behavior via global random seed.
- Multi-role agent pipeline: DataIndexer → PersonaProfiler → ItemProfiler → PlanComposer
  → ReviewAuthor → StarRater → QAConsistency → Evaluator.
- Information propagation via peer influence (optional) and broadcast item reputation priors.
- Exogenous signals: platform policy, item reputation, user leniency, and domain tags.
- Temporal/random holdout splitting with training/validation sets.
- Parameter calibration via seeded random search on the training split.
- Forward simulation on validation split; per-record traces saved to JSONL.
- Metrics: RMSE/MAE (stars), Text similarity, Sentiment agreement, Aspect coverage,
  Consistency score, and Length deviation.
- OpenAI LLM integration using the Responses API to generate review text and to rate stars.

Run:
  python simulate.py --seed 42 --num-trials 20 --max-records 200

Environment variables:
  - PROJECT_ROOT: Absolute path to project root directory.
  - DATA_PATH: Path relative to project root for data directory.
  - OPENAI_API_KEY: Required (unless OFFLINE_MODE=1) to call OpenAI Responses API.
  - OPENAI_MODEL: Optional, default "gpt-4.1-mini".
  - OFFLINE_MODE: If "1", uses deterministic offline stubs for text and rating.

Outputs (written under DATA_DIR):
  - calibrated_parameters.json
  - simulation_traces.jsonl
  - evaluation_metrics.json
  - ablation_report.json
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
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Path handling per instructions
PROJECT_ROOT = os.environ.get("PROJECT_ROOT") or os.getcwd()
DATA_PATH = os.environ.get("DATA_PATH") or "data"
DATA_DIR = os.path.join(PROJECT_ROOT, DATA_PATH)

# Optional OpenAI import with guard
try:
    from openai import OpenAI  # type: ignore
except Exception:
    OpenAI = None  # Will be validated if LLM is requested


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
            return [data]
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
    profile_map: Dict[str, Dict[str, Any]] = field(init=False, default_factory=dict)
    global_mean: float = field(init=False, default=3.0)
    user_history_texts: Dict[str, List[str]] = field(init=False, default_factory=dict)
    item_history_texts: Dict[str, List[str]] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        self._build_indices()
        self._build_profiles()
        self._build_item_tags()
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
        # Use training set texts for few-shot hints
        train_idxs = self.split_index.get("train", [])
        for i in train_idxs:
            rec = self.interactions[i]
            uid, iid = rec["user_id"], rec["item_id"]
            txt = str(rec.get("review", "")).strip()
            if txt:
                if len(self.user_history_texts[uid]) < 3:
                    self.user_history_texts[uid].append(txt[:240])
                if len(self.item_history_texts[iid]) < 3:
                    self.item_history_texts[iid].append(txt[:240])
        # supplement with history_reviews if available
        for obj in self.history_reviews:
            uid = str(obj.get("user_id", "")).strip()
            iid = str(obj.get("item_id", "")).strip()
            txt = str(obj.get("review", obj.get("text", ""))).strip()
            if uid and txt and len(self.user_history_texts[uid]) < 3:
                self.user_history_texts[uid].append(txt[:240])
            if iid and txt and len(self.item_history_texts[iid]) < 3:
                self.item_history_texts[iid].append(txt[:240])

    def get_user_context(self, user_id: str) -> Dict[str, Any]:
        prior = self.user_priors.get(user_id, {"mean": self.global_mean, "var": 1.0})
        profile = self.profile_map.get(user_id, {})
        return {
            "user_id": user_id,
            "leniency_prior": float(prior["mean"]),
            "profile": profile,
            "history_texts": self.user_history_texts.get(user_id, [])
        }

    def get_item_context(self, item_id: str) -> Dict[str, Any]:
        prior = self.item_priors.get(item_id, {"mean": self.global_mean, "var": 1.0})
        tags = self.item_tags.get(item_id, [])
        return {
            "item_id": item_id,
            "reputation_prior": float(prior["mean"]),
            "tags": tags,
            "history_texts": self.item_history_texts.get(item_id, [])
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
        item_desc = (
            f"- Item reputation prior: {item_ctx.get('reputation_prior', item_ctx.get('quality_prior', 3.0)):.2f}\n"
            f"- Item tags: {', '.join(item_ctx.get('tags', item_ctx.get('domain_tags', [])))}\n"
        )
        plan_desc = (
            f"- Planned aspects: {', '.join(plan.get('planned_aspects', []))}\n"
            f"- Tone target: {plan.get('tone_target', 'neutral')}\n"
            f"- Target sentence count: {plan.get('length_target', 3)}\n"
            f"- Platform policy weight (0-1): {plan.get('platform_policy_weight', 0.5):.2f}\n"
        )
        few_user = user_ctx.get("history_texts", [])[:1]
        few_item = item_ctx.get("history_texts", [])[:1]
        few_shot = ""
        if few_user:
            few_shot += f"\nExample from this user:\n---\n{few_user[0]}\n---\n"
        if few_item:
            few_shot += f"\nExample about this item/category:\n---\n{few_item[0]}\n---\n"
        guidelines = self.generation_guidelines
        prompt = (
            "You are a helpful assistant that writes product reviews for an e-commerce platform.\n"
            "Follow the plan, align with the user's persona, and respect platform policies.\n\n"
            "User persona:\n"
            f"{persona_desc}\n"
            "Item context:\n"
            f"{item_desc}\n"
            "Plan:\n"
            f"{plan_desc}\n"
            f"Guidelines: {guidelines}\n"
            f"{few_shot}\n"
            "Task:\n"
            "Write a natural, coherent review that covers the planned aspects in separate sentences, "
            "matches the tone target, and approximately meets the target sentence count. Avoid profanity. "
            "Do not include star ratings in the text.\n"
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

        cache_key = (
            str(user_ctx.get("user_id", "")),
            str(item_ctx.get("item_id", "")),
            tuple(sorted(aspects)),
            tone,
            length,
            int(self.llm_temperature * 100),
        )
        if cache_key in self.cache:
            text = self.cache[cache_key]
        else:
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
            self.cache[cache_key] = text

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
    mapping_slope: float = 4.0
    mapping_intercept: float = 0.0
    user_bias_weight: float = 0.5
    item_bias_weight: float = 0.5
    uncertainty_scale: float = 0.3
    global_mean: float = 3.0  # updated during fit

    def fit_global(self, data: DataIndexer, train_idxs: List[int]) -> None:
        pairs: List[Tuple[float, float]] = []
        for i in train_idxs:
            r = data.interactions[i]
            s = sentiment_score(r["review"])
            y = int(r["stars"])
            yc = (y - 3) / 1.5
            pairs.append((s, yc))

        if not pairs:
            self.global_mean = data.global_mean
            return

        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]
        xmean = statistics.mean(xs)
        ymean = statistics.mean(ys)
        cov = sum((x - xmean) * (y - ymean) for x, y in zip(xs, ys))
        varx = sum((x - xmean) ** 2 for x in xs) or 1e-6
        slope_lin = cov / varx
        intercept_lin = ymean - slope_lin * xmean
        # Preserve sign of slope; clip to symmetric bounds
        self.mapping_slope = max(-8.0, min(8.0, 4.0 * slope_lin))
        self.mapping_intercept = max(-2.0, min(2.0, intercept_lin))
        self.global_mean = data.global_mean

    def predict(self, text: str, user_prior: float, item_prior: float, rng: random.Random) -> int:
        s = sentiment_score(text)
        bias = self.user_bias_weight * (user_prior - self.global_mean) / 2.0 + self.item_bias_weight * (item_prior - self.global_mean) / 2.0
        z = self.mapping_slope * (s + bias) + self.mapping_intercept
        prob = 1.0 / (1.0 + math.exp(-z))
        mean_star = 1.0 + 4.0 * prob
        noisy = mean_star + rng.gauss(0, self.uncertainty_scale)
        star = int(round(noisy))
        return min(5, max(1, star))

    def predict_via_llm(self, review_text: str, user_ctx: Dict[str, Any], item_ctx: Dict[str, Any],
                         model_name: str, max_output_tokens: int = 100, llm_retries: int = 2) -> int:
        if is_offline_mode():
            # fall back to heuristic
            return self.predict(review_text, user_prior=user_ctx.get("baseline_leniency", self.global_mean),
                                item_prior=item_ctx.get("quality_prior", item_ctx.get("reputation_prior", self.global_mean)),
                                rng=random.Random(GLOBAL_SEED))
        prompt = (
            "You are a rating assistant. Given the following product review and context, "
            "assign an overall star rating from 1 to 5 as an integer. Consider user leniency and item reputation if helpful. "
            "Respond with only the integer 1, 2, 3, 4, or 5.\n\n"
            f"User leniency prior: {user_ctx.get('baseline_leniency', self.global_mean):.2f}\n"
            f"Item reputation prior: {item_ctx.get('quality_prior', item_ctx.get('reputation_prior', self.global_mean)):.2f}\n"
            f"Item tags: {', '.join(item_ctx.get('tags', item_ctx.get('domain_tags', [])))}\n\n"
            "Review:\n---\n"
            f"{review_text}\n---\n"
            "Answer:"
        )
        text = call_gpt5_with_responses_api(
            prompt=prompt,
            model=model_name,
            max_output_tokens=max_output_tokens,
            temperature=0.2,
            retries=llm_retries
        ).strip()
        # Extract standalone integer 1-5 with regex; fallback to heuristic
        parsed = None
        m = re.search(r'(?<!\d)([1-5])(?!\d)', text)
        if m:
            try:
                parsed = int(m.group(1))
            except Exception:
                parsed = None
        if parsed is None:
            parsed = self.predict(review_text, user_prior=user_ctx.get("baseline_leniency", self.global_mean),
                                  item_prior=item_ctx.get("quality_prior", item_ctx.get("reputation_prior", self.global_mean)),
                                  rng=random.Random(GLOBAL_SEED))
        return parsed


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
class ParameterTuner:
    num_trials: int = 20
    early_stop_patience: int = 5
    objective_weights: Dict[str, float] = field(default_factory=lambda: {"stars": 0.7, "text": 0.3, "consistency": 0.0})
    history_of_trials: List[Dict[str, Any]] = field(default_factory=list)
    best_params: Optional[Dict[str, Any]] = None
    best_objective: float = float("inf")

    def propose(self, rng: random.Random) -> Dict[str, Any]:
        def uniform(a: float, b: float) -> float:
            return a + (b - a) * rng.random()

        def randint(a: int, b: int) -> int:
            return rng.randint(a, b)

        params = {
            "neighbor_weight": uniform(0.0, 0.5),
            "ctx_merge_weight": uniform(0.2, 0.8),
            "aspect_topk": randint(3, 6),
            "length_target_mean": randint(2, 6),
            "plan_diversity_temp": uniform(0.3, 1.2),
            "llm_temperature": uniform(0.2, 0.9),
            "style_alignment_weight": uniform(0.3, 1.0),
            "mapping_slope": uniform(-8.0, 8.0),
            "mapping_intercept": uniform(-2.0, 2.0),
            "user_bias_weight": uniform(0.0, 1.0),
            "item_bias_weight": uniform(0.0, 1.0),
            "uncertainty_scale": uniform(0.1, 1.0),
            "consistency_threshold": uniform(0.6, 0.9),
            "max_auto_fix_attempts": randint(0, 2),
            "objective_weights": {
                "stars": uniform(0.4, 0.8),
                "text": uniform(0.2, 0.6),
                "consistency": uniform(0.0, 0.3)
            }
        }
        s = sum(params["objective_weights"].values())
        for k in list(params["objective_weights"].keys()):
            params["objective_weights"][k] /= s
        return params

    def fit(self, data: DataIndexer, train_idxs: List[int], simulator: "Simulator", rng: random.Random, use_llm: bool = True) -> Dict[str, Any]:
        no_improve_rounds = 0
        for t in range(self.num_trials):
            # Reset simulator to pristine state (training-only priors) for a fair trial
            simulator.reset_agents()
            params = self.propose(rng)
            self.objective_weights = params["objective_weights"]

            simulator.apply_params(params)
            metrics, _ = simulator.rollout(idxs=train_idxs, use_llm=use_llm, max_records=len(train_idxs), collect_traces=False)
            rmse = metrics.get("RMSE_stars", 1.0)
            text_sim = metrics.get("Text_Similarity", 0.0)
            consistency = metrics.get("Consistency_Score", 0.0)
            obj = self.objective_weights["stars"] * rmse + self.objective_weights["text"] * (1 - text_sim) + self.objective_weights["consistency"] * (1 - consistency)

            self.history_of_trials.append({"trial": t, "params": params, "metrics": metrics, "objective": obj})
            if obj < self.best_objective:
                self.best_objective = obj
                self.best_params = params
                no_improve_rounds = 0
            else:
                no_improve_rounds += 1
                if no_improve_rounds >= self.early_stop_patience:
                    break

        if self.best_params is None:
            self.best_params = self.propose(rng)
        return self.best_params


@dataclass
class Evaluator:
    metric_definitions: List[str] = field(default_factory=lambda: [
        "RMSE_stars", "MAE_stars", "Text_Similarity", "Sentiment_Agreement", "Aspect_Coverage", "Consistency_Score", "Length_Deviation"
    ])
    last_eval_metrics: Dict[str, float] = field(default_factory=dict)
    by_segment_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def compute_overall_metrics(self, records: List[Dict[str, Any]], preds: List[Dict[str, Any]]) -> Dict[str, float]:
        star_err2 = 0.0
        star_abs = 0.0
        text_sims: List[float] = []
        sentiment_agree: List[float] = []
        aspect_coverage: List[float] = []
        consistency_scores: List[float] = []
        length_dev: List[float] = []

        for gt, pr in zip(records, preds):
            y = int(gt["stars"])
            yhat = int(pr["stars"])
            star_err2 += (y - yhat) ** 2
            star_abs += abs(y - yhat)

            ref_vec = hashed_ngram_vector(gt.get("review", ""))
            hyp_vec = hashed_ngram_vector(pr.get("review", ""))
            text_sim = cosine_similarity(ref_vec, hyp_vec)
            text_sims.append(text_sim)

            s = sentiment_score(pr.get("review", ""))
            star_polarity = 1 if yhat >= 4 else (-1 if yhat <= 2 else 0)
            sent_polarity = 1 if s > 0.15 else (-1 if s < -0.15 else 0)
            sentiment_agree.append(1.0 if star_polarity == sent_polarity else 0.0)

            planned = set(a.lower() for a in pr.get("planned_aspects", []))
            mentioned = detect_aspects_in_tokens(normalized_tokens(pr.get("review", "")))
            coverage = len(planned & mentioned) / max(1, len(planned))
            aspect_coverage.append(coverage)

            consistency_scores.append(float(pr.get("consistency_score", 0.0)))

            tgt = int(pr.get("length_target", 3))
            length_dev.append(abs(len(sentence_split(pr.get("review", ""))) - tgt))

        n = len(records) or 1
        metrics = {
            "RMSE_stars": math.sqrt(star_err2 / n),
            "MAE_stars": star_abs / n,
            "Text_Similarity": sum(text_sims) / n if text_sims else 0.0,
            "Sentiment_Agreement": sum(sentiment_agree) / n if sentiment_agree else 0.0,
            "Aspect_Coverage": sum(aspect_coverage) / n if aspect_coverage else 0.0,
            "Consistency_Score": sum(consistency_scores) / n if consistency_scores else 0.0,
            "Length_Deviation": sum(length_dev) / n if length_dev else 0.0,
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
        self.persona.neighbor_weight = float(params.get("neighbor_weight", self.persona.neighbor_weight))
        self.planner.ctx_merge_weight = float(params.get("ctx_merge_weight", self.planner.ctx_merge_weight))
        self.planner.aspect_topk = int(params.get("aspect_topk", self.planner.aspect_topk))
        self.planner.length_target_mean = int(params.get("length_target_mean", self.planner.length_target_mean))
        self.planner.plan_diversity_temp = float(params.get("plan_diversity_temp", self.planner.plan_diversity_temp))
        self.author.llm_temperature = float(params.get("llm_temperature", self.author.llm_temperature))
        self.author.style_alignment_weight = float(params.get("style_alignment_weight", self.author.style_alignment_weight))
        self.rater.mapping_slope = float(params.get("mapping_slope", self.rater.mapping_slope))
        self.rater.mapping_intercept = float(params.get("mapping_intercept", self.rater.mapping_intercept))
        self.rater.user_bias_weight = float(params.get("user_bias_weight", self.rater.user_bias_weight))
        self.rater.item_bias_weight = float(params.get("item_bias_weight", self.rater.item_bias_weight))
        self.rater.uncertainty_scale = float(params.get("uncertainty_scale", self.rater.uncertainty_scale))
        self.qa.consistency_threshold = float(params.get("consistency_threshold", self.qa.consistency_threshold))
        self.qa.max_auto_fix_attempts = int(params.get("max_auto_fix_attempts", self.qa.max_auto_fix_attempts))

    def reset_agents(self) -> None:
        # Reset RNG and agent internal mutable state; rebuild persona/item priors from training data only
        self.rng = random.Random(self.base_seed)
        self.persona.construct(self.data)
        self.item_profiler.construct(self.data)
        self.author.cache.clear()
        train_idxs = self.data.split_index.get("train", [])
        self.rater.fit_global(self.data, train_idxs)

    def rollout(self, idxs: Optional[List[int]] = None, use_llm: bool = True, max_records: Optional[int] = None, collect_traces: bool = True,
                model_name: Optional[str] = None, max_output_tokens: Optional[int] = None) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
        if idxs is None:
            idxs = list(range(len(self.data.interactions)))
        if max_records is not None:
            idxs = idxs[:max_records]

        preds: List[Dict[str, Any]] = []
        records: List[Dict[str, Any]] = []
        model = model_name or self.model_name
        mot = max_output_tokens if max_output_tokens is not None else self.max_output_tokens

        if collect_traces:
            # Ensure we always close file even on exceptions
            with open(SIM_TRACES_FILE, "w", encoding="utf-8") as traces_fp:
                for i in idxs:
                    rec = self.data.interactions[i]
                    user_id = rec["user_id"]
                    item_id = rec["item_id"]

                    user_ctx_full = self.persona.get_persona(user_id, self.data)
                    item_ctx_full = self.item_profiler.item_state.get(item_id)
                    if not item_ctx_full:
                        item_ctx_full = self.data.get_item_context(item_id)
                        item_ctx_full["aspect_summary"] = {a: 1.0 / len(DEFAULT_ASPECT_VOCAB) for a in DEFAULT_ASPECT_VOCAB}

                    plan = self.planner.compose(user_ctx_full, item_ctx_full, self.platform_policy_weight)

                    review_text = self.author.generate(
                        user_ctx=user_ctx_full, item_ctx=item_ctx_full, plan=plan,
                        use_llm=use_llm and not is_offline_mode(), model_name=model, max_output_tokens=mot, llm_retries=self.llm_retries
                    )

                    user_prior = user_ctx_full.get("baseline_leniency", self.data.global_mean)
                    item_prior = item_ctx_full.get("quality_prior", item_ctx_full.get("reputation_prior", self.data.global_mean))
                    if use_llm and not is_offline_mode():
                        stars = self.rater.predict_via_llm(review_text, user_ctx=user_ctx_full, item_ctx=item_ctx_full,
                                                           model_name=model, max_output_tokens=64, llm_retries=self.llm_retries)
                    else:
                        stars = self.rater.predict(review_text, user_prior=user_prior, item_prior=item_prior, rng=self.rng)

                    final_text, final_stars, consistency_score_val, revision_count = self.qa.maybe_revise(
                        author=self.author, user_ctx=user_ctx_full, item_ctx=item_ctx_full, plan=plan,
                        proposed_stars=stars, text=review_text, use_llm=use_llm and not is_offline_mode(),
                        model_name=model, max_output_tokens=mot, llm_retries=self.llm_retries
                    )

                    self.persona.update_from_recent(user_id, final_text, final_stars)
                    self.item_profiler.update_from_observation(item_id, final_text, final_stars)

                    pred = {
                        "user_id": user_id,
                        "item_id": item_id,
                        "stars": final_stars,
                        "review": final_text,
                        "planned_aspects": plan.get("planned_aspects", []),
                        "length_target": plan.get("length_target", 3),
                        "consistency_score": consistency_score_val,
                        "revision_count": revision_count,
                        "tone_target": plan.get("tone_target", "neutral"),
                    }
                    preds.append(pred)
                    records.append(rec)

                    trace_obj = {
                        "record_index": i,
                        "inputs": {
                            "user_id": user_id,
                            "item_id": item_id
                        },
                        "plan": plan,
                        "generated_text": final_text,
                        "predicted_stars": final_stars,
                        "diagnostics": {
                            "consistency_score": consistency_score_val,
                            "revision_count": revision_count
                        }
                    }
                    traces_fp.write(json.dumps(trace_obj) + "\n")
        else:
            for i in idxs:
                rec = self.data.interactions[i]
                user_id = rec["user_id"]
                item_id = rec["item_id"]

                user_ctx_full = self.persona.get_persona(user_id, self.data)
                item_ctx_full = self.item_profiler.item_state.get(item_id)
                if not item_ctx_full:
                    item_ctx_full = self.data.get_item_context(item_id)
                    item_ctx_full["aspect_summary"] = {a: 1.0 / len(DEFAULT_ASPECT_VOCAB) for a in DEFAULT_ASPECT_VOCAB}

                plan = self.planner.compose(user_ctx_full, item_ctx_full, self.platform_policy_weight)

                review_text = self.author.generate(
                    user_ctx=user_ctx_full, item_ctx=item_ctx_full, plan=plan,
                    use_llm=use_llm and not is_offline_mode(), model_name=model, max_output_tokens=mot, llm_retries=self.llm_retries
                )
                user_prior = user_ctx_full.get("baseline_leniency", self.data.global_mean)
                item_prior = item_ctx_full.get("quality_prior", item_ctx_full.get("reputation_prior", self.data.global_mean))
                if use_llm and not is_offline_mode():
                    stars = self.rater.predict_via_llm(review_text, user_ctx=user_ctx_full, item_ctx=item_ctx_full,
                                                       model_name=model, max_output_tokens=64, llm_retries=self.llm_retries)
                else:
                    stars = self.rater.predict(review_text, user_prior=user_prior, item_prior=item_prior, rng=self.rng)

                final_text, final_stars, consistency_score_val, revision_count = self.qa.maybe_revise(
                    author=self.author, user_ctx=user_ctx_full, item_ctx=item_ctx_full, plan=plan,
                    proposed_stars=stars, text=review_text, use_llm=use_llm and not is_offline_mode(),
                    model_name=model, max_output_tokens=mot, llm_retries=self.llm_retries
                )

                self.persona.update_from_recent(user_id, final_text, final_stars)
                self.item_profiler.update_from_observation(item_id, final_text, final_stars)

                pred = {
                    "user_id": user_id,
                    "item_id": item_id,
                    "stars": final_stars,
                    "review": final_text,
                    "planned_aspects": plan.get("planned_aspects", []),
                    "length_target": plan.get("length_target", 3),
                    "consistency_score": consistency_score_val,
                    "revision_count": revision_count,
                    "tone_target": plan.get("tone_target", "neutral"),
                }
                preds.append(pred)
                records.append(rec)

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


def holdout_split(data_indexer: DataIndexer) -> Tuple[List[int], List[int]]:
    train_idxs = data_indexer.split_index.get("train", [])
    test_idxs = data_indexer.split_index.get("test", [])
    if not train_idxs or not test_idxs:
        all_idxs = list(range(len(data_indexer.interactions)))
        random.shuffle(all_idxs)
        cut = int(0.8 * len(all_idxs))
        train_idxs, test_idxs = all_idxs[:cut], all_idxs[cut:]
    return train_idxs, test_idxs


def save_results(calibrated_params: Dict[str, Any], eval_metrics: Dict[str, Any], ablation: Optional[Dict[str, Any]] = None) -> None:
    ensure_data_dir()
    with open(CALIBRATED_PARAMS_FILE, "w", encoding="utf-8") as f:
        json.dump(calibrated_params, f, indent=2)
    with open(EVAL_METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(eval_metrics, f, indent=2)
    if ablation is not None:
        with open(ABLATION_REPORT_FILE, "w", encoding="utf-8") as f:
            json.dump(ablation, f, indent=2)


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


def main() -> None:
    args = parse_cli()
    set_global_seed(args.seed)

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

    train_idxs, test_idxs = holdout_split(data_indexer)

    tuner = ParameterTuner(num_trials=args.num_trials, early_stop_patience=args.early_stop_patience)

    # Use LLM during calibration unless offline
    use_llm_training = (args.use_llm == 1) and not is_offline_mode()
    best_params = tuner.fit(data_indexer, train_idxs, simulator, rng=random.Random(args.seed), use_llm=use_llm_training)

    # Reset to clean training-only state and apply best params before validation
    simulator.reset_agents()
    simulator.apply_params(best_params)

    use_llm_validation = (args.use_llm == 1) and not is_offline_mode()

    max_val = max(0, args.max_validation_records)

    metrics_overall, preds = simulator.rollout(
        idxs=test_idxs,
        use_llm=use_llm_validation,
        max_records=max_val if max_val > 0 else None,
        collect_traces=True,
        model_name=args.model_name,
        max_output_tokens=args.max_output_tokens
    )

    eval_metrics = {"overall": metrics_overall, "by_segment": simulator.evaluator.by_segment_metrics}

    ablation_report = None
    if args.ablation == 1:
        ablation_report = run_ablation(
            simulator, test_idxs, best_params, use_llm=use_llm_validation,
            model_name=args.model_name, max_output_tokens=args.max_output_tokens,
            max_records=max_val if max_val > 0 else None
        )

    save_results(best_params, eval_metrics, ablation_report=ablation_report)

    print("Calibration complete. Best objective:", tuner.best_objective)
    print("Best parameters saved to:", CALIBRATED_PARAMS_FILE)
    print("Validation metrics (overall):", json.dumps(eval_metrics.get("overall", {}), indent=2))
    print("Simulation traces written to:", SIM_TRACES_FILE)
    print("Evaluation metrics saved to:", EVAL_METRICS_FILE)
    if ablation_report is not None:
        print("Ablation report saved to:", ABLATION_REPORT_FILE)



# Execute main for both direct execution and sandbox wrapper invocation
main()