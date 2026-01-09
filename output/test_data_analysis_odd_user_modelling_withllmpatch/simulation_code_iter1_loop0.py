#!/usr/bin/env python3
import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import random
import statistics
import sys
import time
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Optional OpenAI import with guard
try:
    from openai import OpenAI  # type: ignore
except Exception:
    OpenAI = None  # Will be validated if LLM is requested

# Path handling per instructions
PROJECT_ROOT = os.environ.get("PROJECT_ROOT") or os.getcwd()
DATA_PATH = os.environ.get("DATA_PATH") or "data"
DATA_DIR = os.path.join(PROJECT_ROOT, DATA_PATH)

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
# JSON alternative inputs
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

# Simple platform policy config
BANNED_WORDS = {"damn", "hell", "crap", "sucks", "idiot", "stupid"}  # extend as needed


def ensure_data_dir() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)


def get_openai_api_key() -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        return api_key
    raise ValueError("OpenAI API key not found in environment. Set OPENAI_API_KEY to use LLM features.")


def call_gpt5_with_responses_api(prompt: str, model: Optional[str] = None, max_output_tokens: int = 4000, temperature: Optional[float] = None) -> str:
    if OpenAI is None:
        raise RuntimeError("OpenAI SDK not available. Install the 'openai' package to enable LLM calls.")
    api_key = get_openai_api_key()
    client = OpenAI(api_key=api_key)
    chosen_model = model or os.environ.get("OPENAI_MODEL", "gpt-5")

    responses_kwargs: Dict[str, Any] = {
        "model": chosen_model,
        "input": [
            {"role": "user", "content": [{"type": "input_text", "text": prompt}]}
        ],
        "max_output_tokens": max_output_tokens,
    }
    if temperature is not None:
        responses_kwargs["temperature"] = temperature

    try:
        resp = client.responses.create(**responses_kwargs)
    except Exception as e:
        raise RuntimeError(f"OpenAI Responses API call failed: {e}") from e

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

    return extract_response(resp)


def parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-agent review simulation with calibration and evaluation.")
    parser.add_argument("--seed", type=int, default=GLOBAL_SEED, help="Global random seed.")
    parser.add_argument("--num-trials", type=int, default=20, help="Number of calibration trials.")
    parser.add_argument("--early-stop-patience", type=int, default=5, help="Early stopping patience for calibration.")
    parser.add_argument("--max-records", type=int, default=200, help="Max number of records to use (subset for speed).")
    parser.add_argument("--max-validation-records", type=int, default=30, help="Max records in validation rollout (LLM calls).")
    parser.add_argument("--use-llm", type=int, default=1, help="If 1, use OpenAI LLM for review generation in validation rollout.")
    parser.add_argument("--use-llm-rating", type=int, default=1, help="If 1, use OpenAI LLM for star rating on validation rollout.")
    parser.add_argument("--model-name", type=str, default=os.environ.get("OPENAI_MODEL", "gpt-5"), help="OpenAI model name (responses API).")
    parser.add_argument("--max-output-tokens", type=int, default=600, help="Max tokens for LLM outputs.")
    parser.add_argument("--ablation", type=int, default=1, help="If 1, run a small ablation study.")
    return parser.parse_args()


def set_global_seed(seed: int) -> None:
    random.seed(seed)


def read_csv_if_exists(path: str) -> Optional[List[Dict[str, str]]]:
    if not os.path.isfile(path):
        return None
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader]
    return rows


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Cannot write empty CSV: no rows provided.")
    fieldnames = sorted(set().union(*(row.keys() for row in rows)))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})


def read_json_list_if_exists(path: str) -> Optional[List[Dict[str, Any]]]:
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    # If dict with "data" or known keys
    if isinstance(data, dict):
        for key in ("data", "records", "items"):
            if key in data and isinstance(data[key], list):
                return data[key]
    return None


def synthesize_dataset(n_users: int = 20, n_items: int = 30, n_interactions: int = 200, seed: int = 42) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    rng = random.Random(seed)
    users = [f"U{u:03d}" for u in range(1, n_users + 1)]
    items = [f"I{i:03d}" for i in range(1, n_items + 1)]
    item_cats = ["electronics", "home", "toys", "books", "kitchen", "garden"]
    interactions: List[Dict[str, Any]] = []
    user_profiles: List[Dict[str, Any]] = []
    item_metadata: List[Dict[str, Any]] = []

    user_leniency = {u: rng.uniform(2.5, 3.5) for u in users}
    item_quality = {it: rng.uniform(2.5, 4.2) for it in items}

    for u in users:
        user_profiles.append({
            "user_id": u,
            "avg_stars": round(user_leniency[u], 2),
            "friends": "",
            "review_count": rng.randint(1, 50),
            "tone_hint": rng.choice(["formal", "casual", "enthusiastic", "concise"])
        })

    for it in items:
        cat = rng.choice(item_cats)
        tags = rng.sample(DEFAULT_ASPECT_VOCAB, k=rng.randint(2, 5))
        item_metadata.append({
            "item_id": it,
            "category": cat,
            "keywords": ";".join(tags)
        })

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
            "datatype": ""
        })

    interactions_sorted = sorted(interactions, key=lambda r: r["timestamp"])
    split_idx = int(0.8 * len(interactions_sorted))
    for i, rec in enumerate(interactions_sorted):
        rec["datatype"] = "train" if i < split_idx else "test"

    return interactions_sorted, user_profiles, item_metadata


def load_json_data_if_available() -> Optional[Dict[str, Any]]:
    train = read_json_list_if_exists(AMAZON_TRAIN_JSON)
    test = read_json_list_if_exists(AMAZON_TEST_JSON)
    profiles = read_json_list_if_exists(USER_SAMPLE_JSON)
    items = read_json_list_if_exists(ITEM_SAMPLE_JSON)
    extra_reviews = read_json_list_if_exists(REVIEW_SAMPLE_JSON)

    if train is None and test is None and profiles is None and items is None:
        return None

    interactions: List[Dict[str, Any]] = []

    def normalize_interaction(rec: Dict[str, Any], split: str) -> Dict[str, Any]:
        user_id = str(rec.get("user_id") or rec.get("userId") or rec.get("user"))
        item_id = str(rec.get("item_id") or rec.get("itemId") or rec.get("asin") or rec.get("item"))
        stars = rec.get("stars") or rec.get("rating") or rec.get("star") or 3
        try:
            stars = int(round(float(stars)))
        except Exception:
            stars = 3
        review = str(rec.get("review") or rec.get("text") or rec.get("content") or "").strip()
        timestamp = rec.get("timestamp") or rec.get("time") or ""
        if isinstance(timestamp, (int, float)):
            try:
                timestamp = dt.date.fromtimestamp(int(timestamp)).isoformat()
            except Exception:
                timestamp = dt.date(2023, 1, 1).isoformat()
        elif not timestamp:
            timestamp = dt.date(2023, 1, 1).isoformat()
        return {
            "user_id": user_id,
            "item_id": item_id,
            "stars": stars,
            "review": review,
            "timestamp": timestamp,
            "datatype": split
        }

    if train:
        interactions.extend([normalize_interaction(r, "train") for r in train])
    if test:
        interactions.extend([normalize_interaction(r, "test") for r in test])

    user_profiles = []
    if profiles:
        for p in profiles:
            user_profiles.append({
                "user_id": str(p.get("user_id") or p.get("userId") or p.get("user")),
                "avg_stars": float(p.get("avg_stars") or p.get("average_stars") or 3.0),
                "friends": str(p.get("friends") or ""),
                "review_count": int(p.get("review_count") or p.get("count") or 0),
                "tone_hint": str(p.get("tone_hint") or p.get("tone") or "neutral")
            })

    item_metadata = []
    if items:
        for it in items:
            tags = it.get("keywords") or it.get("tags") or it.get("aspects")
            if isinstance(tags, list):
                keyw = ";".join([str(t) for t in tags])
            else:
                keyw = str(tags or "")
            item_metadata.append({
                "item_id": str(it.get("item_id") or it.get("itemId") or it.get("asin") or it.get("item")),
                "category": str(it.get("category") or ""),
                "keywords": keyw
            })

    # Extra reviews for few-shot
    extra_user_reviews: Dict[str, List[str]] = defaultdict(list)
    extra_item_reviews: Dict[str, List[str]] = defaultdict(list)
    if extra_reviews:
        for rr in extra_reviews:
            uid = str(rr.get("user_id") or rr.get("userId") or rr.get("user") or "")
            itid = str(rr.get("item_id") or rr.get("itemId") or rr.get("asin") or rr.get("item") or "")
            text = str(rr.get("review") or rr.get("text") or rr.get("content") or "").strip()
            if uid and text:
                extra_user_reviews[uid].append(text)
            if itid and text:
                extra_item_reviews[itid].append(text)

    return {
        "interactions": interactions,
        "user_profiles": user_profiles,
        "item_metadata": item_metadata,
        "extra_user_reviews": extra_user_reviews,
        "extra_item_reviews": extra_item_reviews
    }


def load_data(args: argparse.Namespace) -> Dict[str, Any]:
    ensure_data_dir()

    # Prefer JSON dataset if present
    json_bundle = load_json_data_if_available()
    if json_bundle is not None and json_bundle.get("interactions"):
        interactions = json_bundle["interactions"]
        user_profiles = json_bundle.get("user_profiles") or []
        item_metadata = json_bundle.get("item_metadata") or []
        extra_user_reviews = json_bundle.get("extra_user_reviews", defaultdict(list))
        extra_item_reviews = json_bundle.get("extra_item_reviews", defaultdict(list))
    else:
        interactions = read_csv_if_exists(INTERACTIONS_FILE)
        user_profiles = read_csv_if_exists(USER_PROFILES_FILE)
        item_metadata = read_csv_if_exists(ITEM_METADATA_FILE)
        extra_user_reviews = defaultdict(list)
        extra_item_reviews = defaultdict(list)

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
        raise ValueError(f"Failed to load interactions from {INTERACTIONS_FILE}")

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

    data_bundle = {
        "interactions": interactions,
        "user_profiles": user_profiles,
        "item_metadata": item_metadata,
        "extra_user_reviews": extra_user_reviews,
        "extra_item_reviews": extra_item_reviews
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


def mention_aspect(text: str, aspect: str) -> bool:
    if not text or not aspect:
        return False
    txt = text.lower()
    asp = re.escape(aspect.lower())
    pattern = r"\b" + asp + r"\b"
    return re.search(pattern, txt) is not None


POS_WORDS = {
    "good", "great", "excellent", "amazing", "love", "loved", "like", "liked",
    "awesome", "fantastic", "superb", "satisfied", "happy", "positive", "recommend",
    "durable", "reliable", "value", "fast", "comfortable", "nice", "perfect"
}
NEG_WORDS = {
    "bad", "terrible", "awful", "hate", "hated", "dislike", "disliked", "poor",
    "disappointed", "broken", "slow", "uncomfortable", "worse", "worst", "problem",
    "issue", "buggy", "fragile", "cheap", "expensive", "negative", "not recommend"
}


def sentiment_score(text: str) -> float:
    toks = tokenize(text)
    if not toks:
        return 0.0
    pos = sum(1 for t in toks if t in POS_WORDS)
    neg = sum(1 for t in toks if t in NEG_WORDS)
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
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
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
    interactions: List[Dict[str, Any]]
    user_profiles: List[Dict[str, Any]]
    item_metadata: List[Dict[str, Any]]
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
    user_text_history: Dict[str, List[str]] = field(init=False, default_factory=dict)
    item_text_history: Dict[str, List[str]] = field(init=False, default_factory=dict)

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
            self.profile_map[p["user_id"]] = p

    def _build_item_tags(self) -> None:
        self.item_tags = {}
        for it in self.item_metadata:
            tags = str(it.get("keywords", "")).split(";")
            self.item_tags[it["item_id"]] = [t.strip().lower() for t in tags if t.strip()]
        for item_id in self.item_index.keys():
            if item_id not in self.item_tags:
                self.item_tags[item_id] = random.sample(self.aspect_vocab, k=min(3, len(self.aspect_vocab)))

    def _compute_priors(self) -> None:
        all_stars = [int(r["stars"]) for r in self.interactions]
        self.global_mean = statistics.mean(all_stars) if all_stars else 3.0

        self.user_priors = {}
        for user_id, idxs in self.user_index.items():
            stars = [int(self.interactions[i]["stars"]) for i in idxs]
            mean = statistics.mean(stars) if stars else self.global_mean
            var = statistics.pvariance(stars) if len(stars) > 1 else 1.0
            self.user_priors[user_id] = {"mean": float(mean), "var": float(var)}

        self.item_priors = {}
        for item_id, idxs in self.item_index.items():
            stars = [int(self.interactions[i]["stars"]) for i in idxs]
            mean = statistics.mean(stars) if stars else self.global_mean
            var = statistics.pvariance(stars) if len(stars) > 1 else 1.0
            self.item_priors[item_id] = {"mean": float(mean), "var": float(var)}

    def _build_histories(self) -> None:
        self.user_text_history = defaultdict(list)
        self.item_text_history = defaultdict(list)
        # Collect from interactions
        for rec in self.interactions:
            if rec.get("review"):
                self.user_text_history[rec["user_id"]].append(rec["review"])
                self.item_text_history[rec["item_id"]].append(rec["review"])
        # Incorporate extra samples if present
        extra_user_reviews: Dict[str, List[str]] = self.__dict__.get("extra_user_reviews") or {}
        for uid, texts in extra_user_reviews.items():
            self.user_text_history[uid].extend(texts)
        extra_item_reviews: Dict[str, List[str]] = self.__dict__.get("extra_item_reviews") or {}
        for iid, texts in extra_item_reviews.items():
            self.item_text_history[iid].extend(texts)

    def get_user_context(self, user_id: str) -> Dict[str, Any]:
        prior = self.user_priors.get(user_id, {"mean": self.global_mean, "var": 1.0})
        profile = self.profile_map.get(user_id, {})
        return {
            "user_id": user_id,
            "leniency_prior": float(prior["mean"]),
            "profile": profile
        }

    def get_item_context(self, item_id: str) -> Dict[str, Any]:
        prior = self.item_priors.get(item_id, {"mean": self.global_mean, "var": 1.0})
        tags = self.item_tags.get(item_id, [])
        return {
            "item_id": item_id,
            "reputation_prior": float(prior["mean"]),
            "tags": tags
        }

    def refresh(self) -> None:
        """
        Refresh priors and histories from current interactions in memory.
        Note: does not re-read files; call external loaders for that.
        """
        self._compute_priors()
        self._build_histories()


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
        for uid in data.user_index:
            prior = data.user_priors.get(uid, {"mean": data.global_mean})
            tone_hint = data.profile_map.get(uid, {}).get("tone_hint", "neutral")
            history_idxs = data.user_index.get(uid, [])
            recent_reviews = [data.interactions[i]["review"] for i in history_idxs[-5:]]
            if not recent_reviews:
                recent_reviews = data.user_text_history.get(uid, [])[-3:]
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
                "recent_reviews": recent_reviews[-3:]
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
            for a in DEFAULT_ASPECT_VOCAB:
                if mention_aspect(r, a):
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

        for a in DEFAULT_ASPECT_VOCAB:
            hit = mention_aspect(new_review, a)
            prev = state["aspect_preference_weights"].get(a, 1.0 / len(DEFAULT_ASPECT_VOCAB))
            target = 1.0 if hit else 0.0
            state["aspect_preference_weights"][a] = self.aspect_weight_decay * prev + (1 - self.aspect_weight_decay) * target

        s = sentiment_score(new_review)
        state["recent_sentiment_bias"] = 0.8 * state["recent_sentiment_bias"] + 0.2 * s
        # maintain recent reviews buffer
        rr = state.get("recent_reviews", [])
        rr.append(new_review)
        state["recent_reviews"] = rr[-3:]

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
                "recent_reviews": []
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
        for item_id, idxs in data.item_index.items():
            reviews = [data.interactions[i]["review"] for i in idxs]
            stars = [int(data.interactions[i]["stars"]) for i in idxs]
            aspects = self._extract_aspects(reviews)
            item_mean = statistics.mean(stars) if stars else data.global_mean
            prior_mean = self.reputation_inertia * item_mean + (1 - self.reputation_inertia) * data.global_mean
            var = statistics.pvariance(stars) if len(stars) > 1 else 1.0
            self.item_state[item_id] = {
                "reputation_prior": float(prior_mean),
                "variance": float(var),
                "aspect_summary": aspects,
                "domain_tags": data.item_tags.get(item_id, []),
                "freshness_score": min(1.0, len(stars) / 10.0),
                "controversy": min(1.0, var / 2.0),
                "aspect_confidence": min(1.0, len(reviews) / 10.0),
                "recent_reviews": reviews[-3:]
            }

    def _extract_aspects(self, reviews: List[str]) -> Dict[str, float]:
        counts = Counter()
        for r in reviews:
            for a in DEFAULT_ASPECT_VOCAB:
                if mention_aspect(r, a):
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
                "reputation_prior": 3.0,
                "variance": 1.0,
                "aspect_summary": {a: 1.0 / len(DEFAULT_ASPECT_VOCAB) for a in DEFAULT_ASPECT_VOCAB},
                "domain_tags": [],
                "freshness_score": 0.5,
                "controversy": 0.5,
                "aspect_confidence": 0.1,
                "recent_reviews": []
            }
            self.item_state[item_id] = st
        st["reputation_prior"] = self.reputation_inertia * st["reputation_prior"] + (1 - self.reputation_inertia) * new_stars
        new_aspects = self._extract_aspects([new_review])
        for a in DEFAULT_ASPECT_VOCAB:
            st["aspect_summary"][a] = self.aspect_smoothing_alpha * st["aspect_summary"].get(a, 0.0) + (1 - self.aspect_smoothing_alpha) * new_aspects.get(a, 0.0)
        st["freshness_score"] = min(1.0, 0.9 * st["freshness_score"] + 0.1)
        st["aspect_confidence"] = min(1.0, st["aspect_confidence"] + 0.1)
        rr = st.get("recent_reviews", [])
        rr.append(new_review)
        st["recent_reviews"] = rr[-3:]


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
        length_target = int(max(min_len, min(max_len, round(self.length_target_mean * persona.get("style_vector", [0.5, 0.5, 0.5])[2] * 2))))
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
    cache: Dict[str, str] = field(default_factory=dict)

    def _build_prompt(self, user_ctx: Dict[str, Any], item_ctx: Dict[str, Any], plan: Dict[str, Any]) -> str:
        style_vec = user_ctx.get("style_vector", [0.5, 0.5, 0.5])
        tone_hint = user_ctx.get("tone_style_prior", "neutral")
        persona_desc = (
            f"- Baseline leniency: {user_ctx.get('baseline_leniency', 3.0):.2f}\n"
            f"- Style/tone: {tone_hint}, style_vector(formality,enthusiasm,conciseness)={style_vec}\n"
            f"- Domain familiarity: {user_ctx.get('domain_familiarity', 0.5):.2f}\n"
        )
        item_desc = (
            f"- Item reputation prior: {item_ctx.get('reputation_prior', 3.0):.2f}\n"
            f"- Item tags: {', '.join(item_ctx.get('tags', []))}\n"
        )
        plan_desc = (
            f"- Planned aspects: {', '.join(plan.get('planned_aspects', []))}\n"
            f"- Tone target: {plan.get('tone_target', 'neutral')}\n"
            f"- Target sentence count: {plan.get('length_target', 3)}\n"
        )
        # Few-shot examples
        examples = []
        user_examples = user_ctx.get("recent_reviews", []) if isinstance(user_ctx, dict) else []
        if user_examples:
            examples.append(f"User previous review example: {user_examples[-1]}")
        item_examples = item_ctx.get("recent_reviews", []) if isinstance(item_ctx, dict) else []
        if item_examples:
            examples.append(f"Another example about this item: {item_examples[-1]}")
        examples_block = ("\n".join(examples) + "\n") if examples else ""

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
            f"{('Examples:\n' + examples_block) if examples_block else ''}"
            f"Guidelines: {guidelines}\n\n"
            "Task:\n"
            "Write a natural, coherent review that covers the planned aspects in separate sentences, "
            "matches the tone target, and approximately meets the target sentence count. Avoid profanity. "
            "Do not include star ratings in the text.\n"
        )
        return prompt

    def generate(self, user_ctx: Dict[str, Any], item_ctx: Dict[str, Any], plan: Dict[str, Any],
                 use_llm: bool, model_name: str, max_output_tokens: int) -> str:
        # cache key
        cache_key_obj = {
            "user": user_ctx.get("user_id"),
            "item": item_ctx.get("item_id"),
            "aspects": sorted(plan.get("planned_aspects", [])),
            "tone": plan.get("tone_target"),
            "length": plan.get("length_target"),
        }
        cache_key = json.dumps(cache_key_obj, sort_keys=True)
        if cache_key in self.cache:
            return self.cache[cache_key]

        offline = os.environ.get("OFFLINE_MODE", "0") == "1"
        if not use_llm and not offline:
            raise RuntimeError("LLM generation is required unless OFFLINE_MODE=1 is set.")
        if use_llm and (OpenAI is None or not os.environ.get("OPENAI_API_KEY")):
            if offline:
                use_llm = False
            else:
                raise RuntimeError("LLM requested but unavailable. Set OPENAI_API_KEY and install openai.")

        if use_llm:
            prompt = self._build_prompt(user_ctx, item_ctx, plan)
            raw = call_gpt5_with_responses_api(prompt=prompt, model=model_name, max_output_tokens=max_output_tokens, temperature=self.llm_temperature)
            text = raw.strip()
        else:
            # Deterministic template-based fallback for offline mode only
            aspects = plan.get("planned_aspects", [])
            tone = plan.get("tone_target", "neutral")
            length = plan.get("length_target", 3)
            verb = {"positive": "appreciate", "neutral": "note", "negative": "dislike"}.get(tone, "note")
            sentences = []
            for i, a in enumerate(aspects[:length]):
                modifier = ["clearly", "notably", "generally", "mostly", "somewhat"][i % 5]
                polarity = {"positive": "good", "neutral": "average", "negative": "poor"}.get(tone, "average")
                sentences.append(f"I {verb} the {a}; it is {modifier} {polarity}.")
            while len(sentences) < length:
                sentences.append("Overall, the experience matches expectations.")
            text = " ".join(sentences)

        sentences = sentence_split(text)
        target = plan.get("length_target", 3)
        if len(sentences) > target:
            sentences = sentences[:target]
        elif len(sentences) < target:
            sentences += ["Additionally, it performs as expected."] * (target - len(sentences))
        final_text = " ".join(sentences).strip()
        self.cache[cache_key] = final_text
        return final_text


@dataclass
class StarRater:
    mapping_slope: float = 4.0
    mapping_intercept: float = 0.0
    user_bias_weight: float = 0.5
    item_bias_weight: float = 0.5
    uncertainty_scale: float = 0.3
    global_mean: float = 3.0

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

        self.mapping_slope = max(2.0, min(8.0, 4.0 * abs(slope_lin) + 2.0))
        self.mapping_intercept = max(-2.0, min(2.0, intercept_lin))
        self.global_mean = data.global_mean

    def _heuristic_predict(self, text: str, user_prior: float, item_prior: float, rng: random.Random) -> int:
        s = sentiment_score(text)
        bias = self.user_bias_weight * (user_prior - self.global_mean) / 2.0 + self.item_bias_weight * (item_prior - self.global_mean) / 2.0
        z = self.mapping_slope * (s + bias) + self.mapping_intercept
        prob = 1.0 / (1.0 + math.exp(-z))
        mean_star = 1.0 + 4.0 * prob
        noisy = mean_star + rng.gauss(0, self.uncertainty_scale)
        star = int(round(noisy))
        return min(5, max(1, star))

    def _llm_rate(self, text: str, user_prior: float, item_prior: float, model_name: str) -> int:
        offline = os.environ.get("OFFLINE_MODE", "0") == "1"
        if OpenAI is None or not os.environ.get("OPENAI_API_KEY"):
            if offline:
                return int(round((user_prior + item_prior) / 2.0))
            raise RuntimeError("LLM rating requested but OpenAI not available.")

        prompt = (
            "You are rating a product review on a 1-5 star scale.\n"
            f"User leniency prior (typical rating): {user_prior:.2f}\n"
            f"Item reputation prior: {item_prior:.2f}\n"
            "Generated review text:\n"
            f"---\n{text}\n---\n\n"
            "Instructions:\n"
            "- Consider the text's sentiment and the priors as context.\n"
            "- Output ONLY an integer 1, 2, 3, 4, or 5 with no other text.\n"
        )
        resp = call_gpt5_with_responses_api(prompt=prompt, model=model_name, max_output_tokens=5, temperature=0.2)
        parsed = re.findall(r"[1-5]", resp)
        if parsed:
            return int(parsed[0])
        return int(round((user_prior + item_prior) / 2.0))

    def predict(self, text: str, user_prior: float, item_prior: float, rng: random.Random, use_llm_for_rating: bool, model_name: str) -> int:
        if use_llm_for_rating:
            try:
                return self._llm_rate(text, user_prior=user_prior, item_prior=item_prior, model_name=model_name)
            except Exception:
                pass
        return self._heuristic_predict(text, user_prior=user_prior, item_prior=item_prior, rng=rng)


@dataclass
class QAConsistency:
    consistency_threshold: float = 0.75
    max_auto_fix_attempts: int = 1
    penalty_weight_style_violations: float = 0.5

    def _policy_violation_score(self, text: str) -> float:
        # Returns 1.0 if clean, lower if violating
        lower = text.lower()
        violations = 0
        if any(b in lower for b in BANNED_WORDS):
            violations += 1
        # excessive caps/punctuation
        if len(re.findall(r"[A-Z]{5,}", text)) > 0:
            violations += 1
        if len(re.findall(r"!{3,}", text)) > 0:
            violations += 1
        return max(0.0, 1.0 - 0.5 * violations)

    def score(self, text: str, proposed_stars: int, plan: Dict[str, Any]) -> float:
        s = sentiment_score(text)
        star_polarity = 1 if proposed_stars >= 4 else (-1 if proposed_stars <= 2 else 0)
        sent_polarity = 1 if s > 0.15 else (-1 if s < -0.15 else 0)
        agreement = 1.0 if star_polarity == sent_polarity else (0.6 if (star_polarity == 0 or sent_polarity == 0) else 0.0)

        target_len = plan.get("length_target", 3)
        dev = abs(len(sentence_split(text)) - target_len)
        style_penalty = math.exp(-0.5 * dev)

        policy_weight = float(plan.get("platform_policy_weight", 0.5))
        policy_ok = self._policy_violation_score(text)

        # Combine agreement and policy/style adherence
        combined = (agreement + self.penalty_weight_style_violations * style_penalty * policy_ok * policy_weight) / (1.0 + self.penalty_weight_style_violations)
        return max(0.0, min(1.0, combined))

    def maybe_revise(self, author: ReviewAuthor, user_ctx: Dict[str, Any], item_ctx: Dict[str, Any],
                     plan: Dict[str, Any], proposed_stars: int, text: str, use_llm: bool,
                     model_name: str, max_output_tokens: int) -> Tuple[str, int, float, int]:
        score0 = self.score(text, proposed_stars, plan)
        if score0 >= self.consistency_threshold or self.max_auto_fix_attempts <= 0:
            return text, proposed_stars, score0, 0

        revision_count = 0
        final_text = text
        final_stars = proposed_stars
        best_score = score0
        for _ in range(self.max_auto_fix_attempts):
            revision_count += 1
            if proposed_stars >= 4:
                plan["tone_target"] = "positive"
            elif proposed_stars <= 2:
                plan["tone_target"] = "negative"
            else:
                plan["tone_target"] = "neutral"
            author.generation_guidelines = "Revise to better match the target tone and maintain factuality. Avoid profanity and adhere to platform style."

            try:
                new_text = author.generate(user_ctx, item_ctx, plan, use_llm=use_llm, model_name=model_name, max_output_tokens=max_output_tokens)
            except Exception:
                new_text = final_text
            new_score = self.score(new_text, proposed_stars, plan)
            if new_score > best_score:
                final_text = new_text
                best_score = new_score
                if best_score >= self.consistency_threshold:
                    break

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
            "mapping_slope": uniform(2.0, 8.0),
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

    def fit(self, data: DataIndexer, train_idxs: List[int], simulator: "Simulator", rng: random.Random, use_llm: bool = False) -> Dict[str, Any]:
        no_improve_rounds = 0
        for t in range(self.num_trials):
            params = self.propose(rng)
            self.objective_weights = params["objective_weights"]
            simulator.apply_params(params)
            metrics, _ = simulator.rollout(idxs=train_idxs, use_llm=use_llm, use_llm_for_rating=False, max_records=len(train_idxs), collect_traces=False, mutate_state=False)
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

    def _compute_overall(self, records: List[Dict[str, Any]], preds: List[Dict[str, Any]]) -> Dict[str, float]:
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
            coverage_hits = 0
            for a in planned:
                if mention_aspect(pr.get("review", ""), a):
                    coverage_hits += 1
            coverage = coverage_hits / max(1, len(planned))
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

        overall_metrics = self._compute_overall(records, preds)
        self.last_eval_metrics = overall_metrics

        # Segment metrics by user frequency tertiles
        user_counts = Counter([gt["user_id"] for gt in records])
        counts = list(user_counts.values())
        if counts:
            sorted_counts = sorted(counts)
            n = len(sorted_counts)
            t1 = sorted_counts[max(0, (n * 1) // 3 - 1)]
            t2 = sorted_counts[max(0, (n * 2) // 3 - 1)]
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
            seg_overall = self._compute_overall([records[i] for i in idxs], [preds[i] for i in idxs])
            self.by_segment_metrics[seg] = seg_overall

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

    def rollout(self, idxs: Optional[List[int]] = None, use_llm: bool = True, use_llm_for_rating: bool = True,
                max_records: Optional[int] = None, collect_traces: bool = True,
                model_name: str = "gpt-5", max_output_tokens: int = 600, mutate_state: bool = True) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
        if idxs is None:
            idxs = list(range(len(self.data.interactions)))
        if max_records is not None:
            idxs = idxs[:max_records]

        preds: List[Dict[str, Any]] = []
        records: List[Dict[str, Any]] = []

        def process_record(i: int) -> Optional[Dict[str, Any]]:
            rec = self.data.interactions[i]
            user_id = rec["user_id"]
            item_id = rec["item_id"]

            user_ctx_full = self.persona.get_persona(user_id, self.data)
            item_ctx_full = self.item_profiler.item_state.get(item_id)
            if not item_ctx_full:
                item_ctx_full = self.data.get_item_context(item_id)
                item_ctx_full["aspect_summary"] = {a: 1.0 / len(DEFAULT_ASPECT_VOCAB) for a in DEFAULT_ASPECT_VOCAB}
                item_ctx_full["recent_reviews"] = self.data.item_text_history.get(item_id, [])[-3:]

            plan = self.planner.compose(user_ctx_full, item_ctx_full, self.platform_policy_weight)

            review_text = self.author.generate(
                user_ctx=user_ctx_full, item_ctx=item_ctx_full, plan=plan,
                use_llm=use_llm, model_name=model_name, max_output_tokens=max_output_tokens
            )

            user_prior = user_ctx_full.get("baseline_leniency", self.data.global_mean)
            item_prior = item_ctx_full.get("reputation_prior", self.data.global_mean)
            stars = self.rater.predict(review_text, user_prior=user_prior, item_prior=item_prior, rng=self.rng, use_llm_for_rating=use_llm_for_rating, model_name=model_name)

            final_text, final_stars, consistency_score_val, revision_count = self.qa.maybe_revise(
                author=self.author, user_ctx=user_ctx_full, item_ctx=item_ctx_full, plan=plan,
                proposed_stars=stars, text=review_text, use_llm=use_llm, model_name=model_name, max_output_tokens=max_output_tokens
            )

            if mutate_state:
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
            return pred

        if collect_traces:
            try:
                with open(SIM_TRACES_FILE, "w", encoding="utf-8") as traces_fp:
                    for i in idxs:
                        pred = process_record(i)
                        if pred is None:
                            continue
                        preds.append(pred)
                        records.append(self.data.interactions[i])
                        trace_obj = {
                            "record_index": i,
                            "inputs": {
                                "user_id": pred["user_id"],
                                "item_id": pred["item_id"]
                            },
                            "plan": {
                                "planned_aspects": pred["planned_aspects"],
                                "length_target": pred["length_target"],
                                "tone_target": pred["tone_target"],
                                "platform_policy_weight": self.platform_policy_weight
                            },
                            "generated_text": pred["review"],
                            "predicted_stars": pred["stars"],
                            "diagnostics": {
                                "consistency_score": pred["consistency_score"],
                                "revision_count": pred["revision_count"]
                            }
                        }
                        traces_fp.write(json.dumps(trace_obj) + "\n")
            except Exception as e:
                raise
        else:
            for i in idxs:
                pred = process_record(i)
                if pred is None:
                    continue
                preds.append(pred)
                records.append(self.data.interactions[i])

        results = self.evaluator.compute_metrics(records, preds, self.data)
        overall = results.get("overall", {})
        return overall, preds


def build_network_and_agents(data_bundle: Dict[str, Any], seed: int) -> Tuple[DataIndexer, PersonaProfiler, ItemProfiler, PlanComposer, ReviewAuthor, StarRater, QAConsistency, Evaluator, Simulator]:
    rng = random.Random(seed)
    interactions = data_bundle["interactions"]
    user_profiles = data_bundle["user_profiles"]
    item_metadata = data_bundle["item_metadata"]

    data_indexer = DataIndexer(interactions=interactions, user_profiles=user_profiles, item_metadata=item_metadata)
    # Attach extra histories if provided
    extra_user_reviews = data_bundle.get("extra_user_reviews")
    if extra_user_reviews:
        data_indexer.__dict__["extra_user_reviews"] = extra_user_reviews
    extra_item_reviews = data_bundle.get("extra_item_reviews")
    if extra_item_reviews:
        data_indexer.__dict__["extra_item_reviews"] = extra_item_reviews
    data_indexer.refresh()

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
        rng=rng
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


def run_ablation(sim: Simulator, test_idxs: List[int], base_params: Dict[str, Any], use_llm: bool, use_llm_for_rating: bool,
                 model_name: str, max_output_tokens: int, max_records: int) -> Dict[str, Any]:
    results = {}
    configs = {
        "base": base_params,
        "no_peer_influence": {**base_params, "neighbor_weight": 0.0},
        "persona_driven_planning": {**base_params, "ctx_merge_weight": min(0.8, max(0.2, (base_params.get("ctx_merge_weight", 0.5) + 0.2)))},
        "item_driven_planning": {**base_params, "ctx_merge_weight": min(0.8, max(0.2, (base_params.get("ctx_merge_weight", 0.5) - 0.2)))},
    }
    for name, cfg in configs.items():
        sim.apply_params(cfg)
        metrics, _ = sim.rollout(idxs=test_idxs, use_llm=use_llm, use_llm_for_rating=use_llm_for_rating,
                                 max_records=max_records, collect_traces=False, model_name=model_name,
                                 max_output_tokens=max_output_tokens, mutate_state=False)
        results[name] = metrics
    return results


def main() -> None:
    args = parse_cli()
    set_global_seed(args.seed)

    # Enforce LLM availability for validation unless OFFLINE_MODE=1 or explicitly disabled
    offline = os.environ.get("OFFLINE_MODE", "0") == "1"
    if args.use_llm == 1 or args.use_llm_rating == 1:
        if not offline:
            try:
                _ = get_openai_api_key()
                if OpenAI is None:
                    raise RuntimeError("OpenAI SDK not available.")
            except Exception as e:
                raise RuntimeError(f"LLM is required but not available: {e}") from e

    data_bundle = load_data(args)
    data_indexer, persona, item_profiler, planner, author, rater, qa, evaluator, simulator = build_network_and_agents(data_bundle, seed=args.seed)
    train_idxs, test_idxs = holdout_split(data_indexer)

    tuner = ParameterTuner(num_trials=args.num_trials, early_stop_patience=args.early_stop_patience)
    # Calibration offline by default to reduce costs
    best_params = tuner.fit(data_indexer, train_idxs, simulator, rng=random.Random(args.seed), use_llm=False)

    simulator.apply_params(best_params)

    use_llm_validation = bool(args.use_llm == 1)
    use_llm_rating_validation = bool(args.use_llm_rating == 1)

    max_val = max(0, args.max_validation_records) or None

    metrics_overall, preds = simulator.rollout(
        idxs=test_idxs,
        use_llm=use_llm_validation or False,
        use_llm_for_rating=use_llm_rating_validation or False,
        max_records=max_val,
        collect_traces=True,
        model_name=args.model_name,
        max_output_tokens=args.max_output_tokens,
        mutate_state=True
    )

    eval_metrics = {"overall": metrics_overall, "by_segment": simulator.evaluator.by_segment_metrics}

    ablation_report = None
    if args.ablation == 1:
        ablation_report = run_ablation(
            simulator, test_idxs, best_params, use_llm=use_llm_validation or False, use_llm_for_rating=use_llm_rating_validation or False,
            model_name=args.model_name, max_output_tokens=args.max_output_tokens,
            max_records=max_val if isinstance(max_val, int) else None
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