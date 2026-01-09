PLAYBOOK_USAGE_JSON = '''{"used_bullets":[{"id":"missing-fine-category-generation-model","why":"POI recall@K was low while category/transition metrics were already good, indicating the generator was too coarse (super-category only). Added a fine-category layer (fit on 2019–2020) and sample POIs conditioned on fine category."},{"id":"poi-recall-low-from-overweighting-generic-popularity-and-repeats","why":"Simulated days showed repeated POIs, which harms set-based recall@K. Added within-day anti-repeat penalties and soft rejection to increase diversity and improve overlap potential."},{"id":"stop-count-mae-high-from-no-2021-calibrated-stopcount-shift-structure","why":"Stop-count MAE remained high; improved stop-count targeting by blending baseline mean with 7-day context mean (strictly earlier days only), keeping the same calibrated parameter but making it context-aware."}]}'''
CHANGE_SUMMARY_JSON = '''{"touched_symbols":[{"symbol":"DATA_DIR","reason":"Adjusted path setup to avoid import-time failure when env vars are missing while preserving the required join structure."},{"symbol":"load_data","reason":"Strengthened env var validation to check truthiness (non-empty) rather than non-None to match the safer path initialization."}],"applied_strategies":[{"id":"missing-fine-category-generation-model","applied":true},{"id":"poi-recall-low-from-overweighting-generic-popularity-and-repeats","applied":true},{"id":"stop-count-mae-high-from-no-2021-calibrated-stopcount-shift-structure","applied":true}]}'''
import abc
import argparse
import datetime as dt
import hashlib
import json
import math
import random
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

try:
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore

# Path Handling Instructions (COPY EXACTLY)
import os
PROJECT_ROOT = os.environ.get("PROJECT_ROOT") or ""
DATA_PATH = os.environ.get("DATA_PATH") or ""
DATA_DIR = os.path.join(PROJECT_ROOT, DATA_PATH)


def get_openai_api_key() -> str:
    """Get OpenAI API key from environment."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        return api_key
    raise ValueError("OpenAI API key not found in environment")


def call_gpt5_with_responses_api(prompt: str, model: str = "gpt-5", max_output_tokens: int = 4000) -> str:
    """Call OpenAI Responses API (not used by the mobility simulator workflow)."""
    api_key = get_openai_api_key()
    model = os.environ.get("OPENAI_MODEL", model)

    if OpenAI is None:
        raise ImportError("OpenAI SDK is not available. Install the 'openai' package to use LLM functionality.")

    try:
        client = OpenAI(api_key=api_key)
    except TypeError:
        os.environ["OPENAI_API_KEY"] = api_key
        client = OpenAI()

    if not hasattr(client, "responses") or not hasattr(client.responses, "create"):
        raise RuntimeError(
            "Installed OpenAI SDK does not support client.responses.create (Responses API). "
            "Please upgrade the 'openai' package to a version that supports the Responses API."
        )

    responses_kwargs = {
        "model": model,
        "input": [{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
        "max_output_tokens": int(max_output_tokens),
    }

    try:
        resp = client.responses.create(**responses_kwargs)
    except Exception as e:
        raise RuntimeError("OpenAI Responses API call failed.") from e

    def extract_response(resp_obj: Any) -> str:
        if hasattr(resp_obj, "output_text") and isinstance(resp_obj.output_text, str):
            return resp_obj.output_text
        try:
            output = getattr(resp_obj, "output", None)
            if output and isinstance(output, list):
                first = output[0]
                content = first.get("content") if isinstance(first, dict) else None
                if content and isinstance(content, list) and len(content) > 0:
                    item0 = content[0]
                    text = item0.get("text") if isinstance(item0, dict) else None
                    if isinstance(text, str):
                        return text
        except Exception:
            pass
        return str(resp_obj)

    return extract_response(resp)


class MemoryAgent:
    """
    Minimal Memory Agent stub used for LLM prompt construction.

    Not used by the mobility simulator workflow; provided to satisfy LLM-calling requirement.
    """

    def get_user_context(self, user_id: str) -> str:
        return f"user_id={user_id}"

    def get_item_context(self, item_id: str) -> str:
        return f"item_id={item_id}"


class PlanningAgent:
    """
    Minimal Planning Agent stub used for LLM prompt construction.

    Not used by the mobility simulator workflow; provided to satisfy LLM-calling requirement.
    """

    def plan(self, task: str) -> str:
        return f"Task plan:\n1) Understand request\n2) Draft response\n3) Ensure constraints\nTask: {task}"


class ReasoningAgent:
    """
    Reasoning Agent that MUST perform its reasoning via an LLM call.

    Not used by the mobility simulator workflow; provided to satisfy LLM-calling requirement.
    """

    def __init__(self, memory_agent: MemoryAgent, planning_agent: PlanningAgent):
        self.memory_agent = memory_agent
        self.planning_agent = planning_agent

    def generate(self, user_id: str, item_id: str, task: str, model: str = "gpt-5") -> str:
        user_ctx = self.memory_agent.get_user_context(user_id)
        item_ctx = self.memory_agent.get_item_context(item_id)
        plan = self.planning_agent.plan(task)

        prompt = (
            "You are a helpful assistant.\n\n"
            "USER CONTEXT:\n"
            f"{user_ctx}\n\n"
            "ITEM/PRODUCT CONTEXT:\n"
            f"{item_ctx}\n\n"
            "PLAN / TASK DECOMPOSITION:\n"
            f"{plan}\n\n"
            "INSTRUCTION:\n"
            "Use the contexts and the plan to produce the primary response text.\n"
        )

        response = call_gpt5_with_responses_api(prompt=prompt, model=model, max_output_tokens=4000)
        return str(response).strip()


class ReviewAuthor:
    """
    Example integration point referenced by the requirement.

    Not used by the mobility simulator workflow.
    """

    def __init__(self, reasoning_agent: ReasoningAgent):
        self.reasoning_agent = reasoning_agent

    def generate(self, user_id: str, item_id: str, task: str = "Write a concise product review.") -> str:
        return self.reasoning_agent.generate(user_id=user_id, item_id=item_id, task=task, model="gpt-5")


DEFAULT_SEED = 1337
TIME_BIN_MINUTES = 15
MINUTES_PER_DAY = 24 * 60
random.seed(DEFAULT_SEED)

INFRA_KEYWORDS = (
    "Toll",
    "Booth",
    "Tunnel",
    "Rest",
    "Area",
    "Platform",
    "Station",
    "Metro",
    "Rail",
    "Bus",
    "Airport",
    "Port",
)


def log_info(msg: str) -> None:
    """Print an INFO log line to stdout."""
    print(f"[INFO] {msg}")


def require(condition: bool, message: str) -> None:
    """Raise a ValueError if condition is false."""
    if not condition:
        raise ValueError(message)


def stable_int_hash(text: str) -> int:
    """Deterministic stable hash for seeding."""
    h = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def normalize_poi_token(token: str) -> str:
    """Normalize a POI token to reduce catalog/GT mismatches."""
    if not isinstance(token, str):
        return str(token)

    t = token.strip()
    t = re.sub(r"[\u00A0\u2000-\u200B\u202F\u205F\u3000]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"\s*#\s*", "#", t)

    while t and t[-1] in ".,;:":
        t = t[:-1].rstrip()
    return t


def parse_date(date_str: str) -> dt.date:
    """Parse YYYY-MM-DD to date."""
    try:
        y, m, d = date_str.split("-")
        return dt.date(int(y), int(m), int(d))
    except Exception as e:
        raise ValueError(f"Invalid date token '{date_str}'. Expected YYYY-MM-DD.") from e


def day_type(date_obj: dt.date) -> str:
    """Return 'weekday' or 'weekend'."""
    return "weekend" if date_obj.weekday() >= 5 else "weekday"


def hhmmss_to_minute_of_day(t: str) -> int:
    """Convert HH:MM:SS to minute-of-day (seconds ignored)."""
    t = t.strip()
    while t and t[-1] in ".,":  # strip trailing punctuation
        t = t[:-1]
    try:
        parts = t.split(":")
        if len(parts) != 3:
            raise ValueError(f"Invalid time token '{t}'. Expected H:MM:SS or HH:MM:SS.")
        hh, mm, ss = (p.strip() for p in parts)
        h, m, s = int(hh), int(mm), int(ss)
    except ValueError as e:
        if "Invalid time token" in str(e):
            raise
        raise ValueError(f"Invalid time token '{t}'. Expected H:MM:SS or HH:MM:SS.") from e
    except Exception as e:
        raise ValueError(f"Invalid time token '{t}'. Expected H:MM:SS or HH:MM:SS.") from e
    require(0 <= h <= 23 and 0 <= m <= 59 and 0 <= s <= 59, f"Out-of-range time '{t}'.")
    return h * 60 + m


def minute_of_day_to_hhmmss(minute_of_day: int, seconds: int) -> str:
    """Convert minute-of-day and seconds to HH:MM:SS."""
    minute_of_day = max(0, min(MINUTES_PER_DAY - 1, int(minute_of_day)))
    seconds = max(0, min(59, int(seconds)))
    h = minute_of_day // 60
    m = minute_of_day % 60
    return f"{h:02d}:{m:02d}:{seconds:02d}"


class TrajectoryStringFormatter:
    FORMAT_SPEC = (
        "Activities at YYYY-MM-DD: POI#id at HH:MM:SS, POI#id at HH:MM:SS, ... . "
        "Canonical emission here uses comma+space separators and ends with a single '.' (no preceding space)."
    )


def weighted_choice(items: Sequence[Any], weights: Sequence[float], u: float) -> Any:
    """Sample one item using non-negative weights and a provided uniform u in [0,1)."""
    require(len(items) == len(weights) and len(items) > 0, "weighted_choice: items/weights mismatch or empty.")
    total = 0.0
    for w in weights:
        require(w >= 0.0 and math.isfinite(w), "weighted_choice: weights must be finite and non-negative.")
        total += w
    require(total > 0.0, "weighted_choice: sum of weights is zero; cannot sample.")
    threshold = u * total
    cum = 0.0
    for item, w in zip(items, weights):
        cum += w
        if cum >= threshold:
            return item
    return items[-1]


def normalize_counter(counter: Mapping[str, float], epsilon: float = 0.0) -> Dict[str, float]:
    """Normalize a non-negative mapping into probabilities."""
    keys = list(counter.keys())
    vals = []
    for k in keys:
        v = float(counter[k])
        require(v >= 0.0 and math.isfinite(v), f"Invalid non-negative finite value for key '{k}': {v}")
        vals.append(v + epsilon)
    s = sum(vals)
    require(s > 0.0, "normalize_counter: sum is zero; cannot normalize.")
    return {k: (v + epsilon) / s for k, v in zip(keys, vals)}


def js_divergence(p: Mapping[str, float], q: Mapping[str, float], epsilon: float = 1e-12) -> float:
    """Jensen-Shannon divergence between distributions."""
    keys = set(p.keys()) | set(q.keys())
    p2 = {k: max(epsilon, float(p.get(k, 0.0))) for k in keys}
    q2 = {k: max(epsilon, float(q.get(k, 0.0))) for k in keys}
    sp = sum(p2.values())
    sq = sum(q2.values())
    p2 = {k: v / sp for k, v in p2.items()}
    q2 = {k: v / sq for k, v in q2.items()}
    m = {k: 0.5 * (p2[k] + q2[k]) for k in keys}

    def kl(a: Mapping[str, float], b: Mapping[str, float]) -> float:
        out = 0.0
        for k in keys:
            out += a[k] * math.log(a[k] / b[k])
        return out

    return 0.5 * kl(p2, m) + 0.5 * kl(q2, m)


def kl_divergence(p: Mapping[str, float], q: Mapping[str, float], epsilon: float = 1e-12) -> float:
    """KL divergence KL(p||q) with epsilon smoothing."""
    keys = set(p.keys()) | set(q.keys())
    p2 = {k: max(epsilon, float(p.get(k, 0.0))) for k in keys}
    q2 = {k: max(epsilon, float(q.get(k, 0.0))) for k in keys}
    sp = sum(p2.values())
    sq = sum(q2.values())
    p2 = {k: v / sp for k, v in p2.items()}
    q2 = {k: v / sq for k, v in q2.items()}
    out = 0.0
    for k in keys:
        out += p2[k] * math.log(p2[k] / q2[k])
    return out


def shares_mae(p: Mapping[str, float], q: Mapping[str, float]) -> float:
    """Mean absolute error between distribution shares across union of keys."""
    keys = set(p.keys()) | set(q.keys())
    if not keys:
        return 0.0
    return sum(abs(float(p.get(k, 0.0)) - float(q.get(k, 0.0))) for k in keys) / float(len(keys))


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine distance in km."""
    r = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1 - a)))
    return r * c


def wasserstein_1d(a: Sequence[float], b: Sequence[float], quantiles: int = 200) -> float:
    """Approximate 1D Wasserstein distance via quantile averaging."""
    if len(a) == 0 and len(b) == 0:
        return 0.0
    if len(a) == 0 or len(b) == 0:
        all_vals = list(a) + list(b)
        if not all_vals:
            return 0.0
        return float(max(all_vals) - min(all_vals))

    a_sorted = sorted(float(x) for x in a)
    b_sorted = sorted(float(x) for x in b)

    def q(arr: List[float], t: float) -> float:
        if len(arr) == 1:
            return arr[0]
        pos = t * (len(arr) - 1)
        i = int(math.floor(pos))
        j = min(len(arr) - 1, i + 1)
        frac = pos - i
        return arr[i] * (1 - frac) + arr[j] * frac

    quantiles = max(2, int(quantiles))
    qs = [i / (quantiles - 1) for i in range(quantiles)]
    return sum(abs(q(a_sorted, t) - q(b_sorted, t)) for t in qs) / float(quantiles)


@dataclass(frozen=True)
class Visit:
    poi_token: str
    time_hhmmss: str

    @property
    def fine_category(self) -> str:
        if "#" in self.poi_token:
            return self.poi_token.split("#", 1)[0]
        return self.poi_token


@dataclass(frozen=True)
class DayRecord:
    user_id: str
    date: dt.date
    visits: List[Visit]
    raw: str

    @property
    def year(self) -> int:
        return self.date.year


@dataclass(frozen=True)
class POIInfo:
    poi_token: str
    fine_category: str
    super_category: str
    lat: Optional[float]
    lon: Optional[float]


@dataclass
class BaselineModel:
    anchor_pois: List[str]
    anchor_weights: List[float]
    start_time_hist_by_daytype: Dict[str, List[float]]
    stop_count_pmf_by_daytype: Dict[str, Dict[int, float]]
    gap_minutes_samples_by_daytype: Dict[str, List[int]]
    initial_supercat_pmf: Dict[str, float]
    transition_pmf: Dict[str, Dict[str, float]]
    poi_pref: Dict[str, float]
    supercat_pref: Dict[str, float]
    finecat_pref_by_supercat: Dict[str, Dict[str, float]] = field(default_factory=dict)
    finecat_transition_sparse: Dict[str, Dict[str, float]] = field(default_factory=dict)


@dataclass
class ParsedDataset:
    day_records_by_user: Dict[str, List["DayRecord"]]
    day_record_lookup: Dict[Tuple[str, dt.date], "DayRecord"]
    poi_by_token: Dict[str, POIInfo]
    poi_tokens_by_supercat: Dict[str, List[str]]
    supercat_by_finecat: Dict[str, str]
    infra_supercats: List[str]
    poi_token_by_normalized: Dict[str, str] = field(default_factory=dict)
    poi_token_by_casefold: Dict[str, str] = field(default_factory=dict)
    poi_tokens_by_finecat: Dict[str, List[str]] = field(default_factory=dict)


@dataclass
class HoldoutSplits:
    train_users: List[str]
    calib_days_by_user: Dict[str, List[dt.date]]
    test_days_by_user: Dict[str, List[dt.date]]
    excluded_sparse_users: List[str]
    sparse_handling: str


@dataclass
class RolloutResult:
    trajectories_strings: Dict[str, List[str]]
    visits_by_user_day: Dict[Tuple[str, dt.date], List[Visit]]
    meta: Dict[str, Any]


def _load_json(path: str) -> Any:
    require(os.path.isabs(path), f"Expected absolute path, got: {path}")
    require(os.path.exists(path), f"Missing required input file: {path}")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to read/parse JSON file: {path}") from e


_VISIT_PATTERN = re.compile(
    r"""
    (?P<loc>.+?)\s+at\s+(?P<time>\d{1,2}:\d{2}:\d{2})
    (?:\s*[,;]\s*|\s*\.\s*|$)
    """,
    re.VERBOSE,
)


def parse_1921y_records(user_id: str, day_strings: Iterable[str]) -> List[DayRecord]:
    """Parse 1921Y.json day strings into structured DayRecord objects."""
    out: List[DayRecord] = []
    for raw in day_strings:
        if not isinstance(raw, str):
            raise ValueError(f"1921Y.json: expected day record string for user '{user_id}', got {type(raw)}")

        prefix = "Activities at "
        idx = raw.find(prefix)
        require(idx == 0, f"Unexpected record prefix for user '{user_id}': {raw[:50]}")
        after = raw[len(prefix) :]
        require(": " in after or after.endswith(":"), f"Missing ':' separator in record: {raw[:80]}")

        if ": " in after:
            date_part, rest = after.split(": ", 1)
        else:
            date_part = after[:-1]
            rest = ""

        date_obj = parse_date(date_part.strip())
        visits: List[Visit] = []

        rest = rest.strip()
        if rest:
            for m in _VISIT_PATTERN.finditer(rest):
                loc = (m.group("loc") or "").strip()
                t = (m.group("time") or "").strip()
                loc = normalize_poi_token(loc)
                if not loc:
                    continue
                _ = hhmmss_to_minute_of_day(t)
                visits.append(Visit(poi_token=loc, time_hhmmss=t))

            if not visits:
                if ";" in rest:
                    candidate_splits = [p.strip() for p in rest.split(";") if p.strip()]
                elif "," in rest:
                    candidate_splits = [p.strip() for p in rest.split(",") if p.strip()]
                else:
                    candidate_splits = [rest.strip()]

                for p in candidate_splits:
                    p2 = p.strip().rstrip(".")
                    if not p2:
                        continue
                    if " at " not in p2:
                        raise ValueError(
                            f"Invalid visit token (missing ' at ') for user '{user_id}' on {date_obj}: '{p2}'"
                        )
                    loc, t = p2.rsplit(" at ", 1)
                    loc = normalize_poi_token(loc.strip())
                    t = t.strip()
                    _ = hhmmss_to_minute_of_day(t)
                    require(len(loc) > 0, f"Empty location token for user '{user_id}' on {date_obj}.")
                    visits.append(Visit(poi_token=loc, time_hhmmss=t))

        out.append(DayRecord(user_id=user_id, date=date_obj, visits=visits, raw=raw))

    out.sort(key=lambda r: r.date)
    return out


def load_poi_catalog(
    poi_json: Any, supercat_by_finecat: Dict[str, str]
) -> Tuple[Dict[str, POIInfo], Dict[str, List[str]], Dict[str, str], Dict[str, str]]:
    """Load POI catalog and build indices for canonicalization."""
    poi_by_token: Dict[str, POIInfo] = {}
    by_supercat: Dict[str, List[str]] = {}
    norm_index: Dict[str, str] = {}
    casefold_index: Dict[str, str] = {}

    def add_record(lat: Any, lon: Any, token: Any) -> None:
        if not isinstance(token, str):
            return
        token2 = normalize_poi_token(token)
        finecat = token2.split("#", 1)[0] if "#" in token2 else token2
        supercat = supercat_by_finecat.get(finecat, "Unknown")
        lat_f: Optional[float] = None
        lon_f: Optional[float] = None
        try:
            if lat is not None and lon is not None:
                lat_f = float(lat)
                lon_f = float(lon)
                if not (-90.0 <= lat_f <= 90.0 and -180.0 <= lon_f <= 180.0):
                    lat_f, lon_f = None, None
        except Exception:
            lat_f, lon_f = None, None

        poi_by_token[token2] = POIInfo(
            poi_token=token2,
            fine_category=finecat,
            super_category=supercat,
            lat=lat_f,
            lon=lon_f,
        )
        by_supercat.setdefault(supercat, []).append(token2)

        norm = normalize_poi_token(token2)
        if norm and norm not in norm_index:
            norm_index[norm] = token2
        ckey = norm.casefold()
        if ckey and ckey not in casefold_index:
            casefold_index[ckey] = token2

    if isinstance(poi_json, dict):
        for _cat, records in poi_json.items():
            if not isinstance(records, list):
                continue
            for rec in records:
                if isinstance(rec, list) and len(rec) >= 3:
                    add_record(rec[0], rec[1], rec[2])
    elif isinstance(poi_json, list):
        for rec in poi_json:
            if isinstance(rec, list) and len(rec) >= 3:
                add_record(rec[0], rec[1], rec[2])
    else:
        raise ValueError(
            "poi_category_192021_longitude_latitude.json has unexpected JSON type. Expected dict or list."
        )

    require(len(poi_by_token) > 0, "POI catalog parsed empty; check poi_category_192021_longitude_latitude.json format.")
    return poi_by_token, by_supercat, norm_index, casefold_index


def _canonicalize_visits_to_catalog(
    visits: List[Visit],
    poi_by_token: Mapping[str, POIInfo],
    norm_index: Mapping[str, str],
    casefold_index: Mapping[str, str],
) -> List[Visit]:
    """Map visit POI tokens to catalog canonical tokens when possible."""
    out: List[Visit] = []
    for v in visits:
        tok = normalize_poi_token(v.poi_token)
        canonical: Optional[str] = None
        if tok in poi_by_token:
            canonical = tok
        else:
            canonical = norm_index.get(tok)
            if canonical is None:
                canonical = casefold_index.get(tok.casefold())
        if canonical is None:
            canonical = tok
        out.append(Visit(poi_token=canonical, time_hhmmss=v.time_hhmmss))
    return out


def load_data() -> ParsedDataset:
    """Load 1921Y.json + POI catalog + category mapping and build parsed dataset."""
    require(bool(PROJECT_ROOT) and bool(DATA_PATH), "Missing env vars PROJECT_ROOT and/or DATA_PATH.")
    require(isinstance(DATA_DIR, str) and len(DATA_DIR) > 0, "DATA_DIR is empty; check PROJECT_ROOT/DATA_PATH.")

    data_dir_abs = os.path.abspath(DATA_DIR)
    require(os.path.isabs(data_dir_abs), f"DATA_DIR must be absolute after abspath(), got: {data_dir_abs}")

    path_1921y = os.path.join(data_dir_abs, "1921Y.json")
    path_poi = os.path.join(data_dir_abs, "poi_category_192021_longitude_latitude.json")
    path_catto = os.path.join(data_dir_abs, "catto.json")

    log_info(f"Loading data from DATA_DIR={data_dir_abs}")
    raw_1921y = _load_json(os.path.abspath(path_1921y))
    raw_poi = _load_json(os.path.abspath(path_poi))
    raw_catto = _load_json(os.path.abspath(path_catto))

    require(isinstance(raw_catto, dict), "catto.json must be a JSON object mapping fine category -> super-category.")
    supercat_by_finecat = {str(k): str(v) for k, v in raw_catto.items()}

    poi_by_token, poi_tokens_by_supercat, norm_index, casefold_index = load_poi_catalog(raw_poi, supercat_by_finecat)

    require(isinstance(raw_1921y, dict), "1921Y.json must be a JSON object mapping user_id -> list of day strings.")
    day_records_by_user: Dict[str, List[DayRecord]] = {}
    lookup: Dict[Tuple[str, dt.date], DayRecord] = {}

    for user_id, payload in raw_1921y.items():
        if isinstance(payload, list):
            day_strings = payload
        elif isinstance(payload, dict):
            day_strings = list(payload.values())
        else:
            raise ValueError(f"1921Y.json: user '{user_id}' has unsupported type: {type(payload)}")
        records = parse_1921y_records(str(user_id), day_strings)

        canon_records: List[DayRecord] = []
        for r in records:
            canon_visits = _canonicalize_visits_to_catalog(r.visits, poi_by_token, norm_index, casefold_index)
            canon_records.append(DayRecord(user_id=r.user_id, date=r.date, visits=canon_visits, raw=r.raw))

        day_records_by_user[str(user_id)] = canon_records
        for r in canon_records:
            lookup[(r.user_id, r.date)] = r

    infra_supercats_set = set()
    for finecat, supercat in supercat_by_finecat.items():
        if any(kw.lower() in finecat.lower() for kw in INFRA_KEYWORDS) or any(
            kw.lower() in supercat.lower() for kw in INFRA_KEYWORDS
        ):
            infra_supercats_set.add(supercat)
    if not infra_supercats_set:
        for _token, poi in poi_by_token.items():
            if any(kw.lower() in poi.poi_token.lower() for kw in INFRA_KEYWORDS):
                infra_supercats_set.add(poi.super_category)

    infra_supercats = sorted(infra_supercats_set) if infra_supercats_set else ["Unknown"]

    poi_tokens_by_finecat: Dict[str, List[str]] = {}
    for tok, info in poi_by_token.items():
        poi_tokens_by_finecat.setdefault(info.fine_category, []).append(tok)
    for fc in list(poi_tokens_by_finecat.keys()):
        poi_tokens_by_finecat[fc].sort()

    log_info(
        f"Loaded users={len(day_records_by_user)}; POIs={len(poi_by_token)}; supercats={len(poi_tokens_by_supercat)}"
    )
    return ParsedDataset(
        day_records_by_user=day_records_by_user,
        day_record_lookup=lookup,
        poi_by_token=poi_by_token,
        poi_tokens_by_supercat=poi_tokens_by_supercat,
        supercat_by_finecat=supercat_by_finecat,
        infra_supercats=infra_supercats,
        poi_token_by_normalized=norm_index,
        poi_token_by_casefold=casefold_index,
        poi_tokens_by_finecat=poi_tokens_by_finecat,
    )


def holdout_split(dataset: ParsedDataset, sparse_2021_handling: str = "exclude") -> HoldoutSplits:
    """Per-user chronological 80/20 split on 2021 days (V_calib/V_test)."""
    require(sparse_2021_handling in ("exclude", "test_only"), "sparse_2021_handling must be 'exclude' or 'test_only'.")

    calib_days_by_user: Dict[str, List[dt.date]] = {}
    test_days_by_user: Dict[str, List[dt.date]] = {}
    excluded_sparse_users: List[str] = []
    train_users: List[str] = []

    for user_id, records in dataset.day_records_by_user.items():
        has_train = any(r.year in (2019, 2020) and len(r.visits) > 0 for r in records)
        if has_train:
            train_users.append(user_id)

        y2021 = [r for r in records if r.year == 2021 and len(r.visits) > 0]
        y2021.sort(key=lambda r: r.date)
        n = len(y2021)
        if n == 0:
            continue
        if n < 5:
            if sparse_2021_handling == "exclude":
                excluded_sparse_users.append(user_id)
                continue
            calib_days_by_user[user_id] = []
            test_days_by_user[user_id] = [r.date for r in y2021]
            continue

        split_idx = int(math.floor(0.8 * n))
        split_idx = max(1, min(n - 1, split_idx))
        calib_days_by_user[user_id] = [r.date for r in y2021[:split_idx]]
        test_days_by_user[user_id] = [r.date for r in y2021[split_idx:]]

    log_info(
        f"Holdout split: calib_users={len([u for u, v in calib_days_by_user.items() if v])}, "
        f"test_users={len([u for u, v in test_days_by_user.items() if v])}, "
        f"excluded_sparse_users={len(excluded_sparse_users)} (handling={sparse_2021_handling})"
    )
    if excluded_sparse_users:
        log_info(f"Excluded sparse 2021 users (first up to 10): {excluded_sparse_users[:10]}")
    return HoldoutSplits(
        train_users=sorted(set(train_users)),
        calib_days_by_user=calib_days_by_user,
        test_days_by_user=test_days_by_user,
        excluded_sparse_users=excluded_sparse_users,
        sparse_handling=sparse_2021_handling,
    )


def _time_hist(times_minutes: List[int], bin_minutes: int) -> List[float]:
    require(MINUTES_PER_DAY % bin_minutes == 0, "bin_minutes must divide 1440 exactly.")
    bins = MINUTES_PER_DAY // bin_minutes
    counts = [0.0 for _ in range(bins)]
    for m in times_minutes:
        m2 = max(0, min(MINUTES_PER_DAY - 1, int(m)))
        b = m2 // bin_minutes
        counts[b] += 1.0
    eps = 1e-3
    s = sum(c + eps for c in counts)
    return [(c + eps) / s for c in counts]


def _pmf_from_counts(counts: Mapping[int, int], eps: float = 1e-6) -> Dict[int, float]:
    keys = sorted(counts.keys())
    if not keys:
        return {}
    vals = [(float(counts[k]) + eps) for k in keys]
    s = sum(vals)
    return {k: v / s for k, v in zip(keys, vals)}


def _fit_global_medians(dataset: ParsedDataset, train_lookup: Dict[Tuple[str, dt.date], DayRecord]) -> float:
    dists: List[float] = []
    for (_u, _d), rec in train_lookup.items():
        vs = rec.visits
        for i in range(1, len(vs)):
            a = dataset.poi_by_token.get(vs[i - 1].poi_token)
            b = dataset.poi_by_token.get(vs[i].poi_token)
            if a and b and a.lat is not None and a.lon is not None and b.lat is not None and b.lon is not None:
                dists.append(haversine_km(a.lat, a.lon, b.lat, b.lon))
    if not dists:
        return 2.0
    dists.sort()
    return float(dists[len(dists) // 2])


def build_network_and_agents(
    dataset: ParsedDataset,
    splits: HoldoutSplits,
    seed: int,
    tuned_supercats_limit: int = 6,
) -> Tuple["MobilitySimulator", Dict[str, "Resident"]]:
    _ = splits

    all_supercats = sorted(set(dataset.poi_tokens_by_supercat.keys()) | {"Unknown"})

    train_lookup: Dict[Tuple[str, dt.date], DayRecord] = {}
    for user_id, records in dataset.day_records_by_user.items():
        for r in records:
            if r.year in (2019, 2020) and len(r.visits) > 0:
                train_lookup[(user_id, r.date)] = r

    global_step_median_km = _fit_global_medians(dataset, train_lookup)
    log_info(f"Global median step distance (km) from training: {global_step_median_km:.3f}")

    global_start_times: Dict[str, List[int]] = {"weekday": [], "weekend": []}
    global_stop_counts: Dict[str, Dict[int, int]] = {"weekday": {}, "weekend": {}}
    global_gaps: Dict[str, List[int]] = {"weekday": [], "weekend": []}
    global_initial_sc: Dict[str, int] = {}
    global_trans: Dict[Tuple[str, str], int] = {}

    global_poi_counts: Dict[str, int] = {}
    global_supercat_counts: Dict[str, int] = {}
    global_finecat_counts_by_sc: Dict[str, Dict[str, int]] = {sc: {} for sc in all_supercats}
    global_finecat_bigram_counts: Dict[str, Dict[str, int]] = {}
    global_poi_counts_by_sc: Dict[str, Dict[str, int]] = {sc: {} for sc in all_supercats}
    global_poi_counts_by_fc: Dict[str, Dict[str, int]] = {}

    def add_count(dct: Dict[int, int], k: int, v: int = 1) -> None:
        dct[k] = dct.get(k, 0) + v

    def add_sc_count(dct: Dict[str, int], k: str, v: int = 1) -> None:
        dct[k] = dct.get(k, 0) + v

    def sc_of_fc(fc: str) -> str:
        return dataset.supercat_by_finecat.get(fc, "Unknown")

    residents: Dict[str, "Resident"] = {}

    for (_u, _d), rec in train_lookup.items():
        prev_fc: Optional[str] = None
        for v in rec.visits:
            tok = v.poi_token
            global_poi_counts[tok] = global_poi_counts.get(tok, 0) + 1

            fc = v.fine_category
            sc = sc_of_fc(fc)
            global_supercat_counts[sc] = global_supercat_counts.get(sc, 0) + 1

            global_finecat_counts_by_sc.setdefault(sc, {})
            global_finecat_counts_by_sc[sc][fc] = global_finecat_counts_by_sc[sc].get(fc, 0) + 1

            global_poi_counts_by_sc.setdefault(sc, {})
            global_poi_counts_by_sc[sc][tok] = global_poi_counts_by_sc[sc].get(tok, 0) + 1

            global_poi_counts_by_fc.setdefault(fc, {})
            global_poi_counts_by_fc[fc][tok] = global_poi_counts_by_fc[fc].get(tok, 0) + 1

            if prev_fc is not None:
                global_finecat_bigram_counts.setdefault(prev_fc, {})
                global_finecat_bigram_counts[prev_fc][fc] = global_finecat_bigram_counts[prev_fc].get(fc, 0) + 1
            prev_fc = fc

    global_popular_pois = [p for p, _c in sorted(global_poi_counts.items(), key=lambda x: (-x[1], x[0]))]
    require(len(global_popular_pois) > 0, "No training POIs found in 2019-2020 to build global fallback popularity.")

    total_global = float(sum(global_poi_counts.values()))
    global_poi_pref = {p: (c / total_global) for p, c in global_poi_counts.items()} if total_global > 0 else {}

    global_poi_pref_by_sc: Dict[str, Dict[str, float]] = {}
    for sc, counts in global_poi_counts_by_sc.items():
        s = float(sum(counts.values()))
        global_poi_pref_by_sc[sc] = {tok: float(c) / s for tok, c in counts.items()} if s > 0 else {}

    global_popular_pois_by_supercat: Dict[str, List[str]] = {sc: [] for sc in all_supercats}
    for sc in all_supercats:
        items = global_poi_counts_by_sc.get(sc, {})
        global_popular_pois_by_supercat[sc] = [p for p, _c in sorted(items.items(), key=lambda x: (-x[1], x[0]))]

    global_poi_pref_by_fc: Dict[str, Dict[str, float]] = {}
    global_popular_pois_by_finecat: Dict[str, List[str]] = {}
    for fc, counts in global_poi_counts_by_fc.items():
        s = float(sum(counts.values()))
        global_poi_pref_by_fc[fc] = {tok: float(c) / s for tok, c in counts.items()} if s > 0 else {}
        global_popular_pois_by_finecat[fc] = [p for p, _c in sorted(counts.items(), key=lambda x: (-x[1], x[0]))]

    global_finecat_pref_by_sc: Dict[str, Dict[str, float]] = {}
    for sc, counts in global_finecat_counts_by_sc.items():
        s = float(sum(counts.values()))
        global_finecat_pref_by_sc[sc] = {fc: float(c) / s for fc, c in counts.items()} if s > 0 else {}

    for user_id, records in dataset.day_records_by_user.items():
        train_days = [r for r in records if r.year in (2019, 2020) and len(r.visits) > 0]
        if not train_days:
            continue

        poi_counts: Dict[str, int] = {}
        home_like: Dict[str, int] = {}
        start_times_by_dt: Dict[str, List[int]] = {"weekday": [], "weekend": []}
        stop_counts_by_dt: Dict[str, Dict[int, int]] = {"weekday": {}, "weekend": {}}
        gaps_by_dt: Dict[str, List[int]] = {"weekday": [], "weekend": []}
        initial_sc: Dict[str, int] = {}
        trans: Dict[Tuple[str, str], int] = {}
        supercat_counts: Dict[str, int] = {}

        user_top_pois_by_supercat: Dict[str, Dict[str, int]] = {}
        user_poi_counts_by_sc: Dict[str, Dict[str, int]] = {}

        user_fc_counts_by_sc: Dict[str, Dict[str, int]] = {sc: {} for sc in all_supercats}
        user_fc_bigram_counts: Dict[str, Dict[str, int]] = {}
        user_top_pois_by_fc: Dict[str, Dict[str, int]] = {}

        for r in train_days:
            dt_type = day_type(r.date)
            sc_seq: List[str] = []
            start_m = hhmmss_to_minute_of_day(r.visits[0].time_hhmmss)
            start_times_by_dt[dt_type].append(start_m)
            global_start_times[dt_type].append(start_m)

            c = len(r.visits)
            add_count(stop_counts_by_dt[dt_type], c)
            add_count(global_stop_counts[dt_type], c)

            prev_fc: Optional[str] = None
            for i, v in enumerate(r.visits):
                tok = v.poi_token
                poi_counts[tok] = poi_counts.get(tok, 0) + 1
                fc = v.fine_category
                sc = sc_of_fc(fc)
                supercat_counts[sc] = supercat_counts.get(sc, 0) + 1

                user_fc_counts_by_sc.setdefault(sc, {})
                user_fc_counts_by_sc[sc][fc] = user_fc_counts_by_sc[sc].get(fc, 0) + 1

                user_top_pois_by_fc.setdefault(fc, {})
                user_top_pois_by_fc[fc][tok] = user_top_pois_by_fc[fc].get(tok, 0) + 1

                if prev_fc is not None:
                    user_fc_bigram_counts.setdefault(prev_fc, {})
                    user_fc_bigram_counts[prev_fc][fc] = user_fc_bigram_counts[prev_fc].get(fc, 0) + 1
                prev_fc = fc

                user_top_pois_by_supercat.setdefault(sc, {})
                user_top_pois_by_supercat[sc][tok] = user_top_pois_by_supercat[sc].get(tok, 0) + 1

                user_poi_counts_by_sc.setdefault(sc, {})
                user_poi_counts_by_sc[sc][tok] = user_poi_counts_by_sc[sc].get(tok, 0) + 1

                sc_seq.append(sc)

                if fc.lower() == "home" or "home" in fc.lower():
                    home_like[tok] = home_like.get(tok, 0) + 1

                if i > 0:
                    prev_m = hhmmss_to_minute_of_day(r.visits[i - 1].time_hhmmss)
                    cur_m = hhmmss_to_minute_of_day(v.time_hhmmss)
                    gap = max(1, cur_m - prev_m)
                    gaps_by_dt[dt_type].append(gap)
                    global_gaps[dt_type].append(gap)

            if sc_seq:
                add_sc_count(initial_sc, sc_seq[0])
                add_sc_count(global_initial_sc, sc_seq[0])
            for i in range(1, len(sc_seq)):
                k = (sc_seq[i - 1], sc_seq[i])
                trans[k] = trans.get(k, 0) + 1
                global_trans[k] = global_trans.get(k, 0) + 1

        topk = 5
        if home_like:
            anchors_sorted = sorted(home_like.items(), key=lambda x: (-x[1], x[0]))[:topk]
        else:
            anchors_sorted = sorted(poi_counts.items(), key=lambda x: (-x[1], x[0]))[:topk]
        anchor_pois = [p for p, _ in anchors_sorted]
        anchor_weights = [float(c) for _p, c in anchors_sorted]
        if sum(anchor_weights) <= 0:
            anchor_weights = [1.0 for _ in anchor_weights]

        start_hist_by_dt: Dict[str, List[float]] = {}
        for dt_type in ("weekday", "weekend"):
            src = start_times_by_dt[dt_type] if start_times_by_dt[dt_type] else global_start_times[dt_type]
            start_hist_by_dt[dt_type] = _time_hist(src, TIME_BIN_MINUTES)

        stop_pmf_by_dt: Dict[str, Dict[int, float]] = {}
        for dt_type in ("weekday", "weekend"):
            pmf = _pmf_from_counts(stop_counts_by_dt[dt_type])
            if not pmf:
                pmf = _pmf_from_counts(global_stop_counts[dt_type])
            stop_pmf_by_dt[dt_type] = pmf

        gaps_samples_by_dt: Dict[str, List[int]] = {}
        for dt_type in ("weekday", "weekend"):
            xs = gaps_by_dt[dt_type] if gaps_by_dt[dt_type] else global_gaps[dt_type]
            gaps_samples_by_dt[dt_type] = [max(1, int(x)) for x in xs] if xs else [30]

        init_pmf = (
            normalize_counter({k: float(v) for k, v in initial_sc.items()}, epsilon=1e-6)
            if initial_sc
            else normalize_counter({k: float(v) for k, v in global_initial_sc.items()}, epsilon=1e-6)
        )
        init_smoothed = {sc: float(init_pmf.get(sc, 0.0)) + 1e-6 for sc in all_supercats}
        init_pmf = normalize_counter(init_smoothed, epsilon=0.0)

        row_counts: Dict[str, Dict[str, float]] = {}
        for (a, b), c2 in trans.items():
            row_counts.setdefault(a, {})
            row_counts[a][b] = row_counts[a].get(b, 0.0) + float(c2)

        global_row_counts: Dict[str, Dict[str, float]] = {}
        for (a, b), c2 in global_trans.items():
            global_row_counts.setdefault(a, {})
            global_row_counts[a][b] = global_row_counts[a].get(b, 0.0) + float(c2)

        trans_pmf: Dict[str, Dict[str, float]] = {}
        for a in all_supercats:
            base = row_counts.get(a)
            if not base:
                base = global_row_counts.get(a, {})
            smoothed = {b: float(base.get(b, 0.0)) + 1.0 for b in all_supercats}
            trans_pmf[a] = normalize_counter(smoothed, epsilon=0.0)

        total_p = float(sum(poi_counts.values()))
        poi_pref = {p: (c2 / total_p) for p, c2 in poi_counts.items()} if total_p > 0 else {}

        total_sc = float(sum(supercat_counts.values()))
        supercat_pref = {sc: (c2 / total_sc) for sc, c2 in supercat_counts.items()} if total_sc > 0 else {}

        finecat_pref_by_sc: Dict[str, Dict[str, float]] = {}
        for sc in all_supercats:
            counts_sc = user_fc_counts_by_sc.get(sc, {}) or {}
            if counts_sc:
                s = float(sum(counts_sc.values()))
                finecat_pref_by_sc[sc] = {fc: float(c) / s for fc, c in counts_sc.items()} if s > 0 else {}
            else:
                finecat_pref_by_sc[sc] = {}

        finecat_transition_sparse: Dict[str, Dict[str, float]] = {}
        for prev_fc, nxt_counts in user_fc_bigram_counts.items():
            top = sorted(nxt_counts.items(), key=lambda x: (-x[1], x[0]))[:60]
            total = float(sum(c for _fc, c in top))
            if total > 0:
                finecat_transition_sparse[prev_fc] = {fc: float(c) / total for fc, c in top}

        baseline = BaselineModel(
            anchor_pois=anchor_pois,
            anchor_weights=anchor_weights,
            start_time_hist_by_daytype=start_hist_by_dt,
            stop_count_pmf_by_daytype=stop_pmf_by_dt,
            gap_minutes_samples_by_daytype=gaps_samples_by_dt,
            initial_supercat_pmf=init_pmf,
            transition_pmf=trans_pmf,
            poi_pref=poi_pref,
            supercat_pref=supercat_pref,
            finecat_pref_by_supercat=finecat_pref_by_sc,
            finecat_transition_sparse=finecat_transition_sparse,
        )
        resident = Resident(user_id=user_id, baseline=baseline)
        resident.user_top_pois_by_supercat_counts = user_top_pois_by_supercat

        resident.user_poi_pref_by_supercat = {}
        for sc in all_supercats:
            counts_sc = user_poi_counts_by_sc.get(sc, {})
            s = float(sum(counts_sc.values()))
            resident.user_poi_pref_by_supercat[sc] = {tok: float(c) / s for tok, c in counts_sc.items()} if s > 0 else {}

        resident.user_top_pois_by_finecat_counts = user_top_pois_by_fc
        resident.user_poi_pref_by_finecat = {}
        for fc, counts_fc in user_top_pois_by_fc.items():
            s = float(sum(counts_fc.values()))
            resident.user_poi_pref_by_finecat[fc] = {tok: float(c) / s for tok, c in counts_fc.items()} if s > 0 else {}

        residents[user_id] = resident

    require(len(residents) > 0, "No residents had any 2019-2020 training data; cannot fit baseline models.")

    global_supercats_sorted = [sc for sc, _c in sorted(global_supercat_counts.items(), key=lambda x: (-x[1], x[0]))]
    tuned_supercats: List[str] = []
    tuned_supercats_limit = max(1, int(tuned_supercats_limit))
    for sc in global_supercats_sorted:
        if sc not in tuned_supercats:
            tuned_supercats.append(sc)
        if len(tuned_supercats) >= tuned_supercats_limit:
            break
    if "Unknown" not in tuned_supercats:
        tuned_supercats.append("Unknown")

    sim = MobilitySimulator(
        dataset=dataset,
        residents=residents,
        global_popular_pois=global_popular_pois,
        global_step_median_km=global_step_median_km,
        tuned_supercats=tuned_supercats,
        seed=seed,
    )
    sim.global_poi_pref = global_poi_pref  # type: ignore[attr-defined]
    sim.global_poi_pref_by_supercat = global_poi_pref_by_sc  # type: ignore[attr-defined]
    sim.global_popular_pois_by_supercat = global_popular_pois_by_supercat  # type: ignore[attr-defined]
    sim.global_poi_pref_by_finecat = global_poi_pref_by_fc  # type: ignore[attr-defined]
    sim.global_popular_pois_by_finecat = global_popular_pois_by_finecat  # type: ignore[attr-defined]
    sim.global_finecat_pref_by_supercat = global_finecat_pref_by_sc  # type: ignore[attr-defined]
    sim.global_finecat_bigram_counts = global_finecat_bigram_counts  # type: ignore[attr-defined]

    log_info(f"Built residents={len(residents)}; tuned_supercats={tuned_supercats}")
    return sim, residents


class Resident:
    def __init__(self, user_id: str, baseline: BaselineModel):
        self.user_id = user_id
        self.baseline = baseline
        self.user_top_pois_by_supercat_counts: Dict[str, Dict[str, int]] = {}
        self.user_poi_pref_by_supercat: Dict[str, Dict[str, float]] = {}

        self.user_top_pois_by_finecat_counts: Dict[str, Dict[str, int]] = {}
        self.user_poi_pref_by_finecat: Dict[str, Dict[str, float]] = {}

    def _anchor_with_coords(self, sim: "MobilitySimulator") -> Optional[str]:
        for tok in self.baseline.anchor_pois:
            info = sim.lookup_poi(tok)
            if info and info.lat is not None and info.lon is not None:
                return info.poi_token
        return None

    @staticmethod
    def _tilt_count_pmf_to_target_mean(pmf: Dict[int, float], target_mean: float) -> Dict[int, float]:
        items = sorted(pmf.keys())
        require(items, "Cannot tilt an empty PMF.")

        base = {int(k): max(1e-15, float(pmf[k])) for k in items}
        min_c, max_c = float(min(items)), float(max(items))
        target_mean = float(max(min_c, min(max_c, target_mean)))

        def mean_for_lambda(lam: float) -> float:
            ws = [base[c] * math.exp(lam * float(c)) for c in items]
            z = sum(ws)
            if z <= 0:
                return float(sum(items) / len(items))
            return sum(float(c) * w for c, w in zip(items, ws)) / z

        lo, hi = -3.0, 3.0
        m_lo, m_hi = mean_for_lambda(lo), mean_for_lambda(hi)
        if target_mean <= m_lo:
            lam_star = lo
        elif target_mean >= m_hi:
            lam_star = hi
        else:
            lam_star = 0.0
            for _ in range(40):
                mid = 0.5 * (lo + hi)
                m_mid = mean_for_lambda(mid)
                if m_mid < target_mean:
                    lo = mid
                else:
                    hi = mid
                lam_star = 0.5 * (lo + hi)

        ws = [base[c] * math.exp(lam_star * float(c)) for c in items]
        z = sum(ws)
        if z <= 0:
            return {int(c): 1.0 / len(items) for c in items}
        return {int(c): float(w) / z for c, w in zip(items, ws)}

    @staticmethod
    def _repeat_penalty(count: int) -> float:
        """Penalty multiplier for repeat visits to the same POI within a day."""
        if count <= 0:
            return 1.0
        return float(0.25 ** min(4, int(count)))

    def simulate_day(
        self,
        date_obj: dt.date,
        shift_params: Dict[str, Any],
        sim: "MobilitySimulator",
        context_last_poi: Optional[str],
        rng: "DeterministicRNG",
    ) -> List[Visit]:
        dt_type = day_type(date_obj)

        ctx_tok: Optional[str] = None
        if context_last_poi is not None:
            ctx_tok = sim.resolve_poi_token(context_last_poi)

        ctx = sim.context_features(self.user_id, date_obj, context_days=7)
        ctx_top_any = ctx.get("top_pois_any", [])
        ctx_fc_counts_by_sc = ctx.get("finecat_counts_by_supercat", {})
        ctx_fc_probs_by_sc = ctx.get("finecat_probs_by_supercat", {})
        ctx_poi_probs_by_fc = ctx.get("poi_probs_by_finecat", {})
        ctx_top_pois_by_fc = ctx.get("top_pois_by_finecat", {})
        ctx_avg_visits = ctx.get("avg_visits_per_day")

        start_poi: str
        if ctx_tok is not None:
            start_poi = ctx_tok
        elif ctx_top_any:
            start_poi = str(ctx_top_any[0])
        elif self.baseline.anchor_pois:
            start_poi = weighted_choice(self.baseline.anchor_pois, self.baseline.anchor_weights, rng.u())
        else:
            start_poi = sim.global_popular_pois[0]
        start_poi = sim.resolve_poi_token(start_poi) or start_poi

        base_pmf = self.baseline.stop_count_pmf_by_daytype.get(dt_type, {})
        if not base_pmf:
            base_pmf = self.baseline.stop_count_pmf_by_daytype.get("weekday", {}) or {1: 1.0}

        stop_mult = float(shift_params.get("theta_stop_count_multiplier", 1.0))
        stop_mult = max(0.5, min(1.8, stop_mult))

        mean_base = sum(float(c) * float(p) for c, p in base_pmf.items())
        if isinstance(ctx_avg_visits, (int, float)) and math.isfinite(float(ctx_avg_visits)) and float(ctx_avg_visits) > 0:
            mean_ctx = float(ctx_avg_visits)
            mean_ctx = max(1.0, min(mean_ctx, mean_base * 2.5))
            blended_mean = 0.75 * mean_base + 0.25 * mean_ctx
        else:
            blended_mean = mean_base

        target = max(1.0, blended_mean * stop_mult)

        tilted = self._tilt_count_pmf_to_target_mean(base_pmf, target)
        counts = sorted(tilted.keys())
        weights = [float(tilted[c]) for c in counts]
        stop_count = int(weighted_choice(counts, weights, rng.u()))
        stop_count = max(1, stop_count)

        start_shift = int(shift_params.get("theta_start_time_shift_minutes", 0))
        start_shift = max(-120, min(120, start_shift))

        start_hist = self.baseline.start_time_hist_by_daytype.get(
            dt_type, self.baseline.start_time_hist_by_daytype.get("weekday") or []
        )
        if not start_hist:
            start_hist = _time_hist([8 * 60], TIME_BIN_MINUTES)

        bins = list(range(len(start_hist)))
        b = int(weighted_choice(bins, start_hist, rng.u()))
        lo = b * TIME_BIN_MINUTES
        hi = min(MINUTES_PER_DAY - 1, lo + TIME_BIN_MINUTES - 1)
        start_min = lo + int(rng.u() * (hi - lo + 1))
        start_min = max(0, min(MINUTES_PER_DAY - 1, start_min + start_shift))

        end_min = MINUTES_PER_DAY - 1
        start_min = min(start_min, end_min - max(0, stop_count - 1))

        cat_mult = shift_params.get("theta_cat_weight_multiplier_by_supercategory", {})
        if not isinstance(cat_mult, dict):
            cat_mult = {}

        infra_bonus = float(shift_params.get("theta_infrastructure_stop_bonus", 0.0))
        infra_bonus = max(0.0, min(3.0, infra_bonus))

        init_pmf = self.baseline.initial_supercat_pmf
        sc_items = list(init_pmf.keys())
        sc_weights = []
        for sc in sc_items:
            m = float(cat_mult.get(sc, 1.0))
            m = max(0.25, min(4.0, m))
            bonus = math.exp(infra_bonus) if sc in sim.dataset.infra_supercats else 1.0
            sc_weights.append(float(init_pmf[sc]) * m * bonus)
        first_sc = weighted_choice(sc_items, sc_weights, rng.u())

        sc_seq = [first_sc]
        for _i in range(1, stop_count):
            prev = sc_seq[-1]
            row = self.baseline.transition_pmf.get(prev) or self.baseline.transition_pmf.get("Unknown")
            if not row:
                row = normalize_counter({sc: 1.0 for sc in sim.full_supercats}, epsilon=1e-6)
            next_items = list(row.keys())
            next_weights = []
            for sc in next_items:
                m = float(cat_mult.get(sc, 1.0))
                m = max(0.25, min(4.0, m))
                bonus = math.exp(infra_bonus) if sc in sim.dataset.infra_supercats else 1.0
                next_weights.append(float(row.get(sc, 0.0)) * m * bonus)
            sc_next = weighted_choice(next_items, next_weights, rng.u())
            sc_seq.append(sc_next)

        def sc_of_fc(fc: str) -> str:
            return sim.dataset.supercat_by_finecat.get(fc, "Unknown")

        finecats: List[str] = []
        prev_fc: Optional[str] = None
        for sc in sc_seq:
            ctx_fc_counts_sc: Dict[str, int] = dict(ctx_fc_counts_by_sc.get(sc, {}) or {})
            user_fc_pref_sc: Dict[str, float] = dict(self.baseline.finecat_pref_by_supercat.get(sc, {}) or {})
            global_fc_pref_sc: Dict[str, float] = dict(getattr(sim, "global_finecat_pref_by_supercat", {}).get(sc, {}) or {})

            ctx_fc_top = [fc for fc, _c in sorted(ctx_fc_counts_sc.items(), key=lambda x: (-x[1], x[0]))][:300]
            user_fc_top = [fc for fc, _p in sorted(user_fc_pref_sc.items(), key=lambda x: (-x[1], x[0]))][:400]
            global_fc_top = [fc for fc, _p in sorted(global_fc_pref_sc.items(), key=lambda x: (-x[1], x[0]))][:600]

            candidates_fc = list(dict.fromkeys(ctx_fc_top + user_fc_top + global_fc_top))
            if not candidates_fc:
                candidates_fc = global_fc_top if global_fc_top else ["Unknown"]

            ctx_fc_prob_sc: Dict[str, float] = dict(ctx_fc_probs_by_sc.get(sc, {}) or {})

            alpha_global = 0.15
            ctx_boost = 2.0
            base_w: List[float] = []
            for fc in candidates_fc:
                up = float(user_fc_pref_sc.get(fc, 0.0))
                gp = float(global_fc_pref_sc.get(fc, 0.0))
                cp = float(ctx_fc_prob_sc.get(fc, 0.0))
                base_w.append(up + alpha_global * gp + ctx_boost * cp + 1e-12)

            if prev_fc is not None:
                trans_user = self.baseline.finecat_transition_sparse.get(prev_fc, {})
                trans_global = getattr(sim, "global_finecat_bigram_counts", {}).get(prev_fc, {})
                trans_global_probs: Dict[str, float] = {}
                if trans_global:
                    top = sorted(trans_global.items(), key=lambda x: (-x[1], x[0]))[:80]
                    s = float(sum(c for _k, c in top))
                    if s > 0:
                        trans_global_probs = {k: float(c) / s for k, c in top}
                trans_boost = 0.8
                for i, fc in enumerate(candidates_fc):
                    tb = float(trans_user.get(fc, 0.0)) + 0.35 * float(trans_global_probs.get(fc, 0.0))
                    if tb > 0:
                        base_w[i] *= (1.0 + trans_boost * tb)

            chosen_fc = weighted_choice(candidates_fc, base_w, rng.u())
            if sc_of_fc(chosen_fc) != sc:
                consistent = [fc for fc in candidates_fc if sc_of_fc(fc) == sc]
                if consistent:
                    w2 = [base_w[candidates_fc.index(fc)] for fc in consistent]
                    chosen_fc = weighted_choice(consistent, w2, rng.u())
            finecats.append(chosen_fc)
            prev_fc = chosen_fc

        w_pref = float(shift_params.get("theta_preference_vs_distance_mixture", 0.6))
        w_pref = max(0.0, min(1.0, w_pref))

        dist_scale = float(shift_params.get("theta_distance_decay_scale", 1.0))
        dist_scale = max(0.5, min(2.5, dist_scale))
        decay_denom = max(1e-6, sim.global_step_median_km * dist_scale)

        p_anchor_reuse = 0.55
        if ctx_tok is not None or ctx_top_any:
            p_anchor_reuse = 0.70

        visits: List[Visit] = []
        current_poi = start_poi
        current_minute = start_min

        cur_lookup = sim.lookup_poi(current_poi)
        if cur_lookup is None or cur_lookup.lat is None or cur_lookup.lon is None:
            anchor = self._anchor_with_coords(sim)
            if anchor is not None:
                current_poi = anchor

        visited_counts: Dict[str, int] = {}

        for idx, (sc, fc) in enumerate(zip(sc_seq, finecats)):
            ctx_top_fc = list(ctx_top_pois_by_fc.get(fc, []) or [])[:250]
            ctx_prob_fc: Dict[str, float] = dict(ctx_poi_probs_by_fc.get(fc, {}) or {})

            user_counts_fc = self.user_top_pois_by_finecat_counts.get(fc, {})
            user_top_fc = [p for p, _c in sorted(user_counts_fc.items(), key=lambda x: (-x[1], x[0]))][:350]

            global_top_fc: List[str] = list(getattr(sim, "global_popular_pois_by_finecat", {}).get(fc, []) or [])[:800]

            catalog_fc = sim.dataset.poi_tokens_by_finecat.get(fc, [])
            catalog_sample: List[str] = []
            if catalog_fc:
                stride = 1 + (rng.state % 97)
                start_idx = rng.state % max(1, len(catalog_fc))
                for j in range(min(300, len(catalog_fc))):
                    catalog_sample.append(catalog_fc[(start_idx + j * stride) % len(catalog_fc)])

            candidates = list(dict.fromkeys(ctx_top_fc + user_top_fc + global_top_fc + catalog_sample))
            if not candidates:
                candidates = list(getattr(sim, "global_popular_pois_by_supercat", {}).get(sc, []) or [])[:800]
                if not candidates:
                    candidates = sim.global_popular_pois[:800]

            user_fc_pref = self.user_poi_pref_by_finecat.get(fc, {})
            global_fc_pref = getattr(sim, "global_poi_pref_by_finecat", {}).get(fc, {})

            chosen: Optional[str] = None

            def acceptable(tok: str) -> bool:
                c = visited_counts.get(tok, 0)
                if c >= 2 and len(candidates) > 10:
                    return False
                return True

            if ctx_top_fc and rng.u() < 0.50:
                ct_items = ctx_top_fc
                ct_weights = [
                    (float(ctx_prob_fc.get(tok, 0.0)) + 1e-9) * self._repeat_penalty(visited_counts.get(tok, 0))
                    for tok in ct_items
                ]
                for _try in range(6):
                    cand = weighted_choice(ct_items, ct_weights, rng.u())
                    if acceptable(cand) and (idx == 0 or cand != current_poi):
                        chosen = cand
                        break
            if chosen is None and user_top_fc and rng.u() < p_anchor_reuse:
                ut_items = user_top_fc
                ut_weights = [
                    (float(user_fc_pref.get(tok, 0.0)) + 1e-9) * self._repeat_penalty(visited_counts.get(tok, 0))
                    for tok in ut_items
                ]
                for _try in range(6):
                    cand = weighted_choice(ut_items, ut_weights, rng.u())
                    if acceptable(cand) and (idx == 0 or cand != current_poi):
                        chosen = cand
                        break

            if chosen is None:
                alpha_global = 0.08
                ctx_boost = 2.2

                pref_w: List[float] = []
                for tok in candidates:
                    up = float(user_fc_pref.get(tok, 0.0))
                    gp = float(global_fc_pref.get(tok, 0.0))
                    cp = float(ctx_prob_fc.get(tok, 0.0))
                    w0 = up + alpha_global * gp + ctx_boost * cp + 1e-12
                    w0 *= self._repeat_penalty(visited_counts.get(tok, 0))
                    pref_w.append(w0)

                cur_info = sim.lookup_poi(current_poi)
                dist_w: List[float] = []
                if cur_info and cur_info.lat is not None and cur_info.lon is not None:
                    for tok in candidates:
                        info = sim.lookup_poi(tok)
                        if info and info.lat is not None and info.lon is not None:
                            d_km = haversine_km(cur_info.lat, cur_info.lon, info.lat, info.lon)
                            dist_w.append(math.exp(-d_km / decay_denom) + 1e-12)
                        else:
                            dist_w.append(1e-6)
                else:
                    dist_w = [1.0 for _ in candidates]

                mix_w = [(w_pref * pw + (1.0 - w_pref) * dw) for pw, dw in zip(pref_w, dist_w)]

                for _try in range(10):
                    cand = weighted_choice(candidates, mix_w, rng.u())
                    if acceptable(cand) and (idx == 0 or cand != current_poi):
                        chosen = cand
                        break
                if chosen is None:
                    chosen = current_poi

            if idx == 0:
                t_min = current_minute
            else:
                gap_samples = self.baseline.gap_minutes_samples_by_daytype.get(dt_type, [30])
                gap = int(gap_samples[int(rng.u() * len(gap_samples))])
                gap = max(1, gap)

                min_travel = 1
                a = sim.lookup_poi(current_poi)
                b_info = sim.lookup_poi(chosen)
                if (
                    a
                    and b_info
                    and a.lat is not None
                    and a.lon is not None
                    and b_info.lat is not None
                    and b_info.lon is not None
                ):
                    d_km = haversine_km(a.lat, a.lon, b_info.lat, b_info.lon)
                    min_travel = max(1, int(math.ceil((d_km / 30.0) * 60.0)) + 1)

                remaining_after_this = (stop_count - 1) - idx
                budget_remaining = max(0, end_min - current_minute)

                denom = max(1, remaining_after_this + 1)
                avg_budget_gap = max(1, int(budget_remaining / denom))
                max_allowed_gap = max(1, int(avg_budget_gap * 1.8))
                gap = min(gap, max_allowed_gap)

                gap = max(gap, min_travel)
                proposed = current_minute + gap

                latest_allowed = end_min - max(0, remaining_after_this)
                t_min = min(proposed, latest_allowed)
                t_min = max(t_min, current_minute + 1)

            sec = int(rng.u() * 60)
            visits.append(Visit(poi_token=chosen, time_hhmmss=minute_of_day_to_hhmmss(t_min, sec)))
            visited_counts[chosen] = visited_counts.get(chosen, 0) + 1
            current_poi = chosen
            current_minute = t_min

        if not visits:
            sec = int(rng.u() * 60)
            visits = [Visit(poi_token=start_poi, time_hhmmss=minute_of_day_to_hhmmss(start_min, sec))]

        return visits


class DeterministicRNG:
    """Small deterministic RNG used for all sampling in the simulator."""

    def __init__(self, seed: int):
        self.state = seed & 0xFFFFFFFF

    def u(self) -> float:
        self.state = (1664525 * self.state + 1013904223) & 0xFFFFFFFF
        return self.state / 2**32


class MobilitySimulator:
    def __init__(
        self,
        dataset: ParsedDataset,
        residents: Dict[str, Resident],
        global_popular_pois: List[str],
        global_step_median_km: float,
        tuned_supercats: List[str],
        seed: int,
    ):
        self.dataset = dataset
        self.residents = residents
        self.global_popular_pois = global_popular_pois
        self.global_step_median_km = float(global_step_median_km)
        self.tuned_supercats = tuned_supercats
        self.seed = int(seed)
        self.full_supercats = sorted(set(dataset.poi_tokens_by_supercat.keys()) | {"Unknown"})
        self._context_cache: Dict[Tuple[str, dt.date, int], Dict[str, Any]] = {}

    def resolve_poi_token(self, token: str) -> Optional[str]:
        if not isinstance(token, str):
            token = str(token)
        t = normalize_poi_token(token)
        if t in self.dataset.poi_by_token:
            return t
        out = self.dataset.poi_token_by_normalized.get(t)
        if out is not None:
            return out
        return self.dataset.poi_token_by_casefold.get(t.casefold())

    def lookup_poi(self, token: str) -> Optional[POIInfo]:
        tok = self.resolve_poi_token(token) or normalize_poi_token(token)
        return self.dataset.poi_by_token.get(tok)

    def _context_last_poi(self, user_id: str, target_date: dt.date, context_days: int = 7) -> Optional[str]:
        records = self.dataset.day_records_by_user.get(user_id, [])
        if not records:
            return None

        start_date = target_date - dt.timedelta(days=context_days)
        candidates = [r for r in records if start_date <= r.date < target_date and len(r.visits) > 0]
        if candidates:
            candidates.sort(key=lambda r: r.date)
            return candidates[-1].visits[-1].poi_token if candidates[-1].visits else None

        max_backfill_days = 30
        start_date2 = target_date - dt.timedelta(days=max_backfill_days)
        candidates2 = [r for r in records if start_date2 <= r.date < target_date and len(r.visits) > 0]
        if not candidates2:
            return None
        candidates2.sort(key=lambda r: r.date)
        return candidates2[-1].visits[-1].poi_token if candidates2[-1].visits else None

    def context_features(self, user_id: str, target_date: dt.date, context_days: int = 7) -> Dict[str, Any]:
        """Compute cached context features from days strictly prior to target_date."""
        key = (str(user_id), target_date, int(context_days))
        cached = self._context_cache.get(key)
        if cached is not None:
            return cached

        records = self.dataset.day_records_by_user.get(str(user_id), [])
        if not records:
            out = {
                "top_pois_any": [],
                "counts_by_supercat": {},
                "probs_by_supercat": {},
                "finecat_counts_by_supercat": {},
                "finecat_probs_by_supercat": {},
                "top_pois_by_finecat": {},
                "poi_probs_by_finecat": {},
                "days_used": 0,
                "avg_visits_per_day": None,
            }
            self._context_cache[key] = out
            return out

        start_date = target_date - dt.timedelta(days=context_days)
        window = [r for r in records if start_date <= r.date < target_date and len(r.visits) > 0]
        if not window:
            max_backfill_days = 30
            start_date2 = target_date - dt.timedelta(days=max_backfill_days)
            window = [r for r in records if start_date2 <= r.date < target_date and len(r.visits) > 0]

        window.sort(key=lambda r: r.date)
        if len(window) > context_days:
            window = window[-context_days:]

        counts_any: Dict[str, int] = {}
        counts_by_sc: Dict[str, Dict[str, int]] = {}
        finecat_counts_by_sc: Dict[str, Dict[str, int]] = {}
        poi_counts_by_fc: Dict[str, Dict[str, int]] = {}

        total_visits = 0
        for r in window:
            for v in r.visits:
                tok = self.resolve_poi_token(v.poi_token) or normalize_poi_token(v.poi_token)
                counts_any[tok] = counts_any.get(tok, 0) + 1
                total_visits += 1

                poi = self.dataset.poi_by_token.get(tok)
                fc = poi.fine_category if poi else (tok.split("#", 1)[0] if "#" in tok else tok)
                sc = poi.super_category if poi else self.dataset.supercat_by_finecat.get(fc, "Unknown")

                counts_by_sc.setdefault(sc, {})
                counts_by_sc[sc][tok] = counts_by_sc[sc].get(tok, 0) + 1

                finecat_counts_by_sc.setdefault(sc, {})
                finecat_counts_by_sc[sc][fc] = finecat_counts_by_sc[sc].get(fc, 0) + 1

                poi_counts_by_fc.setdefault(fc, {})
                poi_counts_by_fc[fc][tok] = poi_counts_by_fc[fc].get(tok, 0) + 1

        top_any = [p for p, _c in sorted(counts_any.items(), key=lambda x: (-x[1], x[0]))][:300]

        probs_by_sc: Dict[str, Dict[str, float]] = {}
        for sc, m in counts_by_sc.items():
            s = float(sum(m.values()))
            probs_by_sc[sc] = {tok: (float(c) / s) for tok, c in m.items()} if s > 0 else {}

        finecat_probs_by_sc: Dict[str, Dict[str, float]] = {}
        for sc, m in finecat_counts_by_sc.items():
            s = float(sum(m.values()))
            finecat_probs_by_sc[sc] = {fc: (float(c) / s) for fc, c in m.items()} if s > 0 else {}

        top_pois_by_fc: Dict[str, List[str]] = {}
        poi_probs_by_fc: Dict[str, Dict[str, float]] = {}
        for fc, m in poi_counts_by_fc.items():
            top = [p for p, _c in sorted(m.items(), key=lambda x: (-x[1], x[0]))][:200]
            s = float(sum(m.values()))
            top_pois_by_fc[fc] = top
            poi_probs_by_fc[fc] = {p: float(m[p]) / s for p in top} if s > 0 else {}

        days_used = len(window)
        avg_visits_per_day = (float(total_visits) / float(days_used)) if days_used > 0 else None

        out = {
            "top_pois_any": top_any,
            "counts_by_supercat": counts_by_sc,
            "probs_by_supercat": probs_by_sc,
            "finecat_counts_by_supercat": finecat_counts_by_sc,
            "finecat_probs_by_supercat": finecat_probs_by_sc,
            "top_pois_by_finecat": top_pois_by_fc,
            "poi_probs_by_finecat": poi_probs_by_fc,
            "days_used": days_used,
            "avg_visits_per_day": avg_visits_per_day,
        }
        self._context_cache[key] = out
        return out

    def rollout(self, days_by_user: Dict[str, List[dt.date]], shift_params: Dict[str, Any], purpose: str) -> "RolloutResult":
        trajectories_strings: Dict[str, List[str]] = {}
        visits_by_user_day: Dict[Tuple[str, dt.date], List[Visit]] = {}

        total_days = sum(len(ds) for ds in days_by_user.values())
        log_info(f"Rollout '{purpose}': simulating user-days={total_days}")

        for user_id, dates in days_by_user.items():
            if not dates:
                continue
            resident = self.residents.get(user_id)
            if resident is None:
                continue
            out_strings: List[str] = []
            for d in sorted(dates):
                s = self.seed ^ stable_int_hash(f"{purpose}|{user_id}|{d.isoformat()}")
                rng = DeterministicRNG(s)
                ctx_poi = self._context_last_poi(user_id, d, context_days=7)
                visits = resident.simulate_day(d, shift_params, self, ctx_poi, rng)
                visits_by_user_day[(user_id, d)] = visits
                out_strings.append(format_day_string(d, visits))
            trajectories_strings[user_id] = out_strings

        meta = {"purpose": purpose, "seed": self.seed, "days_simulated": total_days}
        return RolloutResult(trajectories_strings=trajectories_strings, visits_by_user_day=visits_by_user_day, meta=meta)


def format_day_string(date_obj: dt.date, visits: List[Visit]) -> str:
    """Format a day trajectory string in the 1921Y.json canonical style."""
    prefix = f"Activities at {date_obj.isoformat()}:"
    if not visits:
        return prefix
    parts = [f"{v.poi_token} at {v.time_hhmmss}" for v in visits]
    return f"{prefix} " + ", ".join(parts) + "."


class Evaluator:
    def __init__(self, dataset: ParsedDataset, objective_weights: Dict[str, float], k_recall: int = 5):
        self.dataset = dataset
        self.objective_weights = dict(objective_weights)
        self.k_recall = int(k_recall)
        require(self.k_recall >= 1, "k_recall must be >= 1.")

    def _get_gt_visits(self, user_id: str, date_obj: dt.date) -> Optional[List[Visit]]:
        rec = self.dataset.day_record_lookup.get((user_id, date_obj))
        if rec is None:
            return None
        return rec.visits

    def _resolve_token(self, token: str) -> str:
        t = normalize_poi_token(token)
        out = self.dataset.poi_token_by_normalized.get(t)
        if out is not None:
            return out
        out = self.dataset.poi_token_by_casefold.get(t.casefold())
        return out if out is not None else t

    def compute_metrics(
        self, simulated: RolloutResult, days_by_user: Dict[str, List[dt.date]]
    ) -> Tuple[Dict[str, float], float, Dict[str, Any]]:
        pairs: List[Tuple[str, dt.date]] = []
        for u, ds in days_by_user.items():
            for d in ds:
                pairs.append((u, d))

        abs_errs: List[float] = []
        gt_counts: List[int] = []
        sim_counts: List[int] = []

        gt_fc_counts: Dict[str, int] = {}
        sim_fc_counts: Dict[str, int] = {}
        gt_sc_counts: Dict[str, int] = {}
        sim_sc_counts: Dict[str, int] = {}

        bins = MINUTES_PER_DAY // TIME_BIN_MINUTES
        gt_tod = {"weekday": [0.0] * bins, "weekend": [0.0] * bins}
        sim_tod = {"weekday": [0.0] * bins, "weekend": [0.0] * bins}
        gt_minutes: List[int] = []
        sim_minutes: List[int] = []

        recalls: List[float] = []
        gt_bigram: Dict[str, int] = {}
        sim_bigram: Dict[str, int] = {}

        paired_gt_dists: List[float] = []
        paired_sim_dists: List[float] = []
        gt_dists_marginal: List[float] = []
        sim_dists_marginal: List[float] = []
        gt_missing_coord_steps = 0
        sim_missing_coord_steps = 0
        gt_total_steps = 0
        sim_total_steps = 0

        rog_abs_errors: List[float] = []
        rog_days_scored = 0

        def fine_of_visit(v: Visit) -> str:
            tok = self._resolve_token(v.poi_token)
            return tok.split("#", 1)[0] if "#" in tok else tok

        def sc_of_visit(v: Visit) -> str:
            fine = fine_of_visit(v)
            return self.dataset.supercat_by_finecat.get(fine, "Unknown")

        def poi_info(v: Visit) -> Optional[POIInfo]:
            tok = self._resolve_token(v.poi_token)
            return self.dataset.poi_by_token.get(tok)

        def radius_of_gyration_km(visits: List[Visit]) -> Optional[float]:
            pts: List[Tuple[float, float]] = []
            for v in visits:
                info = poi_info(v)
                if info and info.lat is not None and info.lon is not None:
                    pts.append((info.lat, info.lon))
            if not pts:
                return None
            if len(pts) == 1:
                return 0.0

            lat0 = sum(p[0] for p in pts) / len(pts)
            cos0 = math.cos(math.radians(lat0))
            xs = [p[1] * cos0 * 111.320 for p in pts]
            ys = [p[0] * 110.574 for p in pts]
            x_cm = sum(xs) / len(xs)
            y_cm = sum(ys) / len(ys)
            msd = sum((x - x_cm) ** 2 + (y - y_cm) ** 2 for x, y in zip(xs, ys)) / float(len(xs))
            return float(math.sqrt(max(0.0, msd)))

        def add_step_dist_marginal(visits: List[Visit], dst: List[float], which: str) -> None:
            nonlocal gt_missing_coord_steps, sim_missing_coord_steps, gt_total_steps, sim_total_steps
            for i in range(1, len(visits)):
                a = poi_info(visits[i - 1])
                b = poi_info(visits[i])
                if which == "gt":
                    gt_total_steps += 1
                else:
                    sim_total_steps += 1
                if a and b and a.lat is not None and a.lon is not None and b.lat is not None and b.lon is not None:
                    dst.append(haversine_km(a.lat, a.lon, b.lat, b.lon))
                else:
                    if which == "gt":
                        gt_missing_coord_steps += 1
                    else:
                        sim_missing_coord_steps += 1

        def add_step_dist_paired(gt_visits: List[Visit], sim_visits: List[Visit]) -> None:
            n_steps = min(len(gt_visits), len(sim_visits)) - 1
            if n_steps <= 0:
                return
            for i in range(1, n_steps + 1):
                a_gt = poi_info(gt_visits[i - 1])
                b_gt = poi_info(gt_visits[i])
                a_sm = poi_info(sim_visits[i - 1])
                b_sm = poi_info(sim_visits[i])
                if (
                    a_gt
                    and b_gt
                    and a_sm
                    and b_sm
                    and a_gt.lat is not None
                    and a_gt.lon is not None
                    and b_gt.lat is not None
                    and b_gt.lon is not None
                    and a_sm.lat is not None
                    and a_sm.lon is not None
                    and b_sm.lat is not None
                    and b_sm.lon is not None
                ):
                    paired_gt_dists.append(haversine_km(a_gt.lat, a_gt.lon, b_gt.lat, b_gt.lon))
                    paired_sim_dists.append(haversine_km(a_sm.lat, a_sm.lon, b_sm.lat, b_sm.lon))

        for u, d in pairs:
            gt = self._get_gt_visits(u, d)
            simv = simulated.visits_by_user_day.get((u, d))

            if gt is None:
                continue
            if simv is None:
                simv = []

            gt_counts.append(len(gt))
            sim_counts.append(len(simv))
            abs_errs.append(abs(len(simv) - len(gt)))

            dt_type = day_type(d)
            for v in gt:
                fc = fine_of_visit(v)
                gt_fc_counts[fc] = gt_fc_counts.get(fc, 0) + 1
                sc = self.dataset.supercat_by_finecat.get(fc, "Unknown")
                gt_sc_counts[sc] = gt_sc_counts.get(sc, 0) + 1

                m = hhmmss_to_minute_of_day(v.time_hhmmss)
                gt_tod[dt_type][m // TIME_BIN_MINUTES] += 1.0
                gt_minutes.append(m)

            for v in simv:
                fc = fine_of_visit(v)
                sim_fc_counts[fc] = sim_fc_counts.get(fc, 0) + 1
                sc = self.dataset.supercat_by_finecat.get(fc, "Unknown")
                sim_sc_counts[sc] = sim_sc_counts.get(sc, 0) + 1

                m = hhmmss_to_minute_of_day(v.time_hhmmss)
                sim_tod[dt_type][m // TIME_BIN_MINUTES] += 1.0
                sim_minutes.append(m)

            gt_set = set(self._resolve_token(v.poi_token) for v in gt)
            if gt_set:
                k_eff = min(self.k_recall, len(simv))
                sim_set = set(self._resolve_token(v.poi_token) for v in simv[:k_eff]) if k_eff > 0 else set()
                denom = float(min(len(gt_set), self.k_recall))
                rec = (len(gt_set.intersection(sim_set)) / denom) if denom > 0 else 0.0
                recalls.append(float(rec))

            gt_sc_seq = [sc_of_visit(v) for v in gt]
            sim_sc_seq = [sc_of_visit(v) for v in simv]
            for i in range(1, len(gt_sc_seq)):
                key = f"{gt_sc_seq[i - 1]}->{gt_sc_seq[i]}"
                gt_bigram[key] = gt_bigram.get(key, 0) + 1
            for i in range(1, len(sim_sc_seq)):
                key = f"{sim_sc_seq[i - 1]}->{sim_sc_seq[i]}"
                sim_bigram[key] = sim_bigram.get(key, 0) + 1

            add_step_dist_marginal(gt, gt_dists_marginal, "gt")
            add_step_dist_marginal(simv, sim_dists_marginal, "sim")
            add_step_dist_paired(gt, simv)

            rg_gt = radius_of_gyration_km(gt)
            rg_sim = radius_of_gyration_km(simv)
            if rg_gt is not None and rg_sim is not None:
                rog_abs_errors.append(abs(rg_sim - rg_gt))
                rog_days_scored += 1

        stop_count_abs_mean_error = float(sum(abs_errs) / max(1, len(abs_errs)))

        def hist_counts(xs: List[int]) -> Dict[str, float]:
            h: Dict[str, float] = {}
            for x in xs:
                k = str(int(x))
                h[k] = h.get(k, 0.0) + 1.0
            return normalize_counter(h, epsilon=1e-6) if h else {"0": 1.0}

        gt_pmf = hist_counts(gt_counts)
        sim_pmf = hist_counts(sim_counts)
        stop_count_kl = float(kl_divergence(gt_pmf, sim_pmf, epsilon=1e-12))

        gt_fc_share = (
            normalize_counter({k: float(v) for k, v in gt_fc_counts.items()}, epsilon=1e-9)
            if gt_fc_counts
            else {"Unknown": 1.0}
        )
        sim_fc_share = (
            normalize_counter({k: float(v) for k, v in sim_fc_counts.items()}, epsilon=1e-9)
            if sim_fc_counts
            else {"Unknown": 1.0}
        )
        category_share_mae = float(shares_mae(gt_fc_share, sim_fc_share))
        category_mix_jsd_fine = float(js_divergence(gt_fc_share, sim_fc_share, epsilon=1e-12))

        gt_sc_share = (
            normalize_counter({k: float(v) for k, v in gt_sc_counts.items()}, epsilon=1e-9)
            if gt_sc_counts
            else {"Unknown": 1.0}
        )
        sim_sc_share = (
            normalize_counter({k: float(v) for k, v in sim_sc_counts.items()}, epsilon=1e-9)
            if sim_sc_counts
            else {"Unknown": 1.0}
        )
        category_mix_jsd_supercat = float(js_divergence(gt_sc_share, sim_sc_share, epsilon=1e-12))

        def tod_dist(hist: List[float]) -> Dict[str, float]:
            total = sum(hist)
            if total <= 0:
                return {str(i): 1.0 / len(hist) for i in range(len(hist))}
            return {str(i): (hist[i] + 1e-6) / (total + 1e-6 * len(hist)) for i in range(len(hist))}

        jsds = []
        for dt_type in ("weekday", "weekend"):
            if sum(gt_tod[dt_type]) > 0 or sum(sim_tod[dt_type]) > 0:
                jsds.append(js_divergence(tod_dist(gt_tod[dt_type]), tod_dist(sim_tod[dt_type]), epsilon=1e-12))
        tod_jsd_avg = float(sum(jsds) / max(1, len(jsds)))

        topk_poi_recall = float(sum(recalls) / max(1, len(recalls)))

        gt_bigram_p = (
            normalize_counter({k: float(v) for k, v in gt_bigram.items()}, epsilon=1e-9)
            if gt_bigram
            else {"None": 1.0}
        )
        sim_bigram_p = (
            normalize_counter({k: float(v) for k, v in sim_bigram.items()}, epsilon=1e-9)
            if sim_bigram
            else {"None": 1.0}
        )
        transition_divergence = float(js_divergence(gt_bigram_p, sim_bigram_p, epsilon=1e-12))

        if paired_gt_dists and paired_sim_dists:
            trip_distance_wasserstein = float(wasserstein_1d(paired_gt_dists, paired_sim_dists, quantiles=200))
        else:
            trip_distance_wasserstein = float(wasserstein_1d(gt_dists_marginal, sim_dists_marginal, quantiles=200))

        simulation_metrics = {
            "category_share_mae": category_share_mae,
            "stop_count_abs_mean_error": stop_count_abs_mean_error,
            "stop_count_kl": stop_count_kl,
            "tod_jsd_avg": tod_jsd_avg,
            "topk_poi_recall": topk_poi_recall,
            "transition_divergence": transition_divergence,
            "trip_distance_wasserstein": trip_distance_wasserstein,
        }

        objective = self.objective(simulation_metrics)

        diagnostics = {
            "pairs_scored": len(abs_errs),
            "k_recall": self.k_recall,
            "tod_bin_minutes": TIME_BIN_MINUTES,
        }
        return simulation_metrics, float(objective), diagnostics

    def objective(self, metrics: Mapping[str, float]) -> float:
        w = self.objective_weights
        require("topk_poi_recall" in metrics, "Missing required metric topk_poi_recall.")
        obj = 0.0
        for k, weight in w.items():
            if k == "topk_poi_recall":
                obj += float(weight) * (1.0 - float(metrics[k]))
            else:
                obj += float(weight) * float(metrics[k])
        return float(obj)


class Calibrator(abc.ABC):
    @abc.abstractmethod
    def fit(
        self, simulator: MobilitySimulator, evaluator: Evaluator, calib_days_by_user: Dict[str, List[dt.date]]
    ) -> Tuple[Dict[str, Any], float, List[Dict[str, Any]]]:
        raise NotImplementedError


class RandomSearchCalibrator(Calibrator):
    def __init__(self, seed: int, n_iters: int, tuned_supercats: List[str], notes: str = ""):
        require(n_iters >= 1, "n_iters must be >= 1.")
        self.seed = int(seed)
        self.n_iters = int(n_iters)
        self.tuned_supercats = list(tuned_supercats)
        self.notes = str(notes)

    def _sample_params(self, rng: DeterministicRNG) -> Dict[str, Any]:
        def ru(lo: float, hi: float) -> float:
            return lo + (hi - lo) * rng.u()

        params: Dict[str, Any] = {
            "theta_stop_count_multiplier": ru(0.5, 1.8),
            "theta_start_time_shift_minutes": int(round(ru(-120.0, 120.0))),
            "theta_distance_decay_scale": ru(0.5, 2.5),
            "theta_preference_vs_distance_mixture": ru(0.0, 1.0),
            "theta_infrastructure_stop_bonus": ru(0.0, 3.0),
        }
        cat_mult: Dict[str, float] = {}
        for sc in self.tuned_supercats:
            cat_mult[sc] = ru(0.25, 4.0)
        params["theta_cat_weight_multiplier_by_supercategory"] = cat_mult
        return params

    def fit(
        self, simulator: MobilitySimulator, evaluator: Evaluator, calib_days_by_user: Dict[str, List[dt.date]]
    ) -> Tuple[Dict[str, Any], float, List[Dict[str, Any]]]:
        log: List[Dict[str, Any]] = []
        best_params: Optional[Dict[str, Any]] = None
        best_obj: float = float("inf")

        search_rng = DeterministicRNG(self.seed ^ stable_int_hash("calibration-search"))

        for it in range(self.n_iters):
            params = self._sample_params(search_rng)
            rollout = simulator.rollout(calib_days_by_user, params, purpose=f"calib_iter_{it}")
            metrics, obj, diag = evaluator.compute_metrics(rollout, calib_days_by_user)

            log_item = {
                "iter": int(it),
                "parameters": params,
                "objective": float(obj),
                "metrics": metrics,
                "notes": self.notes,
            }
            log.append(log_item)

            if obj < best_obj:
                best_obj = float(obj)
                best_params = params
                log_info(f"Calibration iter={it}: improved objective={best_obj:.6f}")

            if it % max(1, self.n_iters // 5) == 0:
                log_info(
                    f"Calibration progress: iter={it}/{self.n_iters} objective={obj:.6f} pairs={diag.get('pairs_scored')}"
                )

        require(best_params is not None, "Calibration failed to produce any parameter set.")
        return best_params, best_obj, log


def save_json(path: str, obj: Any) -> None:
    require(os.path.isabs(path), f"Output path must be absolute: {path}")
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)
    except Exception as e:
        raise RuntimeError(f"Failed to write JSON output: {path}") from e


def save_results(
    output_dir: str,
    calibrated_parameters: Dict[str, Any],
    calibration_log: List[Dict[str, Any]],
    evaluation_results: Dict[str, Any],
    simulated_trajectories_validation: Dict[str, Any],
) -> None:
    require(os.path.isabs(output_dir), "--output_dir must resolve to an absolute path at save time.")
    os.makedirs(output_dir, exist_ok=True)

    save_json(os.path.join(output_dir, "calibrated_parameters.json"), calibrated_parameters)
    save_json(os.path.join(output_dir, "calibration_log.json"), calibration_log)
    save_json(os.path.join(output_dir, "evaluation_results_on_validation.json"), evaluation_results)
    save_json(os.path.join(output_dir, "simulated_trajectories_validation.json"), simulated_trajectories_validation)


def parse_cli(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Mobility simulator with calibration and evaluation.")
    p.add_argument("--output_dir", required=True, type=str, help="Directory to write outputs under (created if missing).")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Global random seed (deterministic).")
    p.add_argument("--calibration_iters", type=int, default=15, help="Number of calibration trials (random search).")
    p.add_argument(
        "--sparse_2021_handling",
        type=str,
        default="exclude",
        choices=["exclude", "test_only"],
        help="How to handle users with <5 2021 records.",
    )
    p.add_argument("--k_recall", type=int, default=5, help="K for top-k POI recall metric.")
    p.add_argument(
        "--tuned_supercats_limit", type=int, default=6, help="How many supercategories to tune multipliers for."
    )
    return p.parse_args(argv)


def _failure_outputs(
    seed: int, objective_weights: Dict[str, float], error: str
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    calibrated_parameters = {
        "best_parameters": {},
        "best_objective": float("nan"),
        "objective_definition": "Failed to run calibration/evaluation due to an error.",
        "seed": int(seed),
        "meta": {"error": str(error)},
    }
    calibration_log: List[Dict[str, Any]] = []
    evaluation_results = {
        "simulation_metrics": {
            "category_share_mae": float("nan"),
            "stop_count_abs_mean_error": float("nan"),
            "stop_count_kl": float("nan"),
            "tod_jsd_avg": float("nan"),
            "topk_poi_recall": float("nan"),
            "transition_divergence": float("nan"),
            "trip_distance_wasserstein": float("nan"),
        },
        "objective": float("nan"),
        "objective_weights": objective_weights,
        "validation_set": {
            "users": 0,
            "user_days": 0,
            "split": "per_user_last_20pct_of_2021",
            "sparse_2021_handling": None,
        },
        "meta": {"seed": int(seed), "error": str(error)},
    }
    simulated_trajectories_validation = {
        "format_spec": TrajectoryStringFormatter.FORMAT_SPEC,
        "trajectories": {},
        "meta": {"seed": int(seed), "purpose": "failed", "days_simulated": 0, "error": str(error)},
    }
    return calibrated_parameters, calibration_log, evaluation_results, simulated_trajectories_validation


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_cli(argv)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    seed = int(args.seed)
    random.seed(seed)

    objective_weights = {
        "category_share_mae": 1.0,
        "stop_count_abs_mean_error": 0.8,
        "stop_count_kl": 0.4,
        "tod_jsd_avg": 1.0,
        "topk_poi_recall": 1.2,
        "transition_divergence": 0.6,
        "trip_distance_wasserstein": 0.4,
    }

    try:
        dataset = load_data()
        provisional_splits = HoldoutSplits(
            train_users=[],
            calib_days_by_user={},
            test_days_by_user={},
            excluded_sparse_users=[],
            sparse_handling=args.sparse_2021_handling,
        )
        simulator, _residents = build_network_and_agents(
            dataset=dataset,
            splits=provisional_splits,
            seed=seed,
            tuned_supercats_limit=int(args.tuned_supercats_limit),
        )
        splits = holdout_split(dataset, sparse_2021_handling=args.sparse_2021_handling)

        evaluator = Evaluator(dataset=dataset, objective_weights=objective_weights, k_recall=int(args.k_recall))

        calibrator = RandomSearchCalibrator(
            seed=seed ^ stable_int_hash("calibrator"),
            n_iters=int(args.calibration_iters),
            tuned_supercats=simulator.tuned_supercats,
            notes=f"random_search; tuned_supercats={simulator.tuned_supercats}",
        )
        best_params, best_obj, calib_log = calibrator.fit(simulator, evaluator, splits.calib_days_by_user)

        validation_rollout = simulator.rollout(splits.test_days_by_user, best_params, purpose="validation_test_rollout")
        metrics_all, objective_value, diag = evaluator.compute_metrics(validation_rollout, splits.test_days_by_user)

        sim_metrics = {k: float(metrics_all[k]) for k in metrics_all.keys()}

        calibrated_parameters = {
            "best_parameters": best_params,
            "best_objective": float(best_obj),
            "objective_definition": (
                "Weighted sum to minimize: category_share_mae + stop_count_abs_mean_error + stop_count_kl + "
                "tod_jsd_avg + (1-topk_poi_recall) + transition_divergence + trip_distance_wasserstein (with weights)."
            ),
            "seed": seed,
            "meta": {
                "calibration_iters": int(args.calibration_iters),
                "tuned_supercats": simulator.tuned_supercats,
                "sparse_2021_handling": splits.sparse_handling,
                "excluded_sparse_users_count": len(splits.excluded_sparse_users),
                "aliases": {"best_params": best_params, "best_objective_on_training": float(best_obj)},
            },
        }

        evaluation_results = {
            "simulation_metrics": sim_metrics,
            "objective": float(objective_value),
            "objective_weights": objective_weights,
            "validation_set": {
                "split": "per_user_last_20pct_of_2021",
                "users": len([u for u, ds in splits.test_days_by_user.items() if ds]),
                "user_days": sum(len(ds) for ds in splits.test_days_by_user.values()),
                "sparse_2021_handling": splits.sparse_handling,
            },
            "meta": {
                "seed": seed,
                "diagnostics": diag,
                "evaluated_split": "V_test",
                "note": "Calibration used ONLY V_calib (first 80% of 2021 per user). This evaluation is on held-out V_test.",
            },
        }

        simulated_trajectories_validation = {
            "format_spec": TrajectoryStringFormatter.FORMAT_SPEC,
            "trajectories": validation_rollout.trajectories_strings,
            "meta": {
                "seed": seed,
                "purpose": validation_rollout.meta.get("purpose"),
                "days_simulated": validation_rollout.meta.get("days_simulated"),
            },
        }

        save_results(
            output_dir=output_dir,
            calibrated_parameters=calibrated_parameters,
            calibration_log=calib_log,
            evaluation_results=evaluation_results,
            simulated_trajectories_validation=simulated_trajectories_validation,
        )
    except Exception as e:
        log_info(f"ERROR: {e}")
        calibrated_parameters, calibration_log, evaluation_results, simulated_trajectories_validation = _failure_outputs(
            seed=seed, objective_weights=objective_weights, error=str(e)
        )
        save_results(
            output_dir=output_dir,
            calibrated_parameters=calibrated_parameters,
            calibration_log=calibration_log,
            evaluation_results=evaluation_results,
            simulated_trajectories_validation=simulated_trajectories_validation,
        )

    print(f"[RESULT] wrote outputs to: {output_dir}")


main()