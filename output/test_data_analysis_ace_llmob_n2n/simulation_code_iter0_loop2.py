from __future__ import annotations

"""
simulate.py

Production-grade, end-to-end multi-agent daily mobility trajectory simulator.

Implements:
- Absolute-path data ingestion from environment-defined DATA_DIR
- Strict 2019-2020 filtering (never uses 2021/out-of-range)
- Per-agent temporal holdout with eligibility constraint (>=7 prior days)
- Per-agent priors learned from TRAIN only
- Calibratable sequential generative simulator (Mode A)
- Pluggable black-box calibration (random search)
- Forward rollout on validation window
- Evaluation metrics and JSON outputs (exactly 4 files)

Run:
  python simulate.py --output_dir /abs/path/out --seed 123 --calib_iters 25
"""

import argparse
import hashlib
import json
import math
import os
import random
import re
import statistics
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore


# ----------------------------
# OpenAI LLM utilities (required)
# ----------------------------

def get_openai_api_key():
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        return api_key
    raise ValueError("OpenAI API key not found in environment")


def call_gpt5_with_responses_api(prompt: str, model: str = "gpt-5", max_output_tokens: int = 4000):
    api_key = get_openai_api_key()
    if OpenAI is None:
        raise ImportError(
            "openai package is not installed or failed to import. "
            "Install/upgrade 'openai' to a version that supports OpenAI(...) and client.responses.create(...)."
        )

    client = OpenAI(api_key=api_key)

    if not hasattr(client, "responses") or not hasattr(getattr(client, "responses", None), "create"):
        raise AttributeError(
            "Installed 'openai' SDK does not support client.responses.create(...). "
            "Upgrade 'openai' to a compatible version that provides the Responses API."
        )

    responses_kwargs = {
        "model": model,
        "input": [
            {"role": "user", "content": [{"type": "input_text", "text": prompt}]}
        ],
        "max_output_tokens": max_output_tokens,
    }

    resp = client.responses.create(**responses_kwargs)

    def extract_response(resp_obj):
        if hasattr(resp_obj, "output_text") and isinstance(resp_obj.output_text, str):
            return resp_obj.output_text
        try:
            output = getattr(resp_obj, "output", None)
            if output and isinstance(output, list) and len(output) > 0:
                first = output[0]
                content = first.get("content") if isinstance(first, dict) else None
                if content and isinstance(content, list) and len(content) > 0:
                    text = content[0].get("text") if isinstance(content[0], dict) else None
                    if isinstance(text, str):
                        return text
        except Exception:
            pass
        return str(resp_obj)

    return extract_response(resp)


# ----------------------------
# Optional: LLM "Reasoning Agent" pipeline (isolated, non-invasive)
# ----------------------------

class MemoryAgent:
    """Minimal memory agent to retrieve user/item context for the reasoning agent."""

    def retrieve_user_context(self, user_id: str) -> str:
        return f"user_id={user_id}"

    def retrieve_item_context(self, item_id: str) -> str:
        return f"item_id={item_id}"


class PlanningAgent:
    """Minimal planning agent to produce a plan/task decomposition."""

    def make_plan(self, task: str) -> List[str]:
        if not task:
            task = "write a review"
        return [
            f"Task: {task}",
            "1) Summarize key user preferences and constraints.",
            "2) Summarize key product/item attributes and tradeoffs.",
            "3) Draft review with balanced pros/cons and a final rating justification.",
        ]


class ReviewAuthor:
    """
    Reasoning agent: MUST perform its reasoning via an LLM call.
    The LLM output is the primary review text.
    """

    def __init__(self, memory_agent: MemoryAgent, planning_agent: PlanningAgent) -> None:
        self.memory_agent = memory_agent
        self.planning_agent = planning_agent

    def generate(self, *, user_id: str, item_id: str, task: str = "Write a helpful product review.") -> str:
        user_ctx = self.memory_agent.retrieve_user_context(user_id)
        item_ctx = self.memory_agent.retrieve_item_context(item_id)
        plan = self.planning_agent.make_plan(task)

        prompt = (
            "You are a review-writing assistant.\n\n"
            "USER CONTEXT (from Memory Agent):\n"
            f"{user_ctx}\n\n"
            "ITEM CONTEXT (from Memory Agent):\n"
            f"{item_ctx}\n\n"
            "PLAN / TASK DECOMPOSITION (from Planning Agent):\n"
            + "\n".join(plan)
            + "\n\n"
            "Write the review body text as the final output. Do not include headings like 'Review:' unless asked."
        )

        response = call_gpt5_with_responses_api(prompt=prompt, model="gpt-5", max_output_tokens=4000)
        return response.strip()


# ----------------------------
# Path Handling (deferred env validation)
# ----------------------------

PROJECT_ROOT: Optional[str] = os.environ.get("PROJECT_ROOT")
DATA_PATH: Optional[str] = os.environ.get("DATA_PATH")
DATA_DIR: Optional[str] = None


def _require_data_dir() -> str:
    global DATA_DIR, PROJECT_ROOT, DATA_PATH
    if DATA_DIR and os.path.isabs(DATA_DIR):
        return DATA_DIR

    PROJECT_ROOT = os.environ.get("PROJECT_ROOT")
    DATA_PATH = os.environ.get("DATA_PATH")
    if not PROJECT_ROOT or not DATA_PATH:
        raise EnvironmentError(
            "Environment variables PROJECT_ROOT and DATA_PATH must be set. "
            "Example: export PROJECT_ROOT=/abs/proj; export DATA_PATH=data"
        )
    DATA_DIR = os.path.join(PROJECT_ROOT, DATA_PATH)
    DATA_DIR = os.path.abspath(DATA_DIR)
    if not os.path.isabs(DATA_DIR):
        raise ValueError(f"DATA_DIR must be absolute; got: {DATA_DIR}")
    return DATA_DIR


def info(msg: str) -> None:
    """Print an INFO log line to stdout, prefixed as required."""
    print(f"[INFO] {msg}")


# ----------------------------
# Parsing / Data Structures
# ----------------------------

_ALLOWED_START = date(2019, 1, 1)
_ALLOWED_END = date(2020, 12, 31)

_DAYLOG_RE = re.compile(r"^\s*Activities at (\d{4}-\d{2}-\d{2}):\s*(.*?)\s*\.\s*$")
_EVENT_RE = re.compile(r"^\s*(.+?)\s+at\s+(\d{1,2}:\d{2}:\d{2})\s*$")


def _parse_iso_date(s: str) -> date:
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except ValueError as e:
        raise ValueError(f"Invalid date token '{s}'. Expected YYYY-MM-DD.") from e


def _parse_hms_to_minute(hms: str) -> int:
    """
    Parse H:MM:SS or HH:MM:SS and round to minute with seconds>=30 rounding up.
    Clamp to [0, 1439].
    """
    parts = hms.split(":")
    if len(parts) != 3:
        raise ValueError(f"Invalid time token '{hms}'. Expected H:MM:SS or HH:MM:SS.")
    hh, mm, ss = parts
    try:
        h = int(hh)
        m = int(mm)
        s = int(ss)
    except ValueError as e:
        raise ValueError(f"Invalid time token '{hms}'. Non-integer components.") from e
    if not (0 <= h <= 23 and 0 <= m <= 59 and 0 <= s <= 59):
        raise ValueError(f"Invalid time token '{hms}'. Out-of-range component.")
    minute = h * 60 + m + (1 if s >= 30 else 0)
    return max(0, min(1439, minute))


def _minute_to_hms(minute: int) -> str:
    """Format minute-of-day to HH:MM:SS (seconds emitted as 00)."""
    minute = max(0, min(1439, int(minute)))
    hh = minute // 60
    mm = minute % 60
    return f"{hh:02d}:{mm:02d}:00"


def _base_poi_name(full_poi_id: str) -> str:
    """Strip trailing '#id' if present."""
    if "#" in full_poi_id:
        return full_poi_id.split("#", 1)[0]
    return full_poi_id


@dataclass(frozen=True)
class VisitEvent:
    """A single visit event (arrival time only; dwell is imputed in simulation)."""

    poi_id: str
    minute: int
    coarse_category: str


@dataclass(frozen=True)
class DayLog:
    """Parsed day log for one agent and one date."""

    day: date
    events: Tuple[VisitEvent, ...]


@dataclass
class RawData:
    """Raw loaded input data files."""

    trajectories_by_agent: Dict[str, List[str]]
    poi_records: dict
    catto: Dict[str, str]


@dataclass
class ParsedData:
    """Parsed and filtered data scoped strictly to 2019-2020 inclusive."""

    days_by_agent: Dict[str, List[DayLog]]  # sorted ascending by date
    poi_id_to_latlon: Dict[str, Tuple[float, float]]
    base_poi_to_category: Dict[str, str]
    category_to_poi_ids: Dict[str, List[str]]
    meta: Dict[str, object] = field(default_factory=dict)


# ----------------------------
# Environment / POI Universe
# ----------------------------

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute Haversine distance in kilometers."""
    r = 6371.0088
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = phi2 - phi1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dl / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1 - a)))
    return r * c


class CityPOIEnvironment:
    """POI environment holding coordinates and category mappings."""

    def __init__(
        self,
        poi_id_to_latlon: Dict[str, Tuple[float, float]],
        base_poi_to_category: Dict[str, str],
    ) -> None:
        self.poi_id_to_latlon = poi_id_to_latlon
        self.base_poi_to_category = base_poi_to_category
        self.category_to_poi_ids: Dict[str, List[str]] = {}
        for poi_id in poi_id_to_latlon.keys():
            base = _base_poi_name(poi_id)
            cat = base_poi_to_category.get(base, "Unknown")
            self.category_to_poi_ids.setdefault(cat, []).append(poi_id)

    def distance_km(self, a: str, b: str) -> Optional[float]:
        """
        Return distance in km or None if coordinates are missing.

        NOTE: Missing is None (not 0.0) to avoid under-penalization.
        """
        la = self.poi_id_to_latlon.get(a)
        lb = self.poi_id_to_latlon.get(b)
        if la is None or lb is None:
            return None
        return haversine_km(la[0], la[1], lb[0], lb[1])


# ----------------------------
# Formatter (strict)
# ----------------------------

class TrajectoryStringFormatter:
    """
    Strict serializer/validator for 1921Y.json day_log strings.

    Canonical format (no deviations allowed):
      'Activities at YYYY-MM-DD: POI#id at HH:MM:SS, POI#id at HH:MM:SS.'
    """

    FORMAT_SPEC = (
        "Activities at YYYY-MM-DD: POI#id at HH:MM:SS, POI#id at HH:MM:SS, ... . "
        "Canonical emission here uses comma+space separators and ends with a single '.' (no preceding space)."
    )

    _VALID_RE = re.compile(
        r"^Activities at \d{4}-\d{2}-\d{2}: "
        r"(.+ at \d{2}:\d{2}:\d{2})(, .+ at \d{2}:\d{2}:\d{2})*\.$"
    )

    @classmethod
    def format_day_log(cls, day: date, events: Sequence[Tuple[str, int]]) -> str:
        """
        Format a day log from (poi_id, minute) pairs.
        """
        if not events:
            raise ValueError("Cannot format empty event list; ground truth always has at least 1 event.")
        tokens = []
        for poi_id, minute in events:
            if not isinstance(poi_id, str) or not poi_id:
                raise ValueError("poi_id must be a non-empty string.")
            t = _minute_to_hms(minute)
            tokens.append(f"{poi_id} at {t}")
        s = f"Activities at {day.isoformat()}: " + ", ".join(tokens) + "."
        cls.validate_day_log(s)
        return s

    @classmethod
    def validate_day_log(cls, s: str) -> None:
        """Validate that a string exactly matches the canonical grammar."""
        if cls._VALID_RE.match(s) is None:
            raise ValueError(
                "Trajectory string format violation. Expected canonical 1921Y.json grammar exactly. "
                f"Got: {s[:200]!r}"
            )


# ----------------------------
# Agent Priors
# ----------------------------

def _safe_log(x: float) -> float:
    return math.log(max(x, 1e-300))


def _softmax_sample(rng: random.Random, keys: List[str], logits: List[float]) -> str:
    if len(keys) != len(logits) or not keys:
        raise ValueError("Invalid softmax sample inputs.")
    m = max(logits)
    exps = [math.exp(l - m) for l in logits]
    s = sum(exps)
    if s <= 0:
        return rng.choice(keys)
    r = rng.random() * s
    cum = 0.0
    for k, e in zip(keys, exps):
        cum += e
        if cum >= r:
            return k
    return keys[-1]


@dataclass
class AgentPriors:
    """Per-agent priors learned from TRAIN days only."""

    agent_id: str
    home_poi: str
    mobility_radius_km: float

    poi_prob: Dict[str, float]
    cat_prob: Dict[str, float]
    start_minute_hist: List[float]  # binned
    cat_time_hist: Dict[str, List[float]]  # binned
    cat_trans_prob: Dict[Tuple[str, str], float]  # (prev_cat,next_cat)->p
    poi_trans_prob: Dict[Tuple[str, str], float]  # (prev_poi,next_poi)->p
    stop_count_values: List[int]
    p_start_home: float
    p_end_home: float

    base_dwell_lognorm: Dict[str, Tuple[float, float]]  # cat -> (mu, sigma) in log-minutes


@dataclass
class MobilityAgent:
    """Resident mobility actor holding learned priors."""

    priors: AgentPriors


# ----------------------------
# Holdout / Split
# ----------------------------

@dataclass(frozen=True)
class HoldoutSplit:
    """Train/validation split with eligibility filtering."""

    train_by_agent: Dict[str, List[DayLog]]
    val_by_agent: Dict[str, List[DayLog]]  # eligible only
    excluded_agents: Dict[str, str]  # agent_id -> reason
    meta: Dict[str, object]


def holdout_split(parsed: ParsedData) -> HoldoutSplit:
    train_by_agent: Dict[str, List[DayLog]] = {}
    val_by_agent: Dict[str, List[DayLog]] = {}
    excluded: Dict[str, str] = {}

    for agent_id, days in parsed.days_by_agent.items():
        if not days:
            continue
        n = len(days)
        if n < 8:
            excluded[agent_id] = f"low_data_policy: only {n} filtered days (<8)"
            train_by_agent[agent_id] = days[:]
            val_by_agent[agent_id] = []
            continue

        split_idx = int(math.floor(0.8 * n))
        split_idx = max(1, min(n - 1, split_idx))

        train = days[:split_idx]
        candidate_val = days[split_idx:]

        eligible_val: List[DayLog] = []
        for dlog in candidate_val:
            prior_days = [d for d in days if d.day < dlog.day]
            if len(prior_days) >= 7:
                eligible_val.append(dlog)

        train_by_agent[agent_id] = train
        val_by_agent[agent_id] = eligible_val

        if not eligible_val:
            excluded[agent_id] = "no eligible validation days (needs >=7 prior days before each val date)"

    meta = {
        "method": "temporal_holdout",
        "train_fraction": 0.8,
        "eligibility_rule": "validation target requires >=7 prior filtered days before that date",
        "allowed_date_range_inclusive": [str(_ALLOWED_START), str(_ALLOWED_END)],
        "num_agents_total": len(parsed.days_by_agent),
        "num_agents_with_any_val": sum(1 for v in val_by_agent.values() if v),
        "num_agents_excluded_from_val": sum(1 for _ in excluded.items()),
    }
    return HoldoutSplit(train_by_agent=train_by_agent, val_by_agent=val_by_agent, excluded_agents=excluded, meta=meta)


# ----------------------------
# Loading / Parsing
# ----------------------------

def load_data() -> RawData:
    data_dir = _require_data_dir()

    required = [
        "1921Y.json",
        "poi_category_192021_longitude_latitude.json",
        "catto.json",
    ]
    for fn in required:
        p = os.path.join(data_dir, fn)
        if not os.path.isabs(p):
            raise ValueError(f"DATA_DIR-based path must be absolute; got: {p}")
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing required data file: {p}")

    traj_path = os.path.join(data_dir, "1921Y.json")
    poi_path = os.path.join(data_dir, "poi_category_192021_longitude_latitude.json")
    catto_path = os.path.join(data_dir, "catto.json")

    with open(traj_path, "r", encoding="utf-8") as f:
        trajectories_by_agent = json.load(f)
    if not isinstance(trajectories_by_agent, dict):
        raise ValueError("1921Y.json must be a JSON object mapping agent_id -> list[day_log strings].")

    with open(poi_path, "r", encoding="utf-8") as f:
        poi_records = json.load(f)
    if not isinstance(poi_records, dict):
        raise ValueError("poi_category_...json must be a JSON object.")

    with open(catto_path, "r", encoding="utf-8") as f:
        catto = json.load(f)
    if not isinstance(catto, dict):
        raise ValueError("catto.json must be a JSON object mapping base POI name -> coarse category.")

    return RawData(
        trajectories_by_agent=trajectories_by_agent,
        poi_records=poi_records,
        catto=catto,
    )


def _parse_day_log_string(day_log: str, base_poi_to_category: Dict[str, str]) -> DayLog:
    m = _DAYLOG_RE.match(day_log)
    if m is None:
        raise ValueError(
            "Invalid day_log string. Expected 'Activities at YYYY-MM-DD: ... .' "
            f"Got: {day_log[:200]!r}"
        )
    day_s = m.group(1)
    body = m.group(2).strip()

    d = _parse_iso_date(day_s)
    if d < _ALLOWED_START or d > _ALLOWED_END:
        raise ValueError("Internal parser called on out-of-range day; filtering must happen earlier.")

    if not body:
        raise ValueError(f"Empty activity body for date {day_s}.")

    parts = [p.strip() for p in body.split(",")]
    events: List[VisitEvent] = []
    for p in parts:
        em = _EVENT_RE.match(p)
        if em is None:
            raise ValueError(f"Invalid event token: {p!r} in day_log: {day_log[:200]!r}")
        poi_id = em.group(1).strip()
        hms = em.group(2).strip()
        minute = _parse_hms_to_minute(hms)

        base = _base_poi_name(poi_id)
        cat = base_poi_to_category.get(base, "Unknown")
        events.append(VisitEvent(poi_id=poi_id, minute=minute, coarse_category=cat))

    events.sort(key=lambda e: e.minute)
    if not events:
        raise ValueError(f"No events parsed for {day_s}.")
    return DayLog(day=d, events=tuple(events))


def _build_poi_indices(poi_records: dict) -> Dict[str, Tuple[float, float]]:
    poi_id_to_latlon: Dict[str, Tuple[float, float]] = {}
    for _base_name, recs in poi_records.items():
        if not isinstance(recs, list):
            continue
        for rec in recs:
            if not (isinstance(rec, list) and len(rec) == 3):
                continue
            lat_s, lon_s, full_id = rec
            if not isinstance(full_id, str):
                continue
            try:
                lat = float(lat_s)
                lon = float(lon_s)
            except Exception:
                continue
            poi_id_to_latlon[full_id] = (lat, lon)
    if not poi_id_to_latlon:
        raise ValueError("Failed to build poi_id_to_latlon; check POI input file format.")
    return poi_id_to_latlon


def build_network_and_agents(raw: RawData, seed: int) -> Tuple[ParsedData, CityPOIEnvironment, Dict[str, MobilityAgent]]:
    base_poi_to_category = dict(raw.catto)
    poi_id_to_latlon = _build_poi_indices(raw.poi_records)
    env = CityPOIEnvironment(poi_id_to_latlon=poi_id_to_latlon, base_poi_to_category=base_poi_to_category)

    days_by_agent: Dict[str, List[DayLog]] = {}
    dropped_out_of_range = 0
    kept = 0

    for agent_id, day_logs in raw.trajectories_by_agent.items():
        if not isinstance(agent_id, str) or not agent_id:
            continue
        if not isinstance(day_logs, list):
            raise ValueError(f"Agent {agent_id} value must be a list of day_log strings.")
        parsed_days: List[DayLog] = []
        for s in day_logs:
            if not isinstance(s, str):
                continue
            m = _DAYLOG_RE.match(s)
            if m is None:
                raise ValueError(f"Invalid day_log format under agent {agent_id}: {s[:200]!r}")
            d = _parse_iso_date(m.group(1))
            if d < _ALLOWED_START or d > _ALLOWED_END:
                dropped_out_of_range += 1
                continue
            parsed_days.append(_parse_day_log_string(s, base_poi_to_category))
            kept += 1
        parsed_days.sort(key=lambda dl: dl.day)
        if parsed_days:
            days_by_agent[agent_id] = parsed_days

    parsed = ParsedData(
        days_by_agent=days_by_agent,
        poi_id_to_latlon=poi_id_to_latlon,
        base_poi_to_category=base_poi_to_category,
        category_to_poi_ids=env.category_to_poi_ids,
        meta={
            "allowed_date_range_inclusive": [str(_ALLOWED_START), str(_ALLOWED_END)],
            "dropped_out_of_range_daylogs": dropped_out_of_range,
            "kept_daylogs": kept,
            "seed": seed,
        },
    )

    return parsed, env, {}


# ----------------------------
# Priors Learning
# ----------------------------

def _dirichlet_smooth_probs(counts: Dict[str, int], support: Iterable[str], alpha: float) -> Dict[str, float]:
    keys = list(dict.fromkeys(support))
    total = 0.0
    out: Dict[str, float] = {}
    for k in keys:
        c = float(counts.get(k, 0))
        v = c + alpha
        out[k] = v
        total += v
    if total <= 0:
        u = 1.0 / max(1, len(keys))
        return {k: u for k in keys}
    for k in keys:
        out[k] /= total
    return out


def _make_hist(values: List[int], bins: int, max_value: int, smoothing: float) -> List[float]:
    if bins <= 0:
        raise ValueError("bins must be positive.")
    w = (max_value + 1) / bins
    h = [float(smoothing)] * bins
    for v in values:
        v = max(0, min(max_value, int(v)))
        idx = int(v / w)
        if idx >= bins:
            idx = bins - 1
        h[idx] += 1.0
    s = sum(h)
    if s <= 0:
        return [1.0 / bins] * bins
    return [x / s for x in h]


def _hist_sample_minute(rng: random.Random, hist: List[float], bins: int) -> int:
    if len(hist) != bins:
        raise ValueError("Histogram length mismatch.")
    r = rng.random()
    cum = 0.0
    chosen = bins - 1
    for i, p in enumerate(hist):
        cum += p
        if cum >= r:
            chosen = i
            break
    bin_width = 1440 / bins
    lo = int(chosen * bin_width)
    hi = min(1439, int((chosen + 1) * bin_width) - 1)
    if hi < lo:
        hi = lo
    return rng.randint(lo, hi)


def _estimate_lognormal_params(minutes: List[int]) -> Tuple[float, float]:
    xs = [max(1.0, float(m)) for m in minutes if m is not None and m > 0]
    if len(xs) < 5:
        return (math.log(60.0), 0.6)
    logs = [math.log(x) for x in xs]
    mu = statistics.mean(logs)
    sd = statistics.pstdev(logs) if len(logs) > 1 else 0.6
    sd = max(0.1, min(2.0, sd))
    return (mu, sd)


def _median_or_default(xs: List[float], default: float) -> float:
    xs = [x for x in xs if x is not None and math.isfinite(x)]
    if not xs:
        return default
    xs.sort()
    mid = len(xs) // 2
    if len(xs) % 2 == 1:
        return xs[mid]
    return 0.5 * (xs[mid - 1] + xs[mid])


def learn_agent_priors(
    env: CityPOIEnvironment,
    train_by_agent: Dict[str, List[DayLog]],
    *,
    smoothing_strength: float,
    tod_bins: int,
) -> Dict[str, MobilityAgent]:
    agents: Dict[str, MobilityAgent] = {}
    global_categories = sorted(env.category_to_poi_ids.keys())
    global_pois = list(env.poi_id_to_latlon.keys())

    for agent_id, train_days in train_by_agent.items():
        if not train_days:
            continue

        poi_counts: Dict[str, int] = {}
        cat_counts: Dict[str, int] = {}
        start_minutes: List[int] = []
        stop_counts: List[int] = []
        start_home_hits = 0
        end_home_hits = 0
        day_count = 0

        cat_trans_counts: Dict[Tuple[str, str], int] = {}
        poi_trans_counts: Dict[Tuple[str, str], int] = {}

        cat_minutes: Dict[str, List[int]] = {}
        gap_minutes_by_cat: Dict[str, List[int]] = {}

        for dlog in train_days:
            day_count += 1
            events = list(dlog.events)
            if not events:
                continue
            stop_counts.append(len(events))
            start_minutes.append(events[0].minute)

            for e in events:
                poi_counts[e.poi_id] = poi_counts.get(e.poi_id, 0) + 1
                cat_counts[e.coarse_category] = cat_counts.get(e.coarse_category, 0) + 1
                cat_minutes.setdefault(e.coarse_category, []).append(e.minute)

            for a, b in zip(events[:-1], events[1:]):
                cat_trans_counts[(a.coarse_category, b.coarse_category)] = cat_trans_counts.get(
                    (a.coarse_category, b.coarse_category), 0
                ) + 1
                poi_trans_counts[(a.poi_id, b.poi_id)] = poi_trans_counts.get((a.poi_id, b.poi_id), 0) + 1

                gap = max(0, b.minute - a.minute)
                gap_minutes_by_cat.setdefault(a.coarse_category, []).append(gap)

        if not poi_counts:
            continue

        home_poi = max(poi_counts.items(), key=lambda kv: kv[1])[0]

        for dlog in train_days:
            events = list(dlog.events)
            if not events:
                continue
            if events[0].poi_id == home_poi:
                start_home_hits += 1
            if events[-1].poi_id == home_poi:
                end_home_hits += 1
        p_start_home = start_home_hits / max(1, day_count)
        p_end_home = end_home_hits / max(1, day_count)

        dists: List[float] = []
        for poi_id in poi_counts.keys():
            d = env.distance_km(home_poi, poi_id)
            if d is not None:
                dists.append(d)
        mobility_radius = _median_or_default(dists, default=5.0)
        mobility_radius = max(0.5, min(50.0, mobility_radius))

        poi_prob = _dirichlet_smooth_probs(poi_counts, global_pois, alpha=max(1e-9, smoothing_strength))
        cat_prob = _dirichlet_smooth_probs(cat_counts, global_categories, alpha=max(1e-9, smoothing_strength))

        start_min_hist = _make_hist(start_minutes, bins=tod_bins, max_value=1439, smoothing=1.0)
        cat_time_hist: Dict[str, List[float]] = {}
        for cat in global_categories:
            vals = cat_minutes.get(cat, [])
            cat_time_hist[cat] = _make_hist(vals, bins=tod_bins, max_value=1439, smoothing=1.0)

        cat_trans_prob: Dict[Tuple[str, str], float] = {}
        prev_totals: Dict[str, float] = {c: 0.0 for c in global_categories}
        for prev in global_categories:
            for nxt in global_categories:
                cnt = float(cat_trans_counts.get((prev, nxt), 0))
                val = cnt + smoothing_strength
                cat_trans_prob[(prev, nxt)] = val
                prev_totals[prev] += val
        for prev in global_categories:
            denom = prev_totals[prev] if prev_totals[prev] > 0 else 1.0
            for nxt in global_categories:
                cat_trans_prob[(prev, nxt)] /= denom

        seen_pois = list(poi_counts.keys())
        if home_poi not in seen_pois:
            seen_pois.append(home_poi)
        poi_trans_prob: Dict[Tuple[str, str], float] = {}
        prev_poi_totals: Dict[str, float] = {}
        for prev in seen_pois:
            total = 0.0
            for nxt in seen_pois:
                cnt = float(poi_trans_counts.get((prev, nxt), 0))
                val = cnt + smoothing_strength
                poi_trans_prob[(prev, nxt)] = val
                total += val
            prev_poi_totals[prev] = total if total > 0 else 1.0
        for prev in seen_pois:
            denom = prev_poi_totals[prev]
            for nxt in seen_pois:
                poi_trans_prob[(prev, nxt)] /= denom

        base_dwell: Dict[str, Tuple[float, float]] = {}
        for cat in global_categories:
            mu, sig = _estimate_lognormal_params(gap_minutes_by_cat.get(cat, []))
            base_dwell[cat] = (mu, sig)

        priors = AgentPriors(
            agent_id=agent_id,
            home_poi=home_poi,
            mobility_radius_km=mobility_radius,
            poi_prob=poi_prob,
            cat_prob=cat_prob,
            start_minute_hist=start_min_hist,
            cat_time_hist=cat_time_hist,
            cat_trans_prob=cat_trans_prob,
            poi_trans_prob=poi_trans_prob,
            stop_count_values=stop_counts if stop_counts else [1],
            p_start_home=p_start_home,
            p_end_home=p_end_home,
            base_dwell_lognorm=base_dwell,
        )
        agents[agent_id] = MobilityAgent(priors=priors)

    return agents


# ----------------------------
# Simulator
# ----------------------------

@dataclass(frozen=True)
class SimulatorParameters:
    alpha_pref: float
    alpha_transition: float
    beta_distance: float
    travel_time_scale: float
    smoothing_strength: float
    stop_count_scale: float
    dwell_mu_shift: float
    dwell_sigma_mult: float

    def to_jsonable(self, categories: Sequence[str], base_dwell: Dict[str, Tuple[float, float]]) -> Dict[str, object]:
        dwell_params: Dict[str, Dict[str, float]] = {}
        for cat in categories:
            mu0, sig0 = base_dwell.get(cat, (math.log(60.0), 0.6))
            mu = max(0.0, min(6.0, mu0 + self.dwell_mu_shift))
            sig = max(0.1, min(2.0, sig0 * self.dwell_sigma_mult))
            dwell_params[cat] = {"mu": float(mu), "sigma": float(sig)}
        return {
            "alpha_pref": float(self.alpha_pref),
            "alpha_transition": float(self.alpha_transition),
            "beta_distance": float(self.beta_distance),
            "travel_time_scale": float(self.travel_time_scale),
            "smoothing_strength": float(self.smoothing_strength),
            "end_day_or_stop_count_params": {"stop_count_scale": float(self.stop_count_scale)},
            "dwell_time_params_by_category": dwell_params,
        }


class MobilitySimulator:
    def __init__(
        self,
        env: CityPOIEnvironment,
        agents: Dict[str, MobilityAgent],
        *,
        tod_bins: int,
        global_seed: int,
    ) -> None:
        self.env = env
        self.agents = agents
        self.tod_bins = tod_bins
        self.global_seed = global_seed

        self._global_categories = sorted(env.category_to_poi_ids.keys())
        self._fallback_missing_dist_km = 5.0

    def _distance_km_or_fallback(self, agent: MobilityAgent, a: str, b: str) -> float:
        d = self.env.distance_km(a, b)
        if d is None:
            return max(self._fallback_missing_dist_km, agent.priors.mobility_radius_km)
        return d

    def _sample_stop_count(self, rng: random.Random, priors: AgentPriors, stop_count_scale: float) -> int:
        n0 = rng.choice(priors.stop_count_values) if priors.stop_count_values else 1
        n = int(round(max(1.0, float(n0) * stop_count_scale)))
        return max(1, min(50, n))

    def _choose_next_category(
        self,
        rng: random.Random,
        priors: AgentPriors,
        params: SimulatorParameters,
        *,
        prev_cat: Optional[str],
        current_minute: int,
    ) -> Tuple[str, int]:
        cats = self._global_categories
        logits: List[float] = []

        for c in cats:
            pref = priors.cat_prob.get(c, 1e-12)
            if prev_cat is None:
                trans = 1.0 / max(1, len(cats))
            else:
                trans = priors.cat_trans_prob.get((prev_cat, c), 1e-12)

            h = priors.cat_time_hist.get(c)
            if h is None:
                tod_prob = 1.0 / self.tod_bins
                target_min = current_minute
            else:
                bin_width = 1440 / self.tod_bins
                idx = int(current_minute / bin_width)
                idx = max(0, min(self.tod_bins - 1, idx))
                tod_prob = max(1e-12, h[idx])
                target_min = _hist_sample_minute(rng, h, self.tod_bins)

            time_misalignment = abs(target_min - current_minute) / 1440.0
            logit = (
                params.alpha_pref * _safe_log(pref)
                + params.alpha_transition * _safe_log(trans)
                + 1.0 * _safe_log(tod_prob)
                - 2.0 * time_misalignment
            )
            logits.append(logit)

        chosen = _softmax_sample(rng, cats, logits)
        h = priors.cat_time_hist.get(chosen, [1.0 / self.tod_bins] * self.tod_bins)
        target = _hist_sample_minute(rng, h, self.tod_bins)
        return chosen, target

    def _choose_poi(
        self,
        rng: random.Random,
        agent: MobilityAgent,
        params: SimulatorParameters,
        *,
        category: str,
        prev_poi: Optional[str],
        current_location: str,
        is_last_event: bool,
    ) -> str:
        priors = agent.priors
        candidates = self.env.category_to_poi_ids.get(category, [])
        if not candidates:
            keys = list(priors.poi_prob.keys())
            logits = [_safe_log(priors.poi_prob.get(k, 1e-12)) for k in keys]
            return _softmax_sample(rng, keys, logits)

        if len(candidates) > 300:
            candidates = candidates[:300]

        logits = []
        keys = []

        end_home_bonus = 1.0 if (is_last_event and rng.random() < priors.p_end_home) else 0.0

        for poi_id in candidates:
            pref = priors.poi_prob.get(poi_id, 1e-12)

            if prev_poi is None:
                trans = pref
            else:
                trans = priors.poi_trans_prob.get((prev_poi, poi_id), 1e-12)

            dist_km = self._distance_km_or_fallback(agent, current_location, poi_id)
            norm = max(0.5, priors.mobility_radius_km)
            dist_pen = math.log1p(dist_km / norm)

            logit = (
                params.alpha_pref * _safe_log(pref)
                + params.alpha_transition * _safe_log(trans)
                - params.beta_distance * dist_pen
            )

            if poi_id == priors.home_poi:
                logit += end_home_bonus

            keys.append(poi_id)
            logits.append(logit)

        return _softmax_sample(rng, keys, logits)

    def _sample_dwell_minutes(self, rng: random.Random, priors: AgentPriors, params: SimulatorParameters, category: str) -> int:
        mu0, sig0 = priors.base_dwell_lognorm.get(category, (math.log(60.0), 0.6))
        mu = max(0.0, min(6.0, mu0 + params.dwell_mu_shift))
        sig = max(0.1, min(2.0, sig0 * params.dwell_sigma_mult))
        x = rng.lognormvariate(mu, sig)
        return int(max(5, min(6 * 60, round(x))))

    def simulate_one_day(
        self,
        agent_id: str,
        target_day: date,
        params: SimulatorParameters,
        *,
        rng: random.Random,
    ) -> Tuple[str, List[VisitEvent]]:
        if agent_id not in self.agents:
            raise KeyError(f"Agent '{agent_id}' has no learned priors (TRAIN empty?).")
        agent = self.agents[agent_id]
        priors = agent.priors

        n_events = self._sample_stop_count(rng, priors, params.stop_count_scale)

        start_min = _hist_sample_minute(rng, priors.start_minute_hist, self.tod_bins)
        current_time = start_min
        current_loc = priors.home_poi

        prev_cat: Optional[str] = None
        prev_poi: Optional[str] = None

        out_events: List[VisitEvent] = []
        out_pairs: List[Tuple[str, int]] = []

        for i in range(n_events):
            is_last = (i == n_events - 1)

            if i == 0 and rng.random() < priors.p_start_home:
                chosen_poi = priors.home_poi
                chosen_cat = self.env.base_poi_to_category.get(_base_poi_name(chosen_poi), "Unknown")
                arrival = current_time
                if arrival > 1439:
                    break
                out_events.append(VisitEvent(poi_id=chosen_poi, minute=arrival, coarse_category=chosen_cat))
                out_pairs.append((chosen_poi, arrival))
                dwell = self._sample_dwell_minutes(rng, priors, params, chosen_cat)
                current_time = min(1439, arrival + dwell)
                current_loc = chosen_poi
                prev_cat, prev_poi = chosen_cat, chosen_poi
                continue

            chosen_cat, target_min = self._choose_next_category(
                rng, priors, params, prev_cat=prev_cat, current_minute=current_time
            )
            chosen_poi = self._choose_poi(
                rng,
                agent,
                params,
                category=chosen_cat,
                prev_poi=prev_poi,
                current_location=current_loc,
                is_last_event=is_last,
            )

            dist_km = self._distance_km_or_fallback(agent, current_loc, chosen_poi)
            travel_minutes = int(round(max(0.0, dist_km * params.travel_time_scale)))

            arrival = int(max(current_time + travel_minutes, target_min))
            if arrival > 1439:
                break

            out_events.append(VisitEvent(poi_id=chosen_poi, minute=arrival, coarse_category=chosen_cat))
            out_pairs.append((chosen_poi, arrival))

            dwell = self._sample_dwell_minutes(rng, priors, params, chosen_cat)
            current_time = min(1439, arrival + dwell)
            current_loc = chosen_poi
            prev_cat, prev_poi = chosen_cat, chosen_poi

            if current_time >= 1439:
                break

        if not out_pairs:
            chosen_poi = priors.home_poi
            chosen_cat = self.env.base_poi_to_category.get(_base_poi_name(chosen_poi), "Unknown")
            arrival = max(0, min(1439, start_min))
            out_events = [VisitEvent(poi_id=chosen_poi, minute=arrival, coarse_category=chosen_cat)]
            out_pairs = [(chosen_poi, arrival)]

        s = TrajectoryStringFormatter.format_day_log(target_day, out_pairs)
        return s, out_events

    @staticmethod
    def _stable_day_seed(global_seed: int, agent_id: str, day_iso: str) -> int:
        payload = f"{global_seed}|{agent_id}|{day_iso}".encode("utf-8")
        digest = hashlib.sha256(payload).digest()
        return int.from_bytes(digest[:4], "big", signed=False)

    def rollout(
        self,
        split: HoldoutSplit,
        params: SimulatorParameters,
        *,
        subset: Optional[List[Tuple[str, date]]] = None,
    ) -> Tuple[Dict[str, List[str]], Dict[Tuple[str, date], List[VisitEvent]]]:
        targets: List[Tuple[str, date]] = []
        if subset is not None:
            targets = list(subset)
        else:
            for agent_id, val_days in split.val_by_agent.items():
                for dlog in val_days:
                    targets.append((agent_id, dlog.day))
            targets.sort(key=lambda x: (x[0], x[1].isoformat()))

        trajectories_by_agent: Dict[str, List[str]] = {}
        sim_events: Dict[Tuple[str, date], List[VisitEvent]] = {}

        for agent_id, d in targets:
            if agent_id not in self.agents:
                info(f"Skipping validation target for agent '{agent_id}' (no learned priors).")
                continue

            day_seed = self._stable_day_seed(self.global_seed, agent_id, d.isoformat())
            local_rng = random.Random(day_seed)

            s, evs = self.simulate_one_day(agent_id, d, params, rng=local_rng)
            TrajectoryStringFormatter.validate_day_log(s)
            trajectories_by_agent.setdefault(agent_id, []).append(s)
            sim_events[(agent_id, d)] = evs

        return trajectories_by_agent, sim_events


# ----------------------------
# Evaluation Metrics
# ----------------------------

def _js_divergence(p: List[float], q: List[float]) -> float:
    if len(p) != len(q):
        raise ValueError("JSD requires same-length vectors.")
    m = [(pi + qi) / 2.0 for pi, qi in zip(p, q)]

    def kl(a: List[float], b: List[float]) -> float:
        s = 0.0
        for ai, bi in zip(a, b):
            ai = max(ai, 1e-12)
            bi = max(bi, 1e-12)
            s += ai * math.log(ai / bi)
        return s

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def _kl_divergence(p: Dict[int, float], q: Dict[int, float]) -> float:
    keys = sorted(set(p.keys()) | set(q.keys()))
    s = 0.0
    for k in keys:
        pk = max(p.get(k, 0.0), 1e-12)
        qk = max(q.get(k, 0.0), 1e-12)
        s += pk * math.log(pk / qk)
    return float(s)


def _wasserstein_1d(a: List[float], b: List[float]) -> float:
    if not a and not b:
        return 0.0
    if not a or not b:
        xs = a if a else b
        return float(sum(abs(x) for x in xs) / max(1, len(xs)))

    a_sorted = sorted(a)
    b_sorted = sorted(b)
    n = len(a_sorted)
    m = len(b_sorted)

    points = sorted(set(a_sorted + b_sorted))
    ia = 0
    ib = 0
    wa = 1.0 / n
    wb = 1.0 / m
    cdf_a = 0.0
    cdf_b = 0.0

    prev = points[0]
    dist = 0.0

    for x in points:
        dx = x - prev
        if dx != 0:
            dist += abs(cdf_a - cdf_b) * dx

        while ia < n and a_sorted[ia] <= x:
            cdf_a += wa
            ia += 1
        while ib < m and b_sorted[ib] <= x:
            cdf_b += wb
            ib += 1

        prev = x

    return float(dist)


class Evaluator:
    def __init__(self, env: CityPOIEnvironment, parsed: ParsedData, *, tod_bins_eval: int = 144) -> None:
        self.env = env
        self.parsed = parsed
        self.tod_bins_eval = tod_bins_eval
        self._global_categories = sorted(env.category_to_poi_ids.keys())

    def _build_ground_truth(self, split: HoldoutSplit) -> Dict[Tuple[str, date], List[VisitEvent]]:
        gt: Dict[Tuple[str, date], List[VisitEvent]] = {}
        for agent_id, val_days in split.val_by_agent.items():
            for dlog in val_days:
                gt[(agent_id, dlog.day)] = list(dlog.events)
        return gt

    def compute_metrics(
        self,
        split: HoldoutSplit,
        sim_events: Dict[Tuple[str, date], List[VisitEvent]],
        *,
        objective_weights: Dict[str, float],
    ) -> Tuple[Dict[str, float], float, Dict[str, object]]:
        gt = self._build_ground_truth(split)
        targets = sorted(gt.keys(), key=lambda x: (x[0], x[1].isoformat()))
        if not targets:
            raise ValueError("Validation set is empty after eligibility filtering; cannot evaluate.")

        gt_stop_counts: List[int] = []
        sim_stop_counts: List[int] = []

        gt_cat_counts: Dict[str, int] = {c: 0 for c in self._global_categories}
        sim_cat_counts: Dict[str, int] = {c: 0 for c in self._global_categories}

        gt_tod_by_cat: Dict[str, List[float]] = {c: [1.0] * self.tod_bins_eval for c in self._global_categories}
        sim_tod_by_cat: Dict[str, List[float]] = {c: [1.0] * self.tod_bins_eval for c in self._global_categories}

        gt_bigram: Dict[Tuple[str, str], int] = {}
        sim_bigram: Dict[Tuple[str, str], int] = {}

        gt_poi_counts_by_agent: Dict[str, Dict[str, int]] = {}
        sim_poi_set_by_agent: Dict[str, set] = {}

        gt_step_dists: List[float] = []
        sim_step_dists: List[float] = []

        def tod_bin(minute: int) -> int:
            w = 1440 / self.tod_bins_eval
            idx = int(max(0, min(1439, minute)) / w)
            return max(0, min(self.tod_bins_eval - 1, idx))

        for key in targets:
            agent_id, _d = key
            gt_events = gt[key]
            se = sim_events.get(key, [])
            if not se:
                se = []

            gt_stop_counts.append(len(gt_events))
            sim_stop_counts.append(len(se))

            for e in gt_events:
                if e.coarse_category in gt_cat_counts:
                    gt_cat_counts[e.coarse_category] += 1
                    gt_tod_by_cat[e.coarse_category][tod_bin(e.minute)] += 1.0
                gt_poi_counts_by_agent.setdefault(agent_id, {})
                gt_poi_counts_by_agent[agent_id][e.poi_id] = gt_poi_counts_by_agent[agent_id].get(e.poi_id, 0) + 1

            for e in se:
                if e.coarse_category in sim_cat_counts:
                    sim_cat_counts[e.coarse_category] += 1
                    sim_tod_by_cat[e.coarse_category][tod_bin(e.minute)] += 1.0
                sim_poi_set_by_agent.setdefault(agent_id, set()).add(e.poi_id)

            for a, b in zip(gt_events[:-1], gt_events[1:]):
                gt_bigram[(a.coarse_category, b.coarse_category)] = gt_bigram.get((a.coarse_category, b.coarse_category), 0) + 1
                dkm = self.env.distance_km(a.poi_id, b.poi_id)
                if dkm is not None:
                    gt_step_dists.append(float(dkm))

            for a, b in zip(se[:-1], se[1:]):
                sim_bigram[(a.coarse_category, b.coarse_category)] = sim_bigram.get((a.coarse_category, b.coarse_category), 0) + 1
                dkm = self.env.distance_km(a.poi_id, b.poi_id)
                if dkm is not None:
                    sim_step_dists.append(float(dkm))

        gt_total = sum(gt_cat_counts.values())
        sim_total = sum(sim_cat_counts.values())
        gt_share = {c: (gt_cat_counts[c] / gt_total) if gt_total > 0 else 0.0 for c in self._global_categories}
        sim_share = {c: (sim_cat_counts[c] / sim_total) if sim_total > 0 else 0.0 for c in self._global_categories}
        category_share_mae = float(
            sum(abs(gt_share[c] - sim_share[c]) for c in self._global_categories) / max(1, len(self._global_categories))
        )

        gt_mean = statistics.mean(gt_stop_counts) if gt_stop_counts else 0.0
        sim_mean = statistics.mean(sim_stop_counts) if sim_stop_counts else 0.0
        stop_count_abs_mean_error = float(abs(gt_mean - sim_mean))

        def dist_from_counts(xs: List[int]) -> Dict[int, float]:
            c: Dict[int, int] = {}
            for x in xs:
                c[int(x)] = c.get(int(x), 0) + 1
            keys = sorted(c.keys())
            if not keys:
                return {0: 1.0}
            out: Dict[int, float] = {}
            total = 0.0
            for k in keys:
                out[k] = float(c.get(k, 0) + 1)
                total += out[k]
            for k in out:
                out[k] /= total
            return out

        p_sc = dist_from_counts(gt_stop_counts)
        q_sc = dist_from_counts(sim_stop_counts)
        stop_count_kl = float(_kl_divergence(p_sc, q_sc))

        jsds: List[float] = []
        for c in self._global_categories:
            g = gt_tod_by_cat[c]
            s = sim_tod_by_cat[c]
            sg = sum(g)
            ss = sum(s)
            pg = [x / sg for x in g] if sg > 0 else [1.0 / self.tod_bins_eval] * self.tod_bins_eval
            ps = [x / ss for x in s] if ss > 0 else [1.0 / self.tod_bins_eval] * self.tod_bins_eval
            jsds.append(_js_divergence(pg, ps))
        tod_jsd_avg = float(sum(jsds) / max(1, len(jsds)))

        K = 10
        recalls: List[float] = []
        for agent_id, poi_counts in gt_poi_counts_by_agent.items():
            topk = [p for p, _ in sorted(poi_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:K]]
            if not topk:
                continue
            sim_set = sim_poi_set_by_agent.get(agent_id, set())
            hit = sum(1 for p in topk if p in sim_set)
            recalls.append(hit / len(topk))
        topk_poi_recall = float(statistics.mean(recalls) if recalls else 0.0)

        all_pairs = [(a, b) for a in self._global_categories for b in self._global_categories]
        gt_vec = []
        sim_vec = []
        gt_sum = 0.0
        sim_sum = 0.0
        for pair in all_pairs:
            gv = float(gt_bigram.get(pair, 0) + 1)
            sv = float(sim_bigram.get(pair, 0) + 1)
            gt_vec.append(gv)
            sim_vec.append(sv)
            gt_sum += gv
            sim_sum += sv
        gt_vec = [x / gt_sum for x in gt_vec]
        sim_vec = [x / sim_sum for x in sim_vec]
        transition_divergence = float(_js_divergence(gt_vec, sim_vec))

        trip_distance_wasserstein = float(_wasserstein_1d(gt_step_dists, sim_step_dists))

        metrics = {
            "category_share_mae": category_share_mae,
            "stop_count_abs_mean_error": stop_count_abs_mean_error,
            "stop_count_kl": stop_count_kl,
            "tod_jsd_avg": tod_jsd_avg,
            "topk_poi_recall": topk_poi_recall,
            "transition_divergence": transition_divergence,
            "trip_distance_wasserstein": trip_distance_wasserstein,
        }

        objective = 0.0
        for k, w in objective_weights.items():
            if k not in metrics:
                raise KeyError(f"objective_weights contains unknown metric '{k}'.")
            val = metrics[k]
            if k == "topk_poi_recall":
                val = 1.0 - val
            objective += float(w) * float(val)

        validation_summary = {
            "num_agent_days": len(targets),
            "num_agents": len(set(a for a, _ in targets)),
            "targets_date_range": [targets[0][1].isoformat(), targets[-1][1].isoformat()],
            "excluded_agents": split.excluded_agents,
            "holdout_meta": split.meta,
        }
        return metrics, float(objective), validation_summary


# ----------------------------
# Calibration (pluggable)
# ----------------------------

class Calibrator(ABC):
    @abstractmethod
    def fit(self) -> Tuple[SimulatorParameters, float, List[Dict[str, object]]]:
        """Fit parameters and return (best_params, best_objective, calibration_log)."""
        raise NotImplementedError


class RandomSearchCalibrator(Calibrator):
    def __init__(
        self,
        simulator: MobilitySimulator,
        evaluator: Evaluator,
        split: HoldoutSplit,
        agents: Dict[str, MobilityAgent],
        env: CityPOIEnvironment,
        *,
        seed: int,
        iters: int,
        objective_weights: Dict[str, float],
        subsample_days: int,
    ) -> None:
        self.simulator = simulator
        self.evaluator = evaluator
        self.split = split
        self.agents = agents
        self.env = env
        self.seed = seed
        self.iters = iters
        self.objective_weights = objective_weights
        self.subsample_days = subsample_days
        self.rng = random.Random(seed + 12345)

        self._all_targets: List[Tuple[str, date]] = []
        for agent_id, val_days in split.val_by_agent.items():
            if agent_id not in agents:
                continue
            for dlog in val_days:
                self._all_targets.append((agent_id, dlog.day))
        self._all_targets.sort(key=lambda x: (x[0], x[1].isoformat()))

        if not self._all_targets:
            raise ValueError("No eligible validation targets for calibration (after filtering to agents with priors).")

        self._proxy_targets = self._make_proxy_targets()

    def _make_proxy_targets(self) -> List[Tuple[str, date]]:
        if self.subsample_days <= 0 or self.subsample_days >= len(self._all_targets):
            return self._all_targets
        rng = random.Random(self.seed + 7777)
        idxs = list(range(len(self._all_targets)))
        rng.shuffle(idxs)
        chosen = sorted(idxs[: self.subsample_days])
        return [self._all_targets[i] for i in chosen]

    def _sample_params(self) -> SimulatorParameters:
        alpha_pref = self.rng.random()
        alpha_transition = 1.0 - alpha_pref

        beta_distance = self.rng.random() * 5.0
        travel_time_scale = 0.1 + self.rng.random() * (5.0 - 0.1)

        lo = math.log(1e-6)
        hi = math.log(10.0)
        smoothing_strength = math.exp(lo + self.rng.random() * (hi - lo))

        stop_count_scale = 0.6 + self.rng.random() * 1.0

        dwell_mu_shift = -0.5 + self.rng.random() * 1.0
        dwell_sigma_mult = 0.8 + self.rng.random() * 0.4

        return SimulatorParameters(
            alpha_pref=float(alpha_pref),
            alpha_transition=float(alpha_transition),
            beta_distance=float(beta_distance),
            travel_time_scale=float(travel_time_scale),
            smoothing_strength=float(smoothing_strength),
            stop_count_scale=float(stop_count_scale),
            dwell_mu_shift=float(dwell_mu_shift),
            dwell_sigma_mult=float(dwell_sigma_mult),
        )

    def fit(self) -> Tuple[SimulatorParameters, float, List[Dict[str, object]]]:
        log: List[Dict[str, object]] = []
        best_params: Optional[SimulatorParameters] = None
        best_obj = float("inf")

        info(
            f"Calibration: random search iters={self.iters}, "
            f"proxy_days={len(self._proxy_targets)}/{len(self._all_targets)} (validation targets)"
        )

        for it in range(1, self.iters + 1):
            params = self._sample_params()

            _, sim_events_proxy = self.simulator.rollout(self.split, params, subset=self._proxy_targets)
            metrics, obj, _ = self.evaluator.compute_metrics(
                self._subset_split(self._proxy_targets),
                sim_events_proxy,
                objective_weights=self.objective_weights,
            )

            log.append(
                {
                    "iter": it,
                    "parameters": self._params_to_loggable(params),
                    "objective": float(obj),
                    "metrics": {k: float(v) for k, v in metrics.items()},
                    "notes": "proxy_validation_subsample",
                }
            )

            if obj < best_obj:
                best_obj = obj
                best_params = params

            if it % max(1, self.iters // 5) == 0:
                info(f"Calibration iter {it}/{self.iters}: best_objective_so_far={best_obj:.6f}")

        if best_params is None:
            raise RuntimeError("Calibration failed to produce any candidate parameters.")

        info("Re-scoring best proxy parameters on FULL validation set.")
        _, sim_events_full = self.simulator.rollout(self.split, best_params, subset=None)
        full_metrics, full_obj, _ = self.evaluator.compute_metrics(
            self.split, sim_events_full, objective_weights=self.objective_weights
        )

        log.append(
            {
                "iter": self.iters + 1,
                "parameters": self._params_to_loggable(best_params),
                "objective": float(full_obj),
                "metrics": {k: float(v) for k, v in full_metrics.items()},
                "notes": "full_validation_rescore_of_best_proxy",
            }
        )
        return best_params, float(full_obj), log

    def _subset_split(self, targets: List[Tuple[str, date]]) -> HoldoutSplit:
        val_by_agent: Dict[str, List[DayLog]] = {a: [] for a in self.split.val_by_agent.keys()}
        original: Dict[Tuple[str, date], DayLog] = {}
        for a, dlogs in self.split.val_by_agent.items():
            for dl in dlogs:
                original[(a, dl.day)] = dl
        for a, d in targets:
            dl = original.get((a, d))
            if dl is not None:
                val_by_agent.setdefault(a, []).append(dl)

        for a in val_by_agent:
            val_by_agent[a].sort(key=lambda dl: dl.day)

        return HoldoutSplit(
            train_by_agent=self.split.train_by_agent,
            val_by_agent=val_by_agent,
            excluded_agents=self.split.excluded_agents,
            meta={**self.split.meta, "subset_validation_targets": len(targets)},
        )

    @staticmethod
    def _params_to_loggable(p: SimulatorParameters) -> Dict[str, float]:
        return {
            "alpha_pref": p.alpha_pref,
            "alpha_transition": p.alpha_transition,
            "beta_distance": p.beta_distance,
            "travel_time_scale": p.travel_time_scale,
            "smoothing_strength": p.smoothing_strength,
            "stop_count_scale": p.stop_count_scale,
            "dwell_mu_shift": p.dwell_mu_shift,
            "dwell_sigma_mult": p.dwell_sigma_mult,
        }


# ----------------------------
# Saving outputs (exactly 4 files)
# ----------------------------

def save_results(
    output_dir: str,
    *,
    seed: int,
    best_params_json: Dict[str, object],
    best_objective: float,
    objective_definition: str,
    calibration_log: List[Dict[str, object]],
    evaluation_results: Dict[str, object],
    simulated_trajectories: Dict[str, object],
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    abs_out = os.path.abspath(output_dir)

    def dump(name: str, obj: object) -> None:
        path = os.path.join(abs_out, name)
        tmp_path = f"{path}.tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(obj, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, path)
        except Exception as e:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass
            raise OSError(f"Failed to write output file {path!r}: {e}") from e

    dump(
        "calibrated_parameters.json",
        {
            "best_parameters": best_params_json,
            "best_objective": float(best_objective),
            "objective_definition": objective_definition,
            "seed": int(seed),
            "meta": {"allowed_date_range_inclusive": [str(_ALLOWED_START), str(_ALLOWED_END)]},
        },
    )
    dump("calibration_log.json", calibration_log)
    dump("evaluation_results_on_validation.json", evaluation_results)
    dump("simulated_trajectories_validation.json", simulated_trajectories)

    print(f"[RESULT] wrote outputs to: {abs_out}")


# ----------------------------
# CLI / Main Orchestration
# ----------------------------

def parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Daily mobility multi-agent simulator (2019-2020 scoped).")
    p.add_argument("--output_dir", required=True, help="Absolute or relative output directory.")
    p.add_argument("--seed", type=int, default=123, help="Global random seed.")
    p.add_argument("--calib_iters", type=int, default=25, help="Calibration iterations (random search).")
    p.add_argument("--calib_subsample_days", type=int, default=300, help="Proxy validation agent-days per iteration.")
    p.add_argument("--tod_bins_train", type=int, default=48, help="Time-of-day bins for priors/simulation (e.g., 48=30min).")
    p.add_argument("--tod_bins_eval", type=int, default=144, help="Time-of-day bins for evaluation (e.g., 144=10min).")
    return p.parse_args()


def main() -> None:
    args = parse_cli()
    seed = int(args.seed)
    random.seed(seed)

    out_dir = args.output_dir
    if not out_dir:
        raise ValueError("--output_dir must be provided.")
    os.makedirs(out_dir, exist_ok=True)

    data_dir = _require_data_dir()
    info(f"DATA_DIR={os.path.abspath(data_dir)}")
    info(f"Output dir={os.path.abspath(out_dir)}")
    info(f"Seed={seed}")

    raw = load_data()
    parsed, env, _ = build_network_and_agents(raw, seed=seed)

    split = holdout_split(parsed)
    info(
        f"Holdout: agents_total={split.meta['num_agents_total']}, "
        f"agents_with_val={split.meta['num_agents_with_any_val']}, "
        f"excluded_from_val={split.meta['num_agents_excluded_from_val']}"
    )

    base_smoothing = 0.5
    agents = learn_agent_priors(env, split.train_by_agent, smoothing_strength=base_smoothing, tod_bins=int(args.tod_bins_train))
    info(f"Learned priors for {len(agents)} agents (TRAIN non-empty).")

    simulator = MobilitySimulator(env=env, agents=agents, tod_bins=int(args.tod_bins_train), global_seed=seed)
    evaluator = Evaluator(env=env, parsed=parsed, tod_bins_eval=int(args.tod_bins_eval))

    objective_weights = {
        "category_share_mae": 1.0,
        "stop_count_abs_mean_error": 0.5,
        "stop_count_kl": 0.5,
        "tod_jsd_avg": 1.0,
        "topk_poi_recall": 1.0,
        "transition_divergence": 1.0,
        "trip_distance_wasserstein": 0.5,
    }

    calibrator = RandomSearchCalibrator(
        simulator=simulator,
        evaluator=evaluator,
        split=split,
        agents=agents,
        env=env,
        seed=seed,
        iters=int(args.calib_iters),
        objective_weights=objective_weights,
        subsample_days=int(args.calib_subsample_days),
    )

    best_params, best_obj, calib_log = calibrator.fit()

    trajectories_by_agent, sim_events = simulator.rollout(split, best_params, subset=None)

    sim_metrics, objective, validation_set_summary = evaluator.compute_metrics(
        split, sim_events, objective_weights=objective_weights
    )

    global_categories = sorted(env.category_to_poi_ids.keys())
    representative_base = next(iter(agents.values())).priors.base_dwell_lognorm if agents else {}
    best_params_json = best_params.to_jsonable(global_categories, representative_base)

    simulated_trajectories = {
        "format_spec": "Activities at YYYY-MM-DD: POI#id at HH:MM:SS, POI#id at HH:MM:SS, ... . "
                       "Comma+space separators and ends with a single '.' (no preceding space).",
        "trajectories": trajectories_by_agent,
        "meta": {
            "allowed_date_range_inclusive": [str(_ALLOWED_START), str(_ALLOWED_END)],
            "seed": seed,
            "validation_targets_total": validation_set_summary["num_agent_days"],
        },
    }

    evaluation_results = {
        "simulation_metrics": {k: float(v) for k, v in sim_metrics.items()},
        "objective": float(objective),
        "objective_weights": {k: float(v) for k, v in objective_weights.items()},
        "validation_set": validation_set_summary,
        "meta": {
            "seed": seed,
            "scope_note": "All parsing, training, calibration, rollout, and evaluation use ONLY 2019-2020 inclusive.",
        },
    }

    objective_definition = (
        "Weighted sum over validation-only metrics (lower is better). "
        "Objective = Σ w_m * err_m, where err(topk_poi_recall)=1-recall."
    )

    save_results(
        out_dir,
        seed=seed,
        best_params_json=best_params_json,
        best_objective=best_obj,
        objective_definition=objective_definition,
        calibration_log=calib_log,
        evaluation_results=evaluation_results,
        simulated_trajectories=simulated_trajectories,
    )


main()