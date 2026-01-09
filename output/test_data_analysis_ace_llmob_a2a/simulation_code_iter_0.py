from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    # Preferred (openai>=1.x)
    from openai import OpenAI  # type: ignore
except ImportError:  # Prevent import-time failure if openai package isn't installed.
    OpenAI = None  # type: ignore


# -----------------------
# OpenAI LLM integration (required)
# -----------------------
def get_openai_api_key():
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        return api_key
    raise ValueError("OpenAI API key not found in environment")


def _make_openai_client(api_key: str):
    """
    Create an OpenAI client in a defensive way across potential SDK surfaces.
    Hard requirement: use Responses API when available.
    """
    if OpenAI is None:
        raise ImportError("openai package is not installed; cannot call OpenAI API")

    # openai>=1.x exposes OpenAI class callable
    try:
        client = OpenAI(api_key=api_key)  # type: ignore[misc]
    except TypeError as e:
        raise ImportError(
            "Installed openai package does not support `OpenAI(api_key=...)`. "
            "Please install/upgrade to openai>=1.x to use the Responses API."
        ) from e
    except Exception as e:
        raise RuntimeError(f"Failed to initialize OpenAI client: {e}") from e

    if not hasattr(client, "responses") or not hasattr(getattr(client, "responses"), "create"):
        raise ImportError(
            "Installed openai package/client does not expose `client.responses.create`. "
            "Please install/upgrade to openai>=1.x (Responses API)."
        )
    return client


def call_gpt5_with_responses_api(prompt: str, model: str = "gpt-5", max_output_tokens: int = 4000):
    api_key = get_openai_api_key()
    client = _make_openai_client(api_key=api_key)

    responses_kwargs = {
        "model": model,
        "input": [{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
        "max_output_tokens": max_output_tokens,
    }

    resp = client.responses.create(**responses_kwargs)

    def extract_response(resp_obj):
        if hasattr(resp_obj, "output_text") and isinstance(resp_obj.output_text, str):
            return resp_obj.output_text
        try:
            output = getattr(resp_obj, "output", None)
            if output and isinstance(output, list):
                first = output[0]
                content = first.get("content") if isinstance(first, dict) else None
                if content and isinstance(content, list) and len(content) > 0:
                    text = content[0].get("text")
                    if isinstance(text, str):
                        return text
        except Exception:
            pass
        return str(resp_obj)

    return extract_response(resp)


class MemoryAgent:
    """
    Minimal Memory Agent interface to retrieve user/item context.
    """

    def get_user_context(self, user_id: str) -> str:
        return f"user_id={user_id}"

    def get_item_context(self, item_id: str) -> str:
        return f"item_id={item_id}"


class PlanningAgent:
    """
    Minimal Planning Agent interface to produce a plan/task decomposition.
    """

    def build_plan(self, task: str) -> str:
        return (
            f"Task: {task}\nSteps:\n"
            "1) Consider user context.\n"
            "2) Consider item context.\n"
            "3) Write a helpful, grounded review."
        )


class ReviewAuthor:
    """
    Reasoning Agent that MUST perform reasoning via an LLM call.
    The LLM output is used as the primary review text.
    """

    def __init__(self, memory_agent: MemoryAgent, planning_agent: PlanningAgent) -> None:
        self.memory_agent = memory_agent
        self.planning_agent = planning_agent

    def generate(self, user_id: str, item_id: str, task: str = "Write a review with a star rating rationale") -> str:
        user_context = self.memory_agent.get_user_context(user_id)
        item_context = self.memory_agent.get_item_context(item_id)
        plan = self.planning_agent.build_plan(task)

        prompt = (
            "You are a review-writing assistant.\n\n"
            "USER CONTEXT:\n"
            f"{user_context}\n\n"
            "ITEM/PRODUCT CONTEXT:\n"
            f"{item_context}\n\n"
            "PLAN / TASK DECOMPOSITION:\n"
            f"{plan}\n\n"
            "Write the review body as the primary output text. Do not include metadata headers."
        )
        response = call_gpt5_with_responses_api(prompt=prompt, model="gpt-5", max_output_tokens=4000)
        return response.strip()


# -----------------------
# Logging (stdout only)
# -----------------------
def log_info(msg: str) -> None:
    print(f"[INFO] {msg}")


# -----------------------
# Globals / constants
# -----------------------
GLOBAL_SEED = 12345
RNG = random.Random(GLOBAL_SEED)

DATE_START_2021 = date(2021, 1, 1)
DATE_END_2021 = date(2021, 12, 31)

TOD_BIN_MINUTES = 15
TOD_NUM_BINS = 24 * 60 // TOD_BIN_MINUTES

ENCOUNTER_BIN_MINUTES = 10
TOPK_POI_RECALL_K = 50
DEFAULT_CANDIDATE_POOL_SIZE = 100


# -----------------------
# Data structures
# -----------------------
@dataclass(frozen=True)
class Visit:
    poi_token: str
    category: str
    time_min: int


@dataclass(frozen=True)
class Trajectory:
    user_id: str
    day: date
    visits: List[Visit]


@dataclass
class LoadedData:
    trajectories_by_user: Dict[str, Dict[date, Trajectory]]
    poi_catalog: Dict[str, Dict[str, Any]]
    pois_by_category: Dict[str, List[str]]
    coarse_category_map: Dict[str, str]
    meta: Dict[str, Any]


@dataclass
class SplitData:
    train_by_user: Dict[str, Dict[date, Trajectory]]
    validation_by_user: Dict[str, Dict[date, Trajectory]]
    meta: Dict[str, Any]


# -----------------------
# Utilities
# -----------------------
def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def safe_json_load(path: str) -> Any:
    require(os.path.isabs(path), f"Expected absolute path, got: {path}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing required data file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {path}: {e}") from e


def parse_hms_to_min(hms: str) -> int:
    m = re.match(r"^\s*(\d{1,2}):(\d{2}):(\d{2})\s*$", hms)
    require(m is not None, f"Invalid time token (expected HH:MM:SS), got: {hms!r}")
    hh = int(m.group(1))
    mm = int(m.group(2))
    ss = int(m.group(3))
    require(0 <= hh <= 23 and 0 <= mm <= 59 and 0 <= ss <= 59, f"Out-of-range time: {hms!r}")
    total_seconds = hh * 3600 + mm * 60 + ss
    minute = int((total_seconds + 30) // 60)
    if minute >= 1440:
        minute = 1439
    return minute


def format_min_to_hms(minute: int) -> str:
    require(0 <= minute <= 1439, f"Minute out of range: {minute}")
    hh = minute // 60
    mm = minute % 60
    return f"{hh:02d}:{mm:02d}:00"


def parse_date_yyyy_mm_dd(s: str) -> date:
    try:
        return datetime.strptime(s.strip(), "%Y-%m-%d").date()
    except ValueError as e:
        raise ValueError(f"Invalid date token (expected YYYY-MM-DD), got: {s!r}") from e


def is_2021(d: date) -> bool:
    return DATE_START_2021 <= d <= DATE_END_2021


def fine_category_from_poi_token(poi_token: str) -> str:
    if "#" not in poi_token:
        return poi_token.strip()
    return poi_token.split("#", 1)[0].strip()


def to_coarse_category(fine_category: str, coarse_map: Dict[str, str]) -> str:
    return coarse_map.get(fine_category, fine_category)


def stable_softmax(weights: List[float]) -> List[float]:
    require(len(weights) > 0, "softmax called with empty weights")
    m = max(weights)
    exps = [math.exp(w - m) for w in weights]
    s = sum(exps)
    if s <= 0.0 or not math.isfinite(s):
        return [1.0 / len(weights)] * len(weights)
    return [e / s for e in exps]


def sample_from_pmf(rng: random.Random, pmf: Dict[Any, float]) -> Any:
    require(len(pmf) > 0, "Cannot sample from empty PMF")
    total = sum(max(0.0, float(v)) for v in pmf.values())
    if total <= 0.0:
        items = list(pmf.keys())
        return items[rng.randrange(len(items))]
    r = rng.random() * total
    acc = 0.0
    for k, v in pmf.items():
        acc += max(0.0, float(v))
        if r <= acc:
            return k
    return next(iter(pmf.keys()))


def jensen_shannon_divergence(p: List[float], q: List[float], eps: float = 1e-12) -> float:
    require(len(p) == len(q), "JSD requires distributions of same length")
    ps = [max(eps, float(x)) for x in p]
    qs = [max(eps, float(x)) for x in q]
    sp = sum(ps)
    sq = sum(qs)
    ps = [x / sp for x in ps]
    qs = [x / sq for x in qs]
    m = [(ps[i] + qs[i]) / 2.0 for i in range(len(ps))]

    def kl(a: List[float], b: List[float]) -> float:
        out = 0.0
        for i in range(len(a)):
            out += a[i] * math.log(a[i] / b[i])
        return out

    return 0.5 * kl(ps, m) + 0.5 * kl(qs, m)


def kl_divergence(p: List[float], q: List[float], eps: float = 1e-12) -> float:
    require(len(p) == len(q), "KL requires distributions of same length")
    ps = [max(eps, float(x)) for x in p]
    qs = [max(eps, float(x)) for x in q]
    sp = sum(ps)
    sq = sum(qs)
    ps = [x / sp for x in ps]
    qs = [x / sq for x in qs]
    out = 0.0
    for i in range(len(ps)):
        out += ps[i] * math.log(ps[i] / qs[i])
    return out


def wasserstein_1d(u: List[float], v: List[float]) -> float:
    if len(u) == 0 and len(v) == 0:
        return 0.0
    if len(u) == 0 or len(v) == 0:
        return 1e3
    us = sorted(float(x) for x in u)
    vs = sorted(float(x) for x in v)
    n = max(len(us), len(vs))

    def quantile(samples: List[float], i: int, n_: int) -> float:
        if n_ == 1:
            return samples[0]
        t = i / (n_ - 1)
        pos = t * (len(samples) - 1)
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            return samples[lo]
        w = pos - lo
        return samples[lo] * (1 - w) + samples[hi] * w

    total = 0.0
    for i in range(n):
        total += abs(quantile(us, i, n) - quantile(vs, i, n))
    return total / n


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1 - a)))
    return r * c


# -----------------------
# Strict day-string parsing/serialization
# -----------------------
DAY_PREFIX_RE = re.compile(r"Activities\s+at\s+(\d{4}-\d{2}-\d{2})\s*:\s*(.*)\s*$", re.IGNORECASE)
VISIT_RE = re.compile(r"([^,;]+?)\s+at\s+(\d{1,2}:\d{2}:\d{2})", re.IGNORECASE)


def parse_day_record(user_id: str, record: str) -> Optional[Trajectory]:
    if not isinstance(record, str):
        return None
    record = record.strip()
    if not record:
        return None

    m = DAY_PREFIX_RE.match(record)
    if m is None:
        return None
    d = parse_date_yyyy_mm_dd(m.group(1))
    rest = m.group(2).strip()
    if rest.endswith("."):
        rest = rest[:-1].strip()

    visits: List[Visit] = []
    for vm in VISIT_RE.finditer(rest):
        poi_token = vm.group(1).strip()
        time_str = vm.group(2).strip()
        if not poi_token:
            continue
        try:
            tmin = parse_hms_to_min(time_str)
        except ValueError:
            continue
        cat = fine_category_from_poi_token(poi_token)
        visits.append(Visit(poi_token=poi_token, category=cat, time_min=tmin))

    if len(visits) == 0:
        return None

    visits_sorted = sorted(enumerate(visits), key=lambda it: (it[1].time_min, it[0]))
    visits = [v for _, v in visits_sorted]
    return Trajectory(user_id=user_id, day=d, visits=visits)


def serialize_day_record(day: date, poi_time_pairs: List[Tuple[str, int]]) -> str:
    require(isinstance(day, date), "serialize_day_record: 'day' must be a datetime.date")
    require(len(poi_time_pairs) > 0, "serialize_day_record: must have at least one visit")
    parts = []
    last_t = -1
    for poi, tmin in poi_time_pairs:
        require(isinstance(poi, str) and poi.strip(), "serialize_day_record: invalid POI token")
        require(isinstance(tmin, int), "serialize_day_record: time_min must be int")
        require(0 <= tmin <= 1439, "serialize_day_record: time_min out of range")
        if tmin < last_t:
            tmin = last_t
        last_t = tmin
        parts.append(f"{poi.strip()} at {format_min_to_hms(tmin)}")
    joined = ", ".join(parts)
    return f"Activities at {day.isoformat()}: {joined} ."


# -----------------------
# POI Environment
# -----------------------
class POIEnvironment:
    def __init__(
        self,
        poi_catalog: Dict[str, Dict[str, Any]],
        pois_by_category: Dict[str, List[str]],
        coarse_category_map: Dict[str, str],
    ) -> None:
        require(isinstance(poi_catalog, dict) and len(poi_catalog) > 0, "POI catalog is empty")
        require(isinstance(pois_by_category, dict) and len(pois_by_category) > 0, "POI category index empty")
        self.poi_catalog = poi_catalog
        self.pois_by_category = pois_by_category
        self.coarse_category_map = coarse_category_map

    def has_coords(self, poi_token: str) -> bool:
        rec = self.poi_catalog.get(poi_token)
        return rec is not None and isinstance(rec.get("lat"), (int, float)) and isinstance(rec.get("lon"), (int, float))

    def coords(self, poi_token: str) -> Optional[Tuple[float, float]]:
        rec = self.poi_catalog.get(poi_token)
        if rec is None:
            return None
        lat = rec.get("lat")
        lon = rec.get("lon")
        if isinstance(lat, (int, float)) and isinstance(lon, (int, float)):
            return float(lat), float(lon)
        return None

    def distance_km(self, poi_a: str, poi_b: str) -> Optional[float]:
        ca = self.coords(poi_a)
        cb = self.coords(poi_b)
        if ca is None or cb is None:
            return None
        return haversine_km(ca[0], ca[1], cb[0], cb[1])

    def candidates_for_category(
        self,
        category: str,
        k: int,
        anchor_poi: Optional[str] = None,
        rng: Optional[random.Random] = None,
    ) -> List[str]:
        require(k > 0, "Candidate pool size k must be positive")
        pool = self.pois_by_category.get(category)
        if not pool:
            pool = list(self.poi_catalog.keys())

        # Fix 2: Prioritize POIs with coordinates for better distance metric coverage
        candidates_with_coords = [p for p in pool if self.has_coords(p)]
        candidates_without_coords = [p for p in pool if not self.has_coords(p)]
        pool = candidates_with_coords + candidates_without_coords

        if anchor_poi is None or not self.has_coords(anchor_poi):
            if rng is None:
                rng = RNG
            if len(pool) <= k:
                return list(pool)
            # When no anchor, still prioritize POIs with coords but shuffle within each group
            if len(candidates_with_coords) >= k:
                rng.shuffle(candidates_with_coords)
                return candidates_with_coords[:k]
            else:
                rng.shuffle(candidates_with_coords)
                rng.shuffle(candidates_without_coords)
                combined = candidates_with_coords + candidates_without_coords
                return combined[:k]

        anchor_coords = self.coords(anchor_poi)
        if anchor_coords is None:
            return pool[: min(k, len(pool))]
        scored: List[Tuple[float, str]] = []
        for poi in pool:
            c = self.coords(poi)
            if c is None:
                continue
            d = haversine_km(anchor_coords[0], anchor_coords[1], c[0], c[1])
            scored.append((d, poi))
        if not scored:
            return pool[: min(k, len(pool))]
        scored.sort(key=lambda x: x[0])
        return [p for _, p in scored[: min(k, len(scored))]]


# -----------------------
# Agents
# -----------------------
@dataclass
class ResidentPattern:
    home_poi: Optional[str]
    work_poi: Optional[str]
    category_pref: Dict[str, float]
    poi_memory: Dict[str, int]
    start_time_hist: Dict[int, int]
    stop_count_hist: Dict[int, int]
    transition_pmf: Dict[str, Dict[str, float]]
    baseline_mean_stops: float


class ResidentAgent:
    def __init__(self, user_id: str, observed_2021: Dict[date, Trajectory]) -> None:
        require(isinstance(user_id, str) and user_id.strip(), "Invalid user_id")
        require(isinstance(observed_2021, dict), "observed_2021 must be a dict")
        self.user_id = user_id
        self.observed_2021 = observed_2021
        self.pattern: Optional[ResidentPattern] = None
        self.category_awareness: Dict[str, float] = {}

    def fit_pattern_from_training(
        self,
        train_days: Dict[date, Trajectory],
        env: POIEnvironment,
        topk_poi_memory: int = 50,
    ) -> None:
        require(len(train_days) > 0, f"User {self.user_id}: cannot fit pattern from empty training set")

        home_counts: Dict[str, int] = {}
        work_counts: Dict[str, int] = {}
        last_poi_counts: Dict[str, int] = {}

        cat_counts: Dict[str, int] = {}
        poi_counts: Dict[str, int] = {}
        start_hist: Dict[int, int] = {}
        stop_hist: Dict[int, int] = {}
        trans_counts: Dict[str, Dict[str, int]] = {}

        total_stops = 0
        total_days = 0

        for tr in train_days.values():
            if len(tr.visits) == 0:
                continue
            total_days += 1
            total_stops += len(tr.visits)
            stop_hist[len(tr.visits)] = stop_hist.get(len(tr.visits), 0) + 1
            start_hist[tr.visits[0].time_min] = start_hist.get(tr.visits[0].time_min, 0) + 1

            last_poi_counts[tr.visits[-1].poi_token] = last_poi_counts.get(tr.visits[-1].poi_token, 0) + 1

            for v in tr.visits:
                cat_counts[v.category] = cat_counts.get(v.category, 0) + 1
                poi_counts[v.poi_token] = poi_counts.get(v.poi_token, 0) + 1
                if v.poi_token.startswith("Home#"):
                    home_counts[v.poi_token] = home_counts.get(v.poi_token, 0) + 1
                if v.poi_token.startswith("Office#"):
                    work_counts[v.poi_token] = work_counts.get(v.poi_token, 0) + 1

            for i in range(len(tr.visits) - 1):
                c0 = tr.visits[i].category
                c1 = tr.visits[i + 1].category
                trans_counts.setdefault(c0, {})
                trans_counts[c0][c1] = trans_counts[c0].get(c1, 0) + 1

        require(total_days > 0, f"User {self.user_id}: no non-empty training trajectories")

        home_poi = max(home_counts.items(), key=lambda kv: kv[1])[0] if home_counts else None
        work_poi = max(work_counts.items(), key=lambda kv: kv[1])[0] if work_counts else None
        if home_poi is None:
            home_poi = max(last_poi_counts.items(), key=lambda kv: kv[1])[0] if last_poi_counts else None

        cat_total = sum(cat_counts.values())
        category_pref = {c: cnt / cat_total for c, cnt in cat_counts.items()} if cat_total > 0 else {}

        poi_sorted = sorted(poi_counts.items(), key=lambda kv: (-kv[1], kv[0]))
        poi_memory = dict(poi_sorted[:topk_poi_memory])

        transition_pmf: Dict[str, Dict[str, float]] = {}
        all_cats = set(cat_counts.keys())
        for c0, nexts in trans_counts.items():
            denom = 0.0
            pmf: Dict[str, float] = {}
            for c1 in all_cats:
                val = nexts.get(c1, 0) + 1
                pmf[c1] = float(val)
                denom += val
            if denom <= 0:
                continue
            transition_pmf[c0] = {k: v / denom for k, v in pmf.items()}

        baseline_mean_stops = total_stops / max(1, total_days)
        self.pattern = ResidentPattern(
            home_poi=home_poi,
            work_poi=work_poi,
            category_pref=category_pref,
            poi_memory=poi_memory,
            start_time_hist=start_hist,
            stop_count_hist=stop_hist,
            transition_pmf=transition_pmf,
            baseline_mean_stops=baseline_mean_stops,
        )

    def decay_awareness(self, decay: float = 0.5) -> None:
        require(0.0 < decay <= 1.0, "decay must be in (0,1]")
        for k in list(self.category_awareness.keys()):
            self.category_awareness[k] *= decay
            if self.category_awareness[k] < 1e-6:
                del self.category_awareness[k]


# -----------------------
# Simulator parameters
# -----------------------
@dataclass
class SimulatorParameters:
    alpha_dist: float
    alpha_pref: float
    alpha_anchor: float
    routine_strength: float
    exploration_rate: float
    travel_time_scale: float
    return_home_bias: float
    social_influence_strength: float
    global_mobility_noise_sigma: float
    time_of_day_category_profile: Dict[int, Dict[str, float]]
    dwell_time_mean_by_category: Dict[str, float]


def clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


# -----------------------
# Evaluator
# -----------------------
class Evaluator:
    def __init__(self, env: POIEnvironment, coarse_map: Dict[str, str]) -> None:
        self.env = env
        self.coarse_map = coarse_map

    def _collect_category_counts(self, trajs: Iterable[Trajectory]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for tr in trajs:
            for v in tr.visits:
                coarse = to_coarse_category(v.category, self.coarse_map)
                counts[coarse] = counts.get(coarse, 0) + 1
        return counts

    def _category_share_dist(self, counts: Dict[str, int]) -> Dict[str, float]:
        total = sum(counts.values())
        if total <= 0:
            return {}
        return {k: v / total for k, v in counts.items()}

    def _collect_stop_counts(self, trajs: Iterable[Trajectory]) -> List[int]:
        return [len(tr.visits) for tr in trajs]

    def _collect_tod_hist(self, trajs: Iterable[Trajectory]) -> List[float]:
        hist = [0.0] * TOD_NUM_BINS
        total = 0
        for tr in trajs:
            for v in tr.visits:
                b = int(v.time_min // TOD_BIN_MINUTES)
                b = max(0, min(TOD_NUM_BINS - 1, b))
                hist[b] += 1.0
                total += 1
        if total == 0:
            return [1.0 / TOD_NUM_BINS] * TOD_NUM_BINS
        return [x / total for x in hist]

    def _topk_pois(self, trajs: Iterable[Trajectory], k: int) -> List[str]:
        counts: Dict[str, int] = {}
        for tr in trajs:
            for v in tr.visits:
                counts[v.poi_token] = counts.get(v.poi_token, 0) + 1
        items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        return [p for p, _ in items[:k]]

    def _collect_transitions(self, trajs: Iterable[Trajectory]) -> Dict[Tuple[str, str], int]:
        trans: Dict[Tuple[str, str], int] = {}
        for tr in trajs:
            cats = [to_coarse_category(v.category, self.coarse_map) for v in tr.visits]
            for i in range(len(cats) - 1):
                key = (cats[i], cats[i + 1])
                trans[key] = trans.get(key, 0) + 1
        return trans

    def _transition_jsd(self, gt: Dict[Tuple[str, str], int], sim: Dict[Tuple[str, str], int]) -> float:
        keys = sorted(set(gt.keys()) | set(sim.keys()))
        if not keys:
            return 0.0
        p = [gt.get(k, 0) for k in keys]
        q = [sim.get(k, 0) for k in keys]
        sp = sum(p)
        sq = sum(q)
        if sp == 0:
            p = [1.0 / len(keys)] * len(keys)
        else:
            p = [x / sp for x in p]
        if sq == 0:
            q = [1.0 / len(keys)] * len(keys)
        else:
            q = [x / sq for x in q]
        return jensen_shannon_divergence(p, q)

    def _trip_distances(self, trajs: Iterable[Trajectory]) -> Tuple[List[float], Dict[str, Any]]:
        dists: List[float] = []
        eligible_pairs = 0
        matched_pairs = 0
        for tr in trajs:
            for i in range(len(tr.visits) - 1):
                eligible_pairs += 1
                a = tr.visits[i].poi_token
                b = tr.visits[i + 1].poi_token
                dist = self.env.distance_km(a, b)
                if dist is None:
                    continue
                matched_pairs += 1
                dists.append(dist)
        meta = {
            "eligible_pairs": eligible_pairs,
            "matched_pairs_with_coords": matched_pairs,
            "coord_match_rate": (matched_pairs / eligible_pairs) if eligible_pairs > 0 else None,
        }
        return dists, meta

    def compute_metrics(
        self,
        simulated: List[Trajectory],
        ground_truth: List[Trajectory],
        objective_weights: Dict[str, float],
    ) -> Tuple[Dict[str, float], float, Dict[str, Any]]:
        require(len(ground_truth) > 0, "No ground-truth trajectories provided for metric computation")
        require(len(simulated) > 0, "No simulated trajectories provided for metric computation")

        gt_cat_counts = self._collect_category_counts(ground_truth)
        sim_cat_counts = self._collect_category_counts(simulated)
        gt_share = self._category_share_dist(gt_cat_counts)
        sim_share = self._category_share_dist(sim_cat_counts)
        cats = sorted(set(gt_share.keys()) | set(sim_share.keys()))
        if not cats:
            category_share_mae = 0.0
        else:
            category_share_mae = sum(abs(gt_share.get(c, 0.0) - sim_share.get(c, 0.0)) for c in cats) / len(cats)

        gt_counts = self._collect_stop_counts(ground_truth)
        sim_counts = self._collect_stop_counts(simulated)
        gt_mean = sum(gt_counts) / max(1, len(gt_counts))
        sim_mean = sum(sim_counts) / max(1, len(sim_counts))
        stop_count_abs_mean_error = abs(gt_mean - sim_mean)

        max_count = max(max(gt_counts, default=0), max(sim_counts, default=0))
        if max_count <= 0:
            stop_count_kl = 0.0
        else:
            support = list(range(1, max_count + 1))
            eps = 1e-6
            gt_hist = {k: eps for k in support}
            sim_hist = {k: eps for k in support}
            for c in gt_counts:
                if c in gt_hist:
                    gt_hist[c] += 1.0
            for c in sim_counts:
                if c in sim_hist:
                    sim_hist[c] += 1.0
            p = [gt_hist[k] for k in support]
            q = [sim_hist[k] for k in support]
            stop_count_kl = kl_divergence(p, q)

        gt_tod = self._collect_tod_hist(ground_truth)
        sim_tod = self._collect_tod_hist(simulated)
        tod_jsd_avg = jensen_shannon_divergence(gt_tod, sim_tod)

        gt_topk = self._topk_pois(ground_truth, TOPK_POI_RECALL_K)
        gt_k_eff = max(1, min(TOPK_POI_RECALL_K, len(gt_topk)))
        sim_poi_set = set(v.poi_token for tr in simulated for v in tr.visits)
        hit = sum(1 for p in gt_topk[:gt_k_eff] if p in sim_poi_set)
        topk_poi_recall = hit / gt_k_eff

        gt_trans = self._collect_transitions(ground_truth)
        sim_trans = self._collect_transitions(simulated)
        transition_divergence = self._transition_jsd(gt_trans, sim_trans)

        gt_dists, gt_dist_meta = self._trip_distances(ground_truth)
        sim_dists, sim_dist_meta = self._trip_distances(simulated)
        trip_distance_wasserstein = wasserstein_1d(gt_dists, sim_dists)

        metrics = {
            "category_share_mae": float(category_share_mae),
            "stop_count_abs_mean_error": float(stop_count_abs_mean_error),
            "stop_count_kl": float(stop_count_kl),
            "tod_jsd_avg": float(tod_jsd_avg),
            "topk_poi_recall": float(topk_poi_recall),
            "transition_divergence": float(transition_divergence),
            "trip_distance_wasserstein": float(trip_distance_wasserstein),
        }

        obj = 0.0
        for k, w in objective_weights.items():
            if k == "topk_poi_recall":
                obj += w * (1.0 - metrics[k])
            else:
                obj += w * metrics[k]

        meta = {
            "gt_trip_distance_meta": gt_dist_meta,
            "sim_trip_distance_meta": sim_dist_meta,
            "gt_num_days": len(ground_truth),
            "sim_num_days": len(simulated),
        }
        return metrics, float(obj), meta


# -----------------------
# Simulator
# -----------------------
class MobilitySimulator:
    def __init__(
        self,
        env: POIEnvironment,
        agents: Dict[str, ResidentAgent],
        base_parameters: SimulatorParameters,
        seed: int,
    ) -> None:
        require(isinstance(agents, dict) and len(agents) > 0, "No agents provided")
        self.env = env
        self.agents = agents
        self.parameters = base_parameters
        self.seed = int(seed)
        self.rng = random.Random(self.seed)

    def set_parameters(self, params: SimulatorParameters) -> None:
        self.parameters = params

    def _day_exogenous(self, d: date) -> Dict[str, int]:
        return {"day_of_week": d.weekday(), "month": d.month}

    def _compute_motivation(self, agent: ResidentAgent, target_day: date) -> Dict[str, float]:
        require(agent.pattern is not None, "Agent pattern must be fit before computing motivation")
        obs = agent.observed_2021
        days = [(target_day - timedelta(days=i)) for i in range(1, 8)]
        if any(d not in obs for d in days):
            return {"stop_multiplier": 1.0, "routine_delta": 0.0, "exploration_delta": 0.0, "return_home_delta": 0.0}

        last7 = [obs[d] for d in sorted(days)]
        avg_stops = sum(len(tr.visits) for tr in last7) / 7.0
        baseline = max(1e-6, agent.pattern.baseline_mean_stops)
        ratio = avg_stops / baseline

        stop_multiplier = clip(0.8 + 0.4 * ratio, 0.5, 1.5)
        routine_delta = clip((1.0 - ratio) * 0.2, -0.15, 0.25)
        exploration_delta = clip((ratio - 1.0) * 0.2, -0.2, 0.2)
        return_home_delta = clip((1.0 - ratio) * 0.3, -0.2, 0.4)

        return {
            "stop_multiplier": float(stop_multiplier),
            "routine_delta": float(routine_delta),
            "exploration_delta": float(exploration_delta),
            "return_home_delta": float(return_home_delta),
        }

    def _sample_start_time(self, agent: ResidentAgent) -> int:
        require(agent.pattern is not None, "Agent pattern must be fitted")
        hist = agent.pattern.start_time_hist
        if not hist:
            return int(8 * 60)
        total = sum(hist.values())
        r = self.rng.random() * total
        acc = 0.0
        for tmin, cnt in sorted(hist.items()):
            acc += cnt
            if r <= acc:
                return int(clip(tmin, 0, 1439))
        return int(clip(sorted(hist.keys())[0], 0, 1439))

    def _sample_stop_count(self, agent: ResidentAgent, motivation: Dict[str, float]) -> int:
        require(agent.pattern is not None, "Agent pattern must be fitted")
        hist = agent.pattern.stop_count_hist
        if not hist:
            base = 4
            return max(1, int(round(base * motivation.get("stop_multiplier", 1.0))))

        counts = sorted(hist.keys())
        probs = [hist[c] for c in counts]
        s = sum(probs)
        probs = [p / s for p in probs]

        mult = float(motivation.get("stop_multiplier", 1.0))
        mult = clip(mult, 0.5, 1.5)
        logw = [
            math.log(p + 1e-12) + math.log(mult) * (c / max(1.0, agent.pattern.baseline_mean_stops))
            for p, c in zip(probs, counts)
        ]
        pmf = stable_softmax(logw)
        r = self.rng.random()
        acc = 0.0
        sampled = counts[-1]
        for c, p in zip(counts, pmf):
            acc += p
            if r <= acc:
                sampled = c
                break

        min_gap = 5
        max_feasible = max(1, 1 + (1439 - 0) // min_gap)
        sampled = int(clip(sampled, 1, max_feasible))
        return sampled

    def _category_time_profile(self, tmin: int) -> Dict[str, float]:
        b = int(tmin // TOD_BIN_MINUTES)
        b = max(0, min(TOD_NUM_BINS - 1, b))
        prof = self.parameters.time_of_day_category_profile.get(b)
        return prof or {}

    def _choose_next_category(
        self,
        agent: ResidentAgent,
        current_category: Optional[str],
        tmin: int,
        exo: Dict[str, int],
        motivation: Dict[str, float],
    ) -> str:
        require(agent.pattern is not None, "Agent pattern must be fitted")
        routine_strength = clip(self.parameters.routine_strength + motivation.get("routine_delta", 0.0), 0.0, 1.0)

        trans: Dict[str, float] = {}
        if current_category is not None:
            trans = agent.pattern.transition_pmf.get(current_category, {})

        prof = self._category_time_profile(tmin)

        if not trans and not prof:
            base = agent.pattern.category_pref or {"Other": 1.0}
            return sample_from_pmf(self.rng, base)
        if not trans:
            return sample_from_pmf(self.rng, prof)
        if not prof:
            return sample_from_pmf(self.rng, trans)

        cats = sorted(set(trans.keys()) | set(prof.keys()))
        eps = 1e-9
        logw: List[float] = []
        for c in cats:
            p_trans = trans.get(c, 0.0) + eps
            p_prof = prof.get(c, 0.0) + eps
            p_mix = routine_strength * p_trans + (1.0 - routine_strength) * p_prof

            if exo["day_of_week"] in (5, 6) and c.lower().startswith("office"):
                p_mix *= 0.5

            coarse = to_coarse_category(c, self.env.coarse_category_map)
            aw = agent.category_awareness.get(coarse, 0.0)
            p_mix *= (1.0 + self.parameters.social_influence_strength * aw)

            logw.append(math.log(p_mix))

        pmf = stable_softmax(logw)
        r = self.rng.random()
        acc = 0.0
        for c, p in zip(cats, pmf):
            acc += p
            if r <= acc:
                return c
        return cats[-1]

    def _choose_next_poi(self, agent: ResidentAgent, category: str, current_poi: Optional[str]) -> str:
        require(agent.pattern is not None, "Agent pattern must be fitted")
        pat = agent.pattern

        mem_in_cat = [p for p in pat.poi_memory.keys() if fine_category_from_poi_token(p) == category]
        anchor = pat.home_poi or current_poi
        candidates = self.env.candidates_for_category(category, k=DEFAULT_CANDIDATE_POOL_SIZE, anchor_poi=anchor, rng=self.rng)
        merged = list(dict.fromkeys(mem_in_cat + candidates))
        if not merged:
            merged = list(self.env.poi_catalog.keys())

        exploration = clip(self.parameters.exploration_rate, 0.0, 1.0)
        if mem_in_cat and self.rng.random() > exploration:
            pool = mem_in_cat[: min(len(mem_in_cat), DEFAULT_CANDIDATE_POOL_SIZE)]
        else:
            pool = merged[: min(len(merged), DEFAULT_CANDIDATE_POOL_SIZE)]

        # Fix 3: Prioritize POIs with coordinates in the pool for better distance metrics
        pool_with_coords = [p for p in pool if self.env.has_coords(p)]
        pool_without_coords = [p for p in pool if not self.env.has_coords(p)]
        # If we have enough POIs with coords, prefer them; otherwise include some without
        if len(pool_with_coords) >= min(30, len(pool) // 2):
            pool = pool_with_coords + pool_without_coords[:max(0, DEFAULT_CANDIDATE_POOL_SIZE - len(pool_with_coords))]
        # else keep original pool order

        alpha_dist = clip(self.parameters.alpha_dist, 0.0, 10.0)
        alpha_pref = clip(self.parameters.alpha_pref, 0.0, 10.0)
        alpha_anchor = clip(self.parameters.alpha_anchor, 0.0, 10.0)
        noise_sigma = clip(self.parameters.global_mobility_noise_sigma, 0.0, 5.0)

        # Default distance penalty for POIs without coordinates (mild penalty to prefer those with coords)
        default_distance_penalty = 5.0  # km equivalent penalty for missing coords

        util: List[float] = []
        for poi in pool:
            u = 0.0
            cnt = pat.poi_memory.get(poi, 0)
            if cnt > 0:
                u += alpha_pref * math.log(1.0 + cnt)

            # Distance from current POI
            if current_poi is not None:
                dcur = self.env.distance_km(current_poi, poi)
                if dcur is not None:
                    u -= alpha_dist * dcur
                elif not self.env.has_coords(poi):
                    # Mild penalty for POIs without coordinates to prefer those with coords
                    u -= alpha_dist * default_distance_penalty * 0.3

            # Distance from home/anchor
            if pat.home_poi is not None:
                danc = self.env.distance_km(pat.home_poi, poi)
                if danc is not None:
                    u -= alpha_anchor * danc
                elif not self.env.has_coords(poi):
                    u -= alpha_anchor * default_distance_penalty * 0.3

            if noise_sigma > 0:
                u += self.rng.gauss(0.0, noise_sigma)

            util.append(u)

        pmf = stable_softmax(util)
        r = self.rng.random()
        acc = 0.0
        for poi, p in zip(pool, pmf):
            acc += p
            if r <= acc:
                return poi
        return pool[-1]

    def _sample_dwell_minutes(self, category: str) -> int:
        mean = float(self.parameters.dwell_time_mean_by_category.get(category, 45.0))
        mean = clip(mean, 5.0, 240.0)
        sigma = 0.5
        mu = math.log(mean) - 0.5 * sigma * sigma
        x = math.exp(self.rng.gauss(mu, sigma))
        return int(clip(round(x), 5, 240))

    def _travel_minutes(self, from_poi: Optional[str], to_poi: str) -> int:
        if from_poi is None:
            return 0
        d = self.env.distance_km(from_poi, to_poi)
        if d is None:
            return 0
        scale = clip(self.parameters.travel_time_scale, 0.0, 60.0)
        return int(clip(round(scale * d), 0, 180))

    def generate_day(self, agent: ResidentAgent, day: date) -> Trajectory:
        require(agent.pattern is not None, "Agent pattern must be fitted before simulation")
        pat = agent.pattern
        exo = self._day_exogenous(day)
        motivation = self._compute_motivation(agent, day)

        start_time = self._sample_start_time(agent)
        stop_count = self._sample_stop_count(agent, motivation)

        current_poi = pat.home_poi
        current_cat = fine_category_from_poi_token(current_poi) if current_poi else None
        t = int(clip(start_time, 0, 1439))

        visits: List[Visit] = []
        recent_pois: List[str] = []

        if current_poi is not None:
            visits.append(Visit(poi_token=current_poi, category=fine_category_from_poi_token(current_poi), time_min=t))
            recent_pois.append(current_poi)

        min_gap = 5
        remaining_budget = 1439 - t
        remaining_steps = stop_count - len(visits)
        if remaining_steps > 0:
            max_additional = remaining_budget // min_gap
            remaining_steps = min(remaining_steps, max_additional)

        for _ in range(remaining_steps):
            next_cat = self._choose_next_category(agent, current_cat, t, exo, motivation)
            next_poi = self._choose_next_poi(agent, next_cat, current_poi)

            if recent_pois and next_poi == recent_pois[-1]:
                alt_poi = self._choose_next_poi(agent, next_cat, current_poi)
                if alt_poi != next_poi:
                    next_poi = alt_poi

            dwell = self._sample_dwell_minutes(next_cat)
            travel = self._travel_minutes(current_poi, next_poi)

            t_next = t + max(min_gap, dwell + travel)
            if t_next > 1439:
                break

            visits.append(Visit(poi_token=next_poi, category=next_cat, time_min=int(t_next)))
            recent_pois.append(next_poi)
            recent_pois = recent_pois[-5:]
            current_poi = next_poi
            current_cat = next_cat
            t = int(t_next)

        if pat.home_poi is not None and len(visits) > 0:
            bias = clip(self.parameters.return_home_bias + motivation.get("return_home_delta", 0.0), 0.0, 5.0)
            late = max(0.0, (t - 18 * 60) / (6 * 60))
            p_home = clip(late * (bias / (1.0 + bias)), 0.0, 0.9)
            if self.rng.random() < p_home:
                if visits[-1].poi_token != pat.home_poi and t + 5 <= 1439:
                    t_home = min(1439, t + 5)
                    visits.append(
                        Visit(
                            poi_token=pat.home_poi,
                            category=fine_category_from_poi_token(pat.home_poi),
                            time_min=int(t_home),
                        )
                    )

        if len(visits) == 0:
            any_poi = next(iter(self.env.poi_catalog.keys()))
            visits = [Visit(poi_token=any_poi, category=fine_category_from_poi_token(any_poi), time_min=8 * 60)]

        return Trajectory(user_id=agent.user_id, day=day, visits=visits)

    def rollout(self, target_by_user: Dict[str, Dict[date, Trajectory]]) -> Tuple[Dict[str, List[str]], List[Trajectory]]:
        targets: List[Tuple[date, str]] = []
        for uid, dmap in target_by_user.items():
            for d in dmap.keys():
                targets.append((d, uid))
        targets.sort(key=lambda x: (x[0].toordinal(), x[1]))

        simulated: List[Trajectory] = []
        serialized: Dict[str, List[str]] = {uid: [] for uid in target_by_user.keys()}

        idx = 0
        while idx < len(targets):
            day = targets[idx][0]
            day_uids: List[str] = []
            while idx < len(targets) and targets[idx][0] == day:
                day_uids.append(targets[idx][1])
                idx += 1

            day_trajs: List[Trajectory] = []
            for uid in day_uids:
                agent = self.agents.get(uid)
                if agent is None:
                    continue
                tr = self.generate_day(agent, day)
                day_trajs.append(tr)
                simulated.append(tr)
                s = serialize_day_record(day, [(v.poi_token, v.time_min) for v in tr.visits])
                serialized[uid].append(s)

            occupancy: Dict[Tuple[str, int], List[str]] = {}
            for tr in day_trajs:
                for v in tr.visits:
                    b = int(v.time_min // ENCOUNTER_BIN_MINUTES)
                    key = (v.poi_token, b)
                    occupancy.setdefault(key, []).append(tr.user_id)

            for (poi, _b), uids in occupancy.items():
                if len(uids) < 2:
                    continue
                cat = fine_category_from_poi_token(poi)
                coarse = to_coarse_category(cat, self.env.coarse_category_map)
                for uid in set(uids):
                    ag = self.agents.get(uid)
                    if ag is None:
                        continue
                    ag.category_awareness[coarse] = ag.category_awareness.get(coarse, 0.0) + 0.1

            for uid in day_uids:
                ag = self.agents.get(uid)
                if ag is not None:
                    ag.decay_awareness(decay=0.6)

        return serialized, simulated


# -----------------------
# Calibration
# -----------------------
class BaseCalibrator:
    def fit(
        self,
        simulator: MobilitySimulator,
        agents: Dict[str, ResidentAgent],
        train_by_user: Dict[str, Dict[date, Trajectory]],
        evaluator: Evaluator,
        objective_weights: Dict[str, float],
    ) -> Tuple[SimulatorParameters, float, List[Dict[str, Any]]]:
        raise NotImplementedError


class RandomSearchCalibrator(BaseCalibrator):
    def __init__(self, n_iter: int, calibration_day_budget: int, seed: int) -> None:
        require(n_iter > 0, "n_iter must be positive")
        require(calibration_day_budget > 0, "calibration_day_budget must be positive")
        self.n_iter = int(n_iter)
        self.calibration_day_budget = int(calibration_day_budget)
        self.seed = int(seed)
        self.rng = random.Random(self.seed)

    def _deterministic_subsample(
        self, train_by_user: Dict[str, Dict[date, Trajectory]], budget: int
    ) -> Dict[str, Dict[date, Trajectory]]:
        items: List[Tuple[str, date]] = []
        for uid, dmap in train_by_user.items():
            for d in dmap.keys():
                items.append((uid, d))
        items.sort(key=lambda x: (x[0], x[1].toordinal()))
        self.rng.shuffle(items)
        items = items[: min(budget, len(items))]

        out: Dict[str, Dict[date, Trajectory]] = {}
        for uid, d in items:
            out.setdefault(uid, {})[d] = train_by_user[uid][d]
        return out

    def _propose_params(self, base: SimulatorParameters) -> SimulatorParameters:
        alpha_dist = self.rng.uniform(0.0, 10.0)
        alpha_pref = self.rng.uniform(0.0, 10.0)
        routine_strength = self.rng.uniform(0.0, 1.0)
        exploration_rate = self.rng.uniform(0.0, 1.0)
        travel_time_scale = self.rng.uniform(0.0, 60.0)
        return_home_bias = self.rng.uniform(0.0, 5.0)
        social_influence_strength = self.rng.uniform(0.0, 2.0)
        global_noise_sigma = self.rng.uniform(0.0, 2.0)
        alpha_anchor = 0.5 * alpha_dist

        return SimulatorParameters(
            alpha_dist=alpha_dist,
            alpha_pref=alpha_pref,
            alpha_anchor=alpha_anchor,
            routine_strength=routine_strength,
            exploration_rate=exploration_rate,
            travel_time_scale=travel_time_scale,
            return_home_bias=return_home_bias,
            social_influence_strength=social_influence_strength,
            global_mobility_noise_sigma=global_noise_sigma,
            time_of_day_category_profile=base.time_of_day_category_profile,
            dwell_time_mean_by_category=base.dwell_time_mean_by_category,
        )

    def fit(
        self,
        simulator: MobilitySimulator,
        agents: Dict[str, ResidentAgent],
        train_by_user: Dict[str, Dict[date, Trajectory]],
        evaluator: Evaluator,
        objective_weights: Dict[str, float],
    ) -> Tuple[SimulatorParameters, float, List[Dict[str, Any]]]:
        require(len(train_by_user) > 0, "Training dataset is empty")
        subsampled_train = self._deterministic_subsample(train_by_user, self.calibration_day_budget)

        for uid, agent in agents.items():
            if uid in train_by_user and len(train_by_user[uid]) > 0:
                agent.fit_pattern_from_training(train_by_user[uid], simulator.env)

        best_params = simulator.parameters
        best_obj = float("inf")
        cal_log: List[Dict[str, Any]] = []

        gt_trajs: List[Trajectory] = []
        for uid, dmap in subsampled_train.items():
            for tr in dmap.values():
                gt_trajs.append(tr)

        for it in range(1, self.n_iter + 1):
            params = self._propose_params(simulator.parameters)
            simulator.set_parameters(params)

            _sim_serialized, sim_trajs = simulator.rollout(subsampled_train)
            metrics, obj, meta = evaluator.compute_metrics(sim_trajs, gt_trajs, objective_weights)

            notes = f"subsample_days={sum(len(v) for v in subsampled_train.values())}; meta={meta}"
            cal_log.append(
                {
                    "iter": int(it),
                    "parameters": {
                        "alpha_dist": params.alpha_dist,
                        "alpha_pref": params.alpha_pref,
                        "alpha_anchor": params.alpha_anchor,
                        "routine_strength": params.routine_strength,
                        "exploration_rate": params.exploration_rate,
                        "travel_time_scale": params.travel_time_scale,
                        "return_home_hazard_params": {"bias": params.return_home_bias},
                        "social_influence_strength": params.social_influence_strength,
                        "global_mobility_noise_sigma": params.global_mobility_noise_sigma,
                    },
                    "objective": float(obj),
                    "metrics": metrics,
                    "notes": notes,
                }
            )

            if obj < best_obj:
                best_obj = obj
                best_params = params

            if it == 1 or it % max(1, self.n_iter // 5) == 0:
                log_info(f"calibration iter={it}/{self.n_iter} objective={obj:.6f} best={best_obj:.6f}")

        simulator.set_parameters(best_params)
        return best_params, float(best_obj), cal_log


# -----------------------
# Data ingestion and derived global profiles
# -----------------------
def _load_poi_catalog(poi_json: Any) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, List[str]]]:
    require(isinstance(poi_json, dict) and len(poi_json) > 0, "POI JSON must be a non-empty object")
    poi_catalog: Dict[str, Dict[str, Any]] = {}
    by_cat: Dict[str, List[str]] = {}
    bad = 0
    for cat, entries in poi_json.items():
        if not isinstance(cat, str) or not isinstance(entries, list):
            continue
        for e in entries:
            if not (isinstance(e, list) and len(e) >= 3):
                bad += 1
                continue
            lat, lon, token = e[0], e[1], e[2]
            if not isinstance(token, str) or "#" not in token:
                bad += 1
                continue
            try:
                latf = float(lat)
                lonf = float(lon)
            except (TypeError, ValueError):
                bad += 1
                continue
            fine_cat = fine_category_from_poi_token(token)
            poi_catalog[token] = {"category": fine_cat, "lat": latf, "lon": lonf}
            by_cat.setdefault(fine_cat, []).append(token)

    require(len(poi_catalog) > 0, "POI catalog parsed empty; check poi_category_192021_longitude_latitude.json format")
    for c in list(by_cat.keys()):
        by_cat[c] = list(dict.fromkeys(by_cat[c]))
    if bad > 0:
        log_info(f"POI catalog: skipped {bad} malformed entries")
    return poi_catalog, by_cat


def _load_catto_map(catto_json: Any) -> Dict[str, str]:
    if not isinstance(catto_json, dict):
        return {}
    out: Dict[str, str] = {}
    for k, v in catto_json.items():
        if isinstance(k, str) and isinstance(v, str) and k.strip() and v.strip():
            out[k.strip()] = v.strip()
    return out


def _compute_global_time_profile(train_by_user: Dict[str, Dict[date, Trajectory]]) -> Dict[int, Dict[str, float]]:
    counts: Dict[int, Dict[str, int]] = {}
    for dmap in train_by_user.values():
        for tr in dmap.values():
            for v in tr.visits:
                b = int(v.time_min // TOD_BIN_MINUTES)
                b = max(0, min(TOD_NUM_BINS - 1, b))
                counts.setdefault(b, {})
                counts[b][v.category] = counts[b].get(v.category, 0) + 1

    profile: Dict[int, Dict[str, float]] = {}
    for b in range(TOD_NUM_BINS):
        c = counts.get(b, {})
        if not c:
            continue
        s = sum(c.values())
        if s <= 0:
            continue
        profile[b] = {k: v / s for k, v in c.items()}
    return profile


def _compute_dwell_means_by_category(train_by_user: Dict[str, Dict[date, Trajectory]]) -> Dict[str, float]:
    sums: Dict[str, float] = {}
    cnts: Dict[str, int] = {}
    for dmap in train_by_user.values():
        for tr in dmap.values():
            vs = tr.visits
            for i in range(len(vs) - 1):
                gap = max(1, vs[i + 1].time_min - vs[i].time_min)
                cat = vs[i].category
                sums[cat] = sums.get(cat, 0.0) + float(gap)
                cnts[cat] = cnts.get(cat, 0) + 1
    out: Dict[str, float] = {}
    for cat, s in sums.items():
        m = s / max(1, cnts.get(cat, 0))
        out[cat] = float(clip(m, 10.0, 180.0))
    return out


# -----------------------
# Required pipeline functions
# -----------------------
def parse_cli(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="2021-only multi-agent mobility simulator")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to write output JSON files")
    return parser.parse_args(argv)


def load_data() -> LoadedData:
    PROJECT_ROOT = os.environ.get("PROJECT_ROOT")
    DATA_PATH = os.environ.get("DATA_PATH")
    require(PROJECT_ROOT is not None and PROJECT_ROOT.strip(), "PROJECT_ROOT env var must be set")
    require(DATA_PATH is not None and DATA_PATH.strip(), "DATA_PATH env var must be set")

    data_dir = os.path.join(PROJECT_ROOT, DATA_PATH)
    require(os.path.isabs(data_dir), f"DATA_DIR must be absolute; got: {data_dir}")

    path_1921 = os.path.abspath(os.path.join(data_dir, "1921Y.json"))
    path_poi = os.path.abspath(os.path.join(data_dir, "poi_category_192021_longitude_latitude.json"))
    path_catto = os.path.abspath(os.path.join(data_dir, "catto.json"))

    log_info(f"Loading data from DATA_DIR={data_dir}")
    raw_1921 = safe_json_load(path_1921)
    raw_poi = safe_json_load(path_poi)
    raw_catto = safe_json_load(path_catto)

    poi_catalog, pois_by_category = _load_poi_catalog(raw_poi)
    coarse_map = _load_catto_map(raw_catto)

    require(isinstance(raw_1921, dict), "1921Y.json must be a JSON object mapping user_id to day records")
    trajectories_by_user: Dict[str, Dict[date, Trajectory]] = {}
    total_records = 0
    parsed_ok = 0
    filtered_out_non2021 = 0
    unparsable = 0

    for uid, records in raw_1921.items():
        if not isinstance(uid, str):
            continue
        if isinstance(records, dict):
            rec_list = list(records.values())
        elif isinstance(records, list):
            rec_list = records
        else:
            continue

        for rec in rec_list:
            total_records += 1
            tr = parse_day_record(uid, rec if isinstance(rec, str) else "")
            if tr is None:
                unparsable += 1
                continue
            if not is_2021(tr.day):
                filtered_out_non2021 += 1
                continue
            parsed_ok += 1
            trajectories_by_user.setdefault(uid, {})
            trajectories_by_user[uid][tr.day] = tr

    trajectories_by_user = {u: dmap for u, dmap in trajectories_by_user.items() if len(dmap) > 0}
    require(len(trajectories_by_user) > 0, "No valid 2021 trajectories found after strict filtering")

    meta = {
        "total_day_records_seen": total_records,
        "parsed_ok_2021": parsed_ok,
        "filtered_out_non2021": filtered_out_non2021,
        "unparsable_records": unparsable,
        "num_users_with_2021_data": len(trajectories_by_user),
        "seed": GLOBAL_SEED,
    }
    log_info(f"Loaded 2021-only trajectories: users={meta['num_users_with_2021_data']} parsed_ok_2021={parsed_ok}")
    return LoadedData(
        trajectories_by_user=trajectories_by_user,
        poi_catalog=poi_catalog,
        pois_by_category=pois_by_category,
        coarse_category_map=coarse_map,
        meta=meta,
    )


def build_network_and_agents(loaded: LoadedData) -> Tuple[POIEnvironment, Dict[str, ResidentAgent], MobilitySimulator, Evaluator]:
    env = POIEnvironment(
        poi_catalog=loaded.poi_catalog,
        pois_by_category=loaded.pois_by_category,
        coarse_category_map=loaded.coarse_category_map,
    )
    agents: Dict[str, ResidentAgent] = {}
    for uid, dmap in loaded.trajectories_by_user.items():
        agents[uid] = ResidentAgent(user_id=uid, observed_2021=dmap)

    base_params = SimulatorParameters(
        alpha_dist=3.0,
        alpha_pref=3.0,
        alpha_anchor=1.5,
        routine_strength=0.7,
        exploration_rate=0.2,
        travel_time_scale=10.0,
        return_home_bias=1.5,
        social_influence_strength=0.5,
        global_mobility_noise_sigma=0.5,
        time_of_day_category_profile={},
        dwell_time_mean_by_category={},
    )
    simulator = MobilitySimulator(env=env, agents=agents, base_parameters=base_params, seed=GLOBAL_SEED)
    evaluator = Evaluator(env=env, coarse_map=loaded.coarse_category_map)
    return env, agents, simulator, evaluator


def holdout_split(loaded: LoadedData) -> SplitData:
    train_by_user: Dict[str, Dict[date, Trajectory]] = {}
    val_by_user: Dict[str, Dict[date, Trajectory]] = {}

    excluded_missing_context = 0
    total_val_candidates = 0

    for uid, dmap in loaded.trajectories_by_user.items():
        dates = sorted(dmap.keys())
        if len(dates) == 0:
            continue
        n_train = max(1, int(math.floor(0.8 * len(dates))))
        train_dates = dates[:n_train]
        val_dates = dates[n_train:]

        train_by_user[uid] = {d: dmap[d] for d in train_dates}

        # Fix: Only require 7 prior observed days, not necessarily consecutive calendar days
        kept_val: Dict[date, Trajectory] = {}
        for d in val_dates:
            total_val_candidates += 1
            # Count observed days strictly before target date d (not consecutive calendar days)
            prior_observed = [obs_d for obs_d in dates if obs_d < d and is_2021(obs_d)]
            if len(prior_observed) >= 7:
                kept_val[d] = dmap[d]
            else:
                excluded_missing_context += 1
        if kept_val:
            val_by_user[uid] = kept_val

    require(len(train_by_user) > 0, "Holdout split produced empty training set")
    require(sum(len(v) for v in train_by_user.values()) > 0, "Holdout split produced no training days")

    meta = {
        "split_method": "temporal_holdout_per_user",
        "train_fraction_per_user": 0.8,
        "validation_fraction_per_user": 0.2,
        "validation_requires_7_prior_observed_days": True,  # Not necessarily consecutive
        "total_validation_candidates": total_val_candidates,
        "excluded_validation_days_insufficient_prior_observed": excluded_missing_context,
        "num_users_with_validation_days": len(val_by_user),
        "num_train_user_days": sum(len(v) for v in train_by_user.values()),
        "num_val_user_days": sum(len(v) for v in val_by_user.values()),
    }
    log_info(
        f"Holdout split: train_days={meta['num_train_user_days']} val_days={meta['num_val_user_days']} "
        f"excluded_val_missing_context={excluded_missing_context}"
    )
    return SplitData(train_by_user=train_by_user, validation_by_user=val_by_user, meta=meta)


def save_results(
    output_dir: str,
    calibrated_parameters_payload: Dict[str, Any],
    calibration_log: List[Dict[str, Any]],
    evaluation_payload: Dict[str, Any],
    simulated_trajectories_payload: Dict[str, Any],
) -> None:
    require(isinstance(output_dir, str) and output_dir.strip(), "output_dir must be a non-empty string")
    abs_out = os.path.abspath(output_dir)
    os.makedirs(abs_out, exist_ok=True)

    paths = {
        "calibrated_parameters.json": os.path.join(abs_out, "calibrated_parameters.json"),
        "calibration_log.json": os.path.join(abs_out, "calibration_log.json"),
        "evaluation_results_on_validation.json": os.path.join(abs_out, "evaluation_results_on_validation.json"),
        "simulated_trajectories_validation.json": os.path.join(abs_out, "simulated_trajectories_validation.json"),
    }

    with open(paths["calibrated_parameters.json"], "w", encoding="utf-8") as f:
        json.dump(calibrated_parameters_payload, f, indent=2, sort_keys=True)
    with open(paths["calibration_log.json"], "w", encoding="utf-8") as f:
        json.dump(calibration_log, f, indent=2, sort_keys=True)
    with open(paths["evaluation_results_on_validation.json"], "w", encoding="utf-8") as f:
        json.dump(evaluation_payload, f, indent=2, sort_keys=True)
    with open(paths["simulated_trajectories_validation.json"], "w", encoding="utf-8") as f:
        json.dump(simulated_trajectories_payload, f, indent=2, sort_keys=True)

    print(f"[RESULT] wrote outputs to: {abs_out}")


# -----------------------
# Orchestrator
# -----------------------
def main(argv: Optional[Sequence[str]] = None) -> None:
    try:
        random.seed(GLOBAL_SEED)
        RNG.seed(GLOBAL_SEED)

        args = parse_cli(argv)
        output_dir = args.output_dir
        require(output_dir is not None, "--output_dir is required")

        loaded = load_data()
        env, agents, simulator, evaluator = build_network_and_agents(loaded)
        split = holdout_split(loaded)

        simulator.parameters.time_of_day_category_profile = _compute_global_time_profile(split.train_by_user)
        simulator.parameters.dwell_time_mean_by_category = _compute_dwell_means_by_category(split.train_by_user)

        objective_weights = {
            "category_share_mae": 1.0,
            "stop_count_abs_mean_error": 0.8,
            "stop_count_kl": 0.5,
            "tod_jsd_avg": 1.0,
            "topk_poi_recall": 1.0,
            "transition_divergence": 0.7,
            "trip_distance_wasserstein": 0.7,
        }

        calibrator: BaseCalibrator = RandomSearchCalibrator(
            n_iter=25,
            calibration_day_budget=250,
            seed=GLOBAL_SEED + 7,
        )

        log_info("Starting calibration on training split (2021-only)")
        best_params, best_obj, calibration_log = calibrator.fit(
            simulator=simulator,
            agents=agents,
            train_by_user=split.train_by_user,
            evaluator=evaluator,
            objective_weights=objective_weights,
        )

        calibrated_parameters_payload = {
            "best_parameters": {
                "alpha_dist": best_params.alpha_dist,
                "alpha_pref": best_params.alpha_pref,
                "alpha_anchor": best_params.alpha_anchor,
                "routine_strength": best_params.routine_strength,
                "exploration_rate": best_params.exploration_rate,
                # Note: time_of_day_category_profile and dwell_time_params_by_category are omitted
                # to reduce output size (~285KB). They are derived statistics from training data
                # and can be recomputed if needed using _compute_global_time_profile() and
                # _compute_dwell_means_by_category().
                "travel_time_scale": best_params.travel_time_scale,
                "return_home_hazard_params": {"bias": best_params.return_home_bias},
                "social_influence_strength": best_params.social_influence_strength,
                "global_mobility_noise_sigma": best_params.global_mobility_noise_sigma,
            },
            "best_objective": float(best_obj),
            "objective_definition": "Weighted sum of validation-style metric discrepancies on training subsample; recall is inverted as (1-recall).",
            "seed": int(GLOBAL_SEED),
            "meta": {
                "data_meta": loaded.meta,
                "split_meta": split.meta,
                "calibrator": {"name": "RandomSearchCalibrator", "n_iter": 25, "calibration_day_budget": 250},
                "note": "time_of_day_category_profile and dwell_time_params_by_category omitted from output for size reduction",
            },
        }

        require(len(split.validation_by_user) > 0, "No eligible validation days after strict 7-day context filtering")
        log_info("Rolling out simulation on validation split")
        simulator.set_parameters(best_params)

        for uid, agent in agents.items():
            if agent.pattern is None and uid in split.train_by_user and len(split.train_by_user[uid]) > 0:
                agent.fit_pattern_from_training(split.train_by_user[uid], env)

        serialized_by_user, sim_trajs = simulator.rollout(split.validation_by_user)

        gt_trajs: List[Trajectory] = []
        for dmap in split.validation_by_user.values():
            gt_trajs.extend(list(dmap.values()))

        metrics, objective, eval_meta = evaluator.compute_metrics(sim_trajs, gt_trajs, objective_weights)

        evaluation_payload = {
            "simulation_metrics": {
                "category_share_mae": metrics["category_share_mae"],
                "stop_count_abs_mean_error": metrics["stop_count_abs_mean_error"],
                "stop_count_kl": metrics["stop_count_kl"],
                "tod_jsd_avg": metrics["tod_jsd_avg"],
                "topk_poi_recall": metrics["topk_poi_recall"],
                "transition_divergence": metrics["transition_divergence"],
                "trip_distance_wasserstein": metrics["trip_distance_wasserstein"],
            },
            "objective": float(objective),
            "objective_weights": objective_weights,
            "validation_set": {
                "num_users": len(split.validation_by_user),
                "num_user_days": sum(len(v) for v in split.validation_by_user.values()),
                "requires_full_7day_context": True,
                "year_filter": ["2021-01-01", "2021-12-31"],
            },
            "meta": {
                "seed": int(GLOBAL_SEED),
                "evaluator_meta": eval_meta,
                "note": "All parsing/splitting/training/calibration/validation are strictly 2021-only after initial filter.",
            },
        }

        simulated_trajectories_payload = {
            "format_spec": "Activities at YYYY-MM-DD: POI#id at HH:MM:SS, POI#id at HH:MM:SS, ... .",
            "trajectories": serialized_by_user,
            "meta": {
                "seed": int(GLOBAL_SEED),
                "num_users": len(serialized_by_user),
                "num_user_days": sum(len(v) for v in serialized_by_user.values()),
            },
        }

        save_results(
            output_dir=output_dir,
            calibrated_parameters_payload=calibrated_parameters_payload,
            calibration_log=calibration_log,
            evaluation_payload=evaluation_payload,
            simulated_trajectories_payload=simulated_trajectories_payload,
        )
    except Exception as e:
        print(f"[INFO] Fatal error: {e}")
        raise


# Execute main for both direct execution and sandbox wrapper invocation
main()