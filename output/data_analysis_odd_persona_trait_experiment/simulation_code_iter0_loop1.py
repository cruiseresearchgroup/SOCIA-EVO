#!/usr/bin/env python
"""
simulate.py

End-to-end multi-agent simulator for LLM-based personas parameterised by
psychometric trait scores. The simulator:

- Loads synthetic persona data and Big Five adjective markers.
- Constructs persona prompts (BFI, NFC, CRT) according to the blueprint.
- Simulates administration of CRT2, bCRT, NFC-18, and BFI-brief tests under
  different administration granularities.
- Generates reflection-related writing tasks and simple LIWC-style features.
- Calibrates simulator parameters on a training split.
- Rolls out the simulator on a validation split.
- Evaluates trait fidelity, behavioural alignment, and response consistency.
- Saves an analysis-ready dataset (CSV + JSON) and evaluation metrics (JSON).

Notes and current limitations:

- Structured psychometric items (BFI, NFC, CRT2, bCRT) are simulated locally
  using a stochastic response generator. The `admin_mode` setting therefore
  controls only *conceptual* API call budgeting and the `num_api_calls`
  accounting; OpenAI LLM calls are only used for reflection-related writing
  tasks via the ReasoningAgent.
- The interaction network is constructed over validation personas and is used
  to propagate simple scalar signals (CRT2 and NFC levels) that then influence
  persona traits prior to simulation, providing a basic multi-agent interaction
  mechanism.
- Response consistency evaluation currently covers only internal reliability
  (Cronbach's alpha) for BFI and NFC. Test-retest consistency and within-session
  order-shuffle consistency are documented but deferred for future extensions.
- Behavioural alignment evaluation focuses on trait–writing-feature
  associations via correlations and simple regressions; explicit comparisons of
  writing distributions across different BFI persona prompt modes
  (coarse_numeric, coarse_descriptive, granular_serapio) are not yet included.
"""

import argparse
import json
import logging
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore

# ---------------------------------------------------------------------------
# Path handling as specified
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.environ.get("PROJECT_ROOT")
DATA_PATH = os.environ.get("DATA_PATH")

# Provide robust defaults if environment variables are not set
if PROJECT_ROOT is None:
    PROJECT_ROOT = os.getcwd()
else:
    PROJECT_ROOT = os.path.abspath(PROJECT_ROOT)
if DATA_PATH is None:
    DATA_PATH = "data"

DATA_DIR = os.path.join(PROJECT_ROOT, DATA_PATH)
DATA_DIR = os.path.abspath(DATA_DIR)

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


# ---------------------------------------------------------------------------
# OpenAI helpers
# ---------------------------------------------------------------------------


def get_openai_api_key() -> str:
    """
    Retrieve the OpenAI API key from the environment.

    Returns
    -------
    str
        The API key.

    Raises
    ------
    ValueError
        If the key is not found.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        return api_key
    raise ValueError("OpenAI API key not found in environment")


def call_gpt5_with_responses_api(
    prompt: str, model: str = "gpt-5", max_output_tokens: int = 4000
) -> str:
    """
    Call the OpenAI Responses API with the given prompt and return text output.

    Parameters
    ----------
    prompt : str
        Prompt string to send to the model.
    model : str, default "gpt-5"
        Model name.
    max_output_tokens : int, default 4000
        Maximum number of output tokens.

    Returns
    -------
    str
        Extracted text response.

    Raises
    ------
    RuntimeError
        If the OpenAI client library is not installed.
    Exception
        Any exception raised during API call is propagated.
    """
    if OpenAI is None:
        raise RuntimeError(
            "The 'openai' package is not installed; cannot call GPT-5 Responses API."
        )

    api_key = get_openai_api_key()
    client = OpenAI(api_key=api_key)

    responses_kwargs = {
        "model": model,
        "input": [
            {"role": "user", "content": [{"type": "input_text", "text": prompt}]}
        ],
        "max_output_tokens": max_output_tokens,
    }

    resp = client.responses.create(**responses_kwargs)

    def extract_response(resp_obj: Any) -> str:
        if hasattr(resp_obj, "output_text") and isinstance(
            getattr(resp_obj, "output_text"), str
        ):
            return getattr(resp_obj, "output_text")
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


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def set_global_seed(seed: int) -> None:
    """
    Set global random seed for deterministic behaviour.

    Parameters
    ----------
    seed : int
        The random seed to use. Must be a non-negative integer.
    """
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("Seed must be a non-negative integer.")
    np.random.seed(seed)
    random.seed(seed)


def ensure_data_dir() -> None:
    """
    Ensure that the DATA_DIR exists, creating it if necessary.

    Relative and absolute paths are both allowed.
    """
    os.makedirs(DATA_DIR, exist_ok=True)


def fit_linear(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """
    Fit a simple linear model y = a * x + b using least squares.

    Parameters
    ----------
    x : np.ndarray
        Predictor values.
    y : np.ndarray
        Response values.

    Returns
    -------
    (a, b) : tuple of float
        Estimated slope and intercept.
    """
    if x.size != y.size:
        raise ValueError("x and y must have the same number of elements.")
    if x.size == 0:
        raise ValueError("Cannot fit linear model on empty data.")
    A = np.vstack([x, np.ones_like(x)]).T
    solution, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    a, b = solution
    return float(a), float(b)


def pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute Pearson correlation coefficient between two 1D arrays.

    Parameters
    ----------
    x : np.ndarray
        First vector.
    y : np.ndarray
        Second vector.

    Returns
    -------
    float
        Pearson correlation coefficient, or np.nan if undefined.
    """
    if x.size != y.size or x.size < 2:
        return float("nan")
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x_mean = x.mean()
    y_mean = y.mean()
    num = np.sum((x - x_mean) * (y - y_mean))
    den = np.sqrt(np.sum((x - x_mean) ** 2) * np.sum((y - y_mean) ** 2))
    if den == 0:
        return float("nan")
    return float(num / den)


def _rankdata(a: np.ndarray) -> np.ndarray:
    """
    Compute ranks of the data, handling ties by assigning average ranks.

    Parameters
    ----------
    a : np.ndarray
        Input array.

    Returns
    -------
    np.ndarray
        Array of ranks (float), same shape as `a`.
    """
    a = np.asarray(a, dtype=float)
    sorter = np.argsort(a)
    inv = np.empty_like(sorter)
    inv[sorter] = np.arange(len(a))
    a_sorted = a[sorter]
    obs = np.concatenate(([True], a_sorted[1:] != a_sorted[:-1]))
    dense_rank = np.cumsum(obs) - 1
    counts = np.bincount(dense_rank)
    cumulative = np.cumsum(counts)
    start = np.concatenate(([0], cumulative[:-1]))
    ranks = (start + cumulative - 1) / 2.0
    return ranks[dense_rank][inv]


def spearmanr(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute Spearman rank correlation coefficient between two 1D arrays.

    Parameters
    ----------
    x : np.ndarray
        First vector.
    y : np.ndarray
        Second vector.

    Returns
    -------
    float
        Spearman correlation coefficient, or np.nan if undefined.
    """
    if x.size != y.size or x.size < 2:
        return float("nan")
    rx = _rankdata(x)
    ry = _rankdata(y)
    return pearsonr(rx, ry)


def cronbach_alpha(item_scores: np.ndarray) -> float:
    """
    Compute Cronbach's alpha for a 2D item score matrix.

    Parameters
    ----------
    item_scores : np.ndarray
        Array of shape (n_observations, n_items) with item responses.

    Returns
    -------
    float
        Cronbach's alpha, or np.nan if undefined.
    """
    if item_scores.ndim != 2:
        raise ValueError("item_scores must be a 2D array.")
    n_obs, n_items = item_scores.shape
    if n_items < 2 or n_obs < 2:
        return float("nan")
    item_var = item_scores.var(axis=0, ddof=1).sum()
    total_var = item_scores.sum(axis=1).var(ddof=1)
    if total_var <= 0:
        return float("nan")
    alpha = n_items / (n_items - 1) * (1.0 - item_var / total_var)
    return float(alpha)


def simple_linear_regression_metrics(
    x: np.ndarray, y: np.ndarray
) -> Tuple[float, float, float]:
    """
    Fit y = a * x + b and compute R^2.

    Parameters
    ----------
    x : np.ndarray
        Predictor.
    y : np.ndarray
        Response.

    Returns
    -------
    (a, b, r2) : tuple of float
        Slope, intercept, and coefficient of determination.
    """
    if x.size != y.size or x.size < 2:
        return float("nan"), float("nan"), float("nan")
    a, b = fit_linear(x, y)
    y_pred = a * x + b
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return a, b, r2


# ---------------------------------------------------------------------------
# CLI configuration
# ---------------------------------------------------------------------------


@dataclass
class SimulationConfig:
    """
    Configuration options for the simulator.

    Attributes
    ----------
    admin_mode : str
        Administration mode: 'per_item', 'per_test', or 'all_tests'. For
        structured psychometric items, this affects only conceptual API
        budgeting and the `num_api_calls` counter; responses are simulated
        locally. LLM calls are used only for reflection-related writing tasks.
    shuffle_items_within_tests : bool
        Whether to shuffle items within each test.
    shuffle_test_order : bool
        Whether to shuffle the order of tests per persona.
    seed : int
        Global random seed.
    holdout_fraction : float
        Fraction of personas reserved for validation.
    crt_numeric_only : bool
        Whether CRT prompts use numeric-only level specification.
    bfi_prompt_mode : str
        BFI prompt mode: 'coarse_numeric', 'coarse_descriptive', or
        'granular_serapio'.
    """

    admin_mode: str = "per_test"
    shuffle_items_within_tests: bool = False
    shuffle_test_order: bool = False
    seed: int = 42
    holdout_fraction: float = 0.2
    crt_numeric_only: bool = False
    bfi_prompt_mode: str = "granular_serapio"


def parse_cli() -> SimulationConfig:
    """
    Parse command-line arguments into a SimulationConfig.

    Returns
    -------
    SimulationConfig
        Parsed configuration object.

    Raises
    ------
    ValueError
        If provided arguments are invalid.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Simulate LLM-based personas taking psychometric tests and "
            "performing writing tasks."
        )
    )
    parser.add_argument(
        "--admin_mode",
        type=str,
        choices=["per_item", "per_test", "all_tests"],
        default="per_test",
        help=(
            "Granularity of conceptual API calls for psychometric tests: "
            "per_item (one call per item), per_test (one call per test), or "
            "all_tests (one call per persona with all tests). Structured test "
            "items are simulated locally; this setting affects only "
            "num_api_calls accounting and not real OpenAI requests for those "
            "items. Reflection-related writing tasks always use real LLM calls."
        ),
    )
    parser.add_argument(
        "--shuffle_items",
        action="store_true",
        help="Shuffle items within each test.",
    )
    parser.add_argument(
        "--shuffle_tests",
        action="store_true",
        help="Shuffle order of tests for each persona.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global random seed (non-negative integer).",
    )
    parser.add_argument(
        "--holdout_fraction",
        type=float,
        default=0.2,
        help=(
            "Fraction of personas reserved for validation (0 < f < 1). "
            "The remainder is used for calibration."
        ),
    )
    parser.add_argument(
        "--crt_numeric_only",
        action="store_true",
        help=(
            "Use numeric-only CRT level specification instead of full "
            "descriptive sentences."
        ),
    )
    parser.add_argument(
        "--bfi_prompt_mode",
        type=str,
        choices=["coarse_numeric", "coarse_descriptive", "granular_serapio"],
        default="granular_serapio",
        help="BFI persona prompt mode.",
    )

    args = parser.parse_args()

    if args.seed < 0:
        raise ValueError("Seed must be a non-negative integer.")
    if not (0.0 < args.holdout_fraction < 1.0):
        raise ValueError("holdout_fraction must be between 0 and 1 (exclusive).")

    config = SimulationConfig(
        admin_mode=args.admin_mode,
        shuffle_items_within_tests=args.shuffle_items,
        shuffle_test_order=args.shuffle_tests,
        seed=args.seed,
        holdout_fraction=args.holdout_fraction,
        crt_numeric_only=args.crt_numeric_only,
        bfi_prompt_mode=args.bfi_prompt_mode,
    )
    return config


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _generate_synthetic_personas(path: str) -> pd.DataFrame:
    """
    Generate a small synthetic personas dataset and save it to CSV.

    This is used as a fallback when no real synthetic_personas.csv is found.

    Parameters
    ----------
    path : str
        Absolute path where the CSV should be saved.

    Returns
    -------
    pandas.DataFrame
        Generated personas dataframe.
    """
    logging.warning(
        "synthetic_personas.csv not found at %s. "
        "Generating a small synthetic example dataset.",
        path,
    )
    n = 40
    rng = np.random.RandomState(123)
    persona_ids = [f"P{i:03d}" for i in range(n)]
    ages = rng.randint(18, 70, size=n)
    genders = rng.choice(["male", "female", "non-binary"], size=n)

    def rand_trait():
        return rng.choice(np.arange(1.0, 5.5, 0.5), size=n)

    bfi_extraversion = rand_trait()
    bfi_agreeableness = rand_trait()
    bfi_conscientiousness = rand_trait()
    bfi_neuroticism = rand_trait()
    bfi_openness = rand_trait()
    nfc_score = rand_trait()
    crt2_score = rng.randint(0, 5, size=n)

    df = pd.DataFrame(
        {
            "persona_id": persona_ids,
            "age": ages,
            "gender": genders,
            "bfi_extraversion": bfi_extraversion,
            "bfi_agreeableness": bfi_agreeableness,
            "bfi_conscientiousness": bfi_conscientiousness,
            "bfi_neuroticism": bfi_neuroticism,
            "bfi_openness": bfi_openness,
            "nfc_score": nfc_score,
            "crt2_score": crt2_score,
        }
    )
    df.to_csv(path, index=False)
    return df


def _generate_serapio_markers(path: str) -> pd.DataFrame:
    """
    Generate a small Serapio/Goldberg markers dataset and save it to CSV.

    Used as a fallback when serapio_goldberg_markers.csv is not found.

    Parameters
    ----------
    path : str
        Absolute path where the CSV should be saved.

    Returns
    -------
    pandas.DataFrame
        Generated markers dataframe.
    """
    logging.warning(
        "serapio_goldberg_markers.csv not found at %s. "
        "Generating a small synthetic markers dataset.",
        path,
    )
    data = [
        {
            "trait": "extraversion",
            "low_markers": "quiet|reserved|introverted",
            "high_markers": "outgoing|sociable|energetic",
        },
        {
            "trait": "agreeableness",
            "low_markers": "critical|quarrelsome|rude",
            "high_markers": "kind|sympathetic|cooperative",
        },
        {
            "trait": "conscientiousness",
            "low_markers": "disorganized|careless|impulsive",
            "high_markers": "organized|dependable|thorough",
        },
        {
            "trait": "neuroticism",
            "low_markers": "calm|emotionally_stable|relaxed",
            "high_markers": "anxious|easily_upset|moody",
        },
        {
            "trait": "openness",
            "low_markers": "conventional|uncreative|narrow_interests",
            "high_markers": "imaginative|curious|artistic",
        },
    ]
    df = pd.DataFrame(data)
    df.to_csv(path, index=False)
    return df


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load required data files: synthetic_personas.csv and Serapio/Goldberg markers.

    If the files are missing, small synthetic datasets are generated and saved
    to the expected locations so the pipeline can run end-to-end.

    Returns
    -------
    (personas_df, markers_df) : tuple of pandas.DataFrame
        Loaded personas and markers dataframes.
    """
    ensure_data_dir()
    personas_path = os.path.join(DATA_DIR, "synthetic_personas.csv")
    markers_path = os.path.join(DATA_DIR, "serapio_goldberg_markers.csv")

    if os.path.exists(personas_path):
        personas_df = pd.read_csv(personas_path)
        logging.info("Loaded personas from %s (n=%d).", personas_path, len(personas_df))
    else:
        personas_df = _generate_synthetic_personas(personas_path)
        logging.info(
            "Generated synthetic personas at %s (n=%d).",
            personas_path,
            len(personas_df),
        )

    if os.path.exists(markers_path):
        markers_df = pd.read_csv(markers_path)
        logging.info(
            "Loaded Serapio/Goldberg markers from %s (n=%d).",
            markers_path,
            len(markers_df),
        )
    else:
        markers_df = _generate_serapio_markers(markers_path)
        logging.info(
            "Generated synthetic markers at %s (n=%d).",
            markers_path,
            len(markers_df),
        )

    return personas_df, markers_df


# ---------------------------------------------------------------------------
# Persona and prompt construction
# ---------------------------------------------------------------------------


QUALIFIER_SCALE_1_TO_9 = [
    "extremely {low_adjective}",
    "very {low_adjective}",
    "{low_adjective}",
    "a bit {low_adjective}",
    "neither {low_adjective} nor {high_adjective}",
    "a bit {high_adjective}",
    "{high_adjective}",
    "very {high_adjective}",
    "extremely {high_adjective}",
]


CRT_LEVEL_DESCRIPTIONS = [
    {
        "level": 0,
        "label": "very low reflection",
        "sentence": (
            "I almost always trust my first impression, answer quickly "
            "without re-checking, and rarely notice when a question "
            "might be tricky."
        ),
    },
    {
        "level": 1,
        "label": "low reflection",
        "sentence": (
            "I often go with my first impression and only occasionally "
            "stop to reconsider whether it might be misleading."
        ),
    },
    {
        "level": 2,
        "label": "mixed reflection",
        "sentence": (
            "I sometimes pause to reconsider my first impression before "
            "answering, but I am inconsistent and often stick with the "
            "obvious answer."
        ),
    },
    {
        "level": 3,
        "label": "high reflection",
        "sentence": (
            "I usually pause to check whether an obvious answer could be "
            "a trap, and I am willing to change my mind after thinking "
            "things through."
        ),
    },
    {
        "level": 4,
        "label": "very high reflection",
        "sentence": (
            "I almost always look for hidden assumptions, carefully check "
            "for tricks, and verify my answers with calculations before "
            "responding."
        ),
    },
]


NFC_HIGH_DESCRIPTORS = [
    "enjoys complex rather than simple problems",
    "likes responsibility for situations that require a lot of thinking",
    "finds satisfaction in deliberating hard and for long periods",
    "enjoys tasks that involve generating new solutions to problems",
    "prefers life to be filled with puzzles and challenging questions",
    "finds abstract thinking appealing",
    "prefers intellectual, difficult, and important tasks",
    "often deliberates about issues even when they do not affect them personally",
]

NFC_LOW_DESCRIPTORS = [
    "thinks only as hard as necessary",
    "prefers tasks that require little thought once learned",
    "would rather do something that requires little thought",
    "tries to avoid situations that demand deep thinking",
    "finds thinking effortful rather than fun",
    "feels relief rather than satisfaction after heavy mental effort",
    "is content when something works without understanding how or why",
]


def _build_markers_dict(markers_df: pd.DataFrame) -> Dict[str, Dict[str, List[str]]]:
    """
    Build a dictionary mapping trait names to low/high adjective lists.

    Parameters
    ----------
    markers_df : pandas.DataFrame
        Dataframe with columns ['trait', 'low_markers', 'high_markers'].

    Returns
    -------
    dict
        Mapping {trait_name: {"low": [adj...], "high": [adj...]}}.

    Raises
    ------
    ValueError
        If required columns are missing.
    """
    required_cols = {"trait", "low_markers", "high_markers"}
    missing = required_cols - set(markers_df.columns)
    if missing:
        raise ValueError(
            f"Markers dataframe is missing required columns: {sorted(missing)}"
        )
    markers: Dict[str, Dict[str, List[str]]] = {}
    for _, row in markers_df.iterrows():
        trait = str(row["trait"]).strip().lower()
        low = [
            s.strip().replace("_", " ")
            for s in str(row["low_markers"]).split("|")
            if s
        ]
        high = [
            s.strip().replace("_", " ")
            for s in str(row["high_markers"]).split("|")
            if s
        ]
        if not low or not high:
            continue
        markers[trait] = {"low": low, "high": high}
    if not markers:
        raise ValueError("No valid markers could be constructed from markers_df.")
    return markers


@dataclass
class Persona:
    """
    Representation of a single synthetic persona and associated prompts.

    Attributes
    ----------
    persona_id : str
        Unique persona identifier.
    demographics : dict
        Demographic attributes (e.g., age, gender).
    traits : dict
        Trait scores (BFI domains, NFC, CRT2).
    markers : dict
        Marker dictionary for BFI adjectives.
    """

    persona_id: str
    demographics: Dict[str, Any]
    traits: Dict[str, float]
    markers: Dict[str, Dict[str, List[str]]]

    @classmethod
    def from_series(
        cls, row: pd.Series, markers: Dict[str, Dict[str, List[str]]]
    ) -> "Persona":
        """
        Construct a Persona instance from a pandas Series row.

        Parameters
        ----------
        row : pandas.Series
            Row from the personas dataframe.
        markers : dict
            BFI markers dictionary from `_build_markers_dict`.

        Returns
        -------
        Persona
            Instantiated persona.

        Raises
        ------
        ValueError
            If required columns are missing or invalid.
        """
        if "persona_id" not in row:
            raise ValueError("Personas dataframe must include 'persona_id' column.")
        persona_id = str(row["persona_id"])

        demographics = {
            k: row[k]
            for k in row.index
            if k
            not in [
                "persona_id",
                "nfc_score",
                "crt2_score",
            ]
            and not k.startswith("bfi_")
        }

        traits: Dict[str, float] = {}
        for col in row.index:
            if col.startswith("bfi_"):
                traits[col] = float(row[col])
        if "nfc_score" in row:
            traits["nfc_score"] = float(row["nfc_score"])
        if "crt2_score" in row:
            traits["crt2_score"] = float(row["crt2_score"])

        for k, v in traits.items():
            if "bfi_" in k or k == "nfc_score":
                if not (1.0 <= v <= 5.0):
                    raise ValueError(
                        f"Trait {k} for persona {persona_id} must be in [1, 5]. "
                        f"Got {v}."
                    )
            if k == "crt2_score":
                if not (0.0 <= v <= 4.0):
                    raise ValueError(
                        f"CRT2 score for persona {persona_id} must be in [0, 4]. "
                        f"Got {v}."
                    )

        return cls(
            persona_id=persona_id,
            demographics=demographics,
            traits=traits,
            markers=markers,
        )

    @staticmethod
    def _bfi_numeric_bin(score: float) -> str:
        """
        Map a continuous BFI score (1-5) into 'low', 'neutral', 'high' bins.

        Parameters
        ----------
        score : float
            BFI score.

        Returns
        -------
        str
            One of 'low', 'neutral', 'high'.
        """
        if score <= 2.5:
            return "low"
        if score >= 3.5:
            return "high"
        return "neutral"

    @staticmethod
    def _trait_to_9_level(score: float) -> int:
        """
        Map a BFI or NFC score (1-5 with .5 increments) to a 1-9 qualifier level.

        Uses the linear rule: level = 2 * (score - 1) + 1.

        Parameters
        ----------
        score : float
            Score in [1, 5].

        Returns
        -------
        int
            Level in {1, ..., 9}.
        """
        level = int(round(2 * (score - 1) + 1))
        return max(1, min(9, level))

    def build_bfi_granular_prompt(self) -> str:
        """
        Construct a granular Serapio-style BFI prompt using 9-level qualifiers.

        Returns
        -------
        str
            Combined BFI persona description.
        """
        parts: List[str] = []
        for domain in [
            "extraversion",
            "agreeableness",
            "conscientiousness",
            "neuroticism",
            "openness",
        ]:
            col = f"bfi_{domain}"
            if col not in self.traits:
                continue
            score = self.traits[col]
            level = self._trait_to_9_level(score)
            markers = self.markers.get(domain)
            if not markers:
                continue
            if level <= 4:
                adjective = random.choice(markers["low"])
                qualifier_tpl = QUALIFIER_SCALE_1_TO_9[level - 1]
                qualified = qualifier_tpl.format(
                    low_adjective=adjective,
                    high_adjective=random.choice(markers["high"]),
                )
            elif level >= 6:
                adjective = random.choice(markers["high"])
                qualifier_tpl = QUALIFIER_SCALE_1_TO_9[level - 1]
                qualified = qualifier_tpl.format(
                    low_adjective=random.choice(markers["low"]),
                    high_adjective=adjective,
                )
            else:
                low_adj = random.choice(markers["low"])
                high_adj = random.choice(markers["high"])
                qualifier_tpl = QUALIFIER_SCALE_1_TO_9[level - 1]
                qualified = qualifier_tpl.format(
                    low_adjective=low_adj, high_adjective=high_adj
                )

            parts.append(f"My {domain} is best described as {qualified}.")

        if not parts:
            return ""
        descriptors = " ".join(parts)
        return (
            'For the following tasks, respond as a person described as: '
            f'"I am {descriptors}"'
        )

    def build_bfi_coarse_numeric_prompt(self) -> str:
        """
        Construct a coarse-numeric BFI prompt using low/neutral/high bins.

        Returns
        -------
        str
            Coarse numeric BFI persona description.
        """
        sentences: List[str] = []
        for domain in [
            "extraversion",
            "agreeableness",
            "conscientiousness",
            "neuroticism",
            "openness",
        ]:
            col = f"bfi_{domain}"
            if col not in self.traits:
                continue
            score = self.traits[col]
            polarity = self._bfi_numeric_bin(score)
            sentences.append(f"You are a person with {polarity} {domain}.")
        return " ".join(sentences)

    def build_bfi_coarse_descriptive_prompt(self) -> str:
        """
        Construct a coarse-descriptive BFI prompt using markers and bins.

        Returns
        -------
        str
            Coarse descriptive BFI persona description.
        """
        sentences: List[str] = []
        for domain in [
            "extraversion",
            "agreeableness",
            "conscientiousness",
            "neuroticism",
            "openness",
        ]:
            col = f"bfi_{domain}"
            if col not in self.traits:
                continue
            score = self.traits[col]
            bin_label = self._bfi_numeric_bin(score)
            markers = self.markers.get(domain)
            if not markers:
                continue
            phrase = ""
            if bin_label == "low":
                adjs = random.sample(
                    markers["low"], k=min(3, len(markers["low"]))
                )
                phrase = ", ".join(adjs)
            elif bin_label == "high":
                adjs = random.sample(
                    markers["high"], k=min(3, len(markers["high"]))
                )
                phrase = ", ".join(adjs)
            else:
                low_adj = random.choice(markers["low"])
                high_adj = random.choice(markers["high"])
                phrase = f"neither particularly {low_adj} nor {high_adj}"
            sentences.append(
                f"You are a person who is {phrase} in terms of {domain}."
            )
        return " ".join(sentences)

    def build_nfc_prompt(self) -> str:
        """
        Construct a Need for Cognition (NFC) prompt using 9-level qualifiers.

        Returns
        -------
        str
            NFC persona description prompt. Empty string if NFC score is missing.
        """
        if "nfc_score" not in self.traits:
            return ""
        score = self.traits["nfc_score"]
        level = self._trait_to_9_level(score)

        rng = np.random.RandomState(abs(hash(self.persona_id)) % (2**32))
        if level <= 4:
            k = rng.randint(1, 4)
            phrases = rng.choice(NFC_LOW_DESCRIPTORS, size=k, replace=False)
            qualifier_tpl = QUALIFIER_SCALE_1_TO_9[level - 1]
            description = ", ".join(phrases)
            qualified = qualifier_tpl.format(
                low_adjective=description, high_adjective="intellectually curious"
            )
            text = f"I {qualified}."
        elif level >= 6:
            k = rng.randint(1, 4)
            phrases = rng.choice(NFC_HIGH_DESCRIPTORS, size=k, replace=False)
            qualifier_tpl = QUALIFIER_SCALE_1_TO_9[level - 1]
            description = ", ".join(phrases)
            qualified = qualifier_tpl.format(
                low_adjective="uninterested in thinking", high_adjective=description
            )
            text = f"I {qualified}."
        else:
            k_low = rng.randint(1, 3)
            k_high = rng.randint(1, 3)
            low_phrases = rng.choice(
                NFC_LOW_DESCRIPTORS, size=k_low, replace=False
            )
            high_phrases = rng.choice(
                NFC_HIGH_DESCRIPTORS, size=k_high, replace=False
            )
            description = (
                "neither especially inclined to think deeply nor strongly avoidant "
                "of thinking; sometimes "
                + ", ".join(high_phrases)
                + ", but other times "
                + ", ".join(low_phrases)
            )
            text = f"I am {description}."

        return (
            'For the following tasks, respond as a person described as: '
            f'"{text}"'
        )

    def build_crt_prompt(self, numeric_only: bool = False) -> str:
        """
        Construct a CRT-style reflective thinking prompt.

        Parameters
        ----------
        numeric_only : bool, default False
            If True, only a numeric CRT2 level statement is used.
            If False, use the full descriptive sentence without mentioning
            scores or correctness, describing thinking style only.

        Returns
        -------
        str
            CRT persona description prompt. Empty string if CRT2 score is missing.
        """
        if "crt2_score" not in self.traits:
            return ""
        level = int(round(self.traits["crt2_score"]))
        level = max(0, min(4, level))
        desc = next(
            (d for d in CRT_LEVEL_DESCRIPTIONS if d["level"] == level),
            CRT_LEVEL_DESCRIPTIONS[2],
        )
        if numeric_only:
            return f"This persona has a CRT2 ability level of {level} on a 0–4 scale."
        sentence = desc["sentence"]
        return (
            "For the following CRT-style questions, respond as a person described as: "
            f"\"{sentence}\""
        )

    def build_full_persona_prompt(
        self,
        crt_numeric_only: bool = False,
        bfi_mode: str = "granular_serapio",
    ) -> str:
        """
        Combine BFI, NFC, and CRT prompts into a single persona configuration.

        Parameters
        ----------
        crt_numeric_only : bool, default False
            Whether to use numeric-only CRT description.
        bfi_mode : str, default "granular_serapio"
            BFI prompt mode: 'coarse_numeric', 'coarse_descriptive',
            or 'granular_serapio'.

        Returns
        -------
        str
            Full persona prompt.
        """
        if bfi_mode == "coarse_numeric":
            bfi_part = self.build_bfi_coarse_numeric_prompt()
        elif bfi_mode == "coarse_descriptive":
            bfi_part = self.build_bfi_coarse_descriptive_prompt()
        else:
            bfi_part = self.build_bfi_granular_prompt()

        pieces = [
            bfi_part,
            self.build_nfc_prompt(),
            self.build_crt_prompt(numeric_only=crt_numeric_only),
        ]
        pieces = [p for p in pieces if p]
        return " ".join(pieces)


# ---------------------------------------------------------------------------
# Network and agents (simple multi-layer interaction structure)
# ---------------------------------------------------------------------------


@dataclass
class InteractionNetwork:
    """
    Simple multi-layer interaction network over personas.

    Attributes
    ----------
    agents : dict
        Mapping from persona_id to Persona.
    layers : dict
        Mapping from layer name to adjacency list:
        {layer_name: {persona_id: [neighbor_ids...]}}.

    Notes
    -----
    The current simulator uses this network to propagate simple scalar signals
    (e.g., CRT2 and NFC trait levels) across personas via averaging, and then
    blends these propagated values back into persona traits before simulating
    responses. This provides a basic form of multi-agent influence while
    keeping the integration surface ready for richer extensions.
    """

    agents: Dict[str, Persona]
    layers: Dict[str, Dict[str, List[str]]]

    @classmethod
    def from_personas(cls, personas: List[Persona]) -> "InteractionNetwork":
        """
        Construct a fully connected two-layer network over personas.

        The layers are:
        - 'social': symmetric connections between all personas.
        - 'information': same structure, conceptually separate.

        Parameters
        ----------
        personas : list of Persona
            Personas to include in the network.

        Returns
        -------
        InteractionNetwork
            Instantiated network.
        """
        ids = [p.persona_id for p in personas]
        agents = {p.persona_id: p for p in personas}
        layers: Dict[str, Dict[str, List[str]]] = {"social": {}, "information": {}}
        for layer in layers:
            for i in ids:
                neighbors = [j for j in ids if j != i]
                layers[layer][i] = neighbors
        return cls(agents=agents, layers=layers)

    def propagate_signal(
        self, layer: str, initial_values: Dict[str, float], n_steps: int = 1
    ) -> Dict[str, float]:
        """
        Propagate a scalar signal over the specified layer using averaging.

        Parameters
        ----------
        layer : str
            Name of the layer ('social' or 'information').
        initial_values : dict
            Mapping from persona_id to initial scalar value.
        n_steps : int, default 1
            Number of synchronous propagation steps.

        Returns
        -------
        dict
            Mapping from persona_id to propagated value after n_steps.

        Raises
        ------
        ValueError
            If the layer name is invalid.
        """
        if layer not in self.layers:
            raise ValueError(f"Unknown layer {layer!r}. Available: {list(self.layers)}")
        values = {
            pid: float(initial_values.get(pid, 0.0)) for pid in self.agents.keys()
        }
        for _ in range(max(0, n_steps)):
            new_values: Dict[str, float] = {}
            for pid, neighbors in self.layers[layer].items():
                neigh_vals = [values[nid] for nid in neighbors] or [values[pid]]
                new_values[pid] = float(
                    0.5 * values[pid] + 0.5 * (sum(neigh_vals) / len(neigh_vals))
                )
            values = new_values
        return values


def build_network_and_agents(
    personas_df: pd.DataFrame, markers_df: pd.DataFrame, config: SimulationConfig
) -> InteractionNetwork:
    """
    Build persona agents and a multi-layer interaction network.

    Parameters
    ----------
    personas_df : pandas.DataFrame
        Personas dataframe (typically the validation split).
    markers_df : pandas.DataFrame
        Loaded Serapio/Goldberg markers dataframe.
    config : SimulationConfig
        Simulation configuration (currently unused here but included for
        extensibility).

    Returns
    -------
    InteractionNetwork
        InteractionNetwork instance over the provided personas.
    """
    del config  # currently unused
    markers = _build_markers_dict(markers_df)
    personas: List[Persona] = []
    for _, row in personas_df.iterrows():
        personas.append(Persona.from_series(row, markers))
    network = InteractionNetwork.from_personas(personas)
    logging.info(
        "Constructed interaction network with %d personas and layers: %s.",
        len(personas),
        list(network.layers.keys()),
    )
    return network


# ---------------------------------------------------------------------------
# Holdout split
# ---------------------------------------------------------------------------


def holdout_split(
    personas_df: pd.DataFrame, config: SimulationConfig
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split personas dataframe into training and validation sets.

    Parameters
    ----------
    personas_df : pandas.DataFrame
        Full personas dataset.
    config : SimulationConfig
        Configuration including holdout_fraction and seed.

    Returns
    -------
    (train_df, valid_df) : tuple of pandas.DataFrame
        Training and validation splits.
    """
    n = len(personas_df)
    if n < 2:
        raise ValueError("Need at least 2 personas for a holdout split.")
    rng = np.random.RandomState(config.seed)
    indices = np.arange(n)
    rng.shuffle(indices)
    n_valid = int(round(config.holdout_fraction * n))
    n_valid = max(1, min(n - 1, n_valid))
    valid_idx = indices[:n_valid]
    train_idx = indices[n_valid:]
    train_df = personas_df.iloc[train_idx].reset_index(drop=True)
    valid_df = personas_df.iloc[valid_idx].reset_index(drop=True)
    logging.info(
        "Holdout split: %d training personas, %d validation personas.",
        len(train_df),
        len(valid_df),
    )
    return train_df, valid_df


# ---------------------------------------------------------------------------
# Psychometric tests definitions
# ---------------------------------------------------------------------------


@dataclass
class TestItem:
    """
    Representation of a single test item.

    Attributes
    ----------
    item_id : str
        Unique item identifier within the test.
    test_name : str
        Name of the test the item belongs to.
    text : str
        Item stimulus text.
    item_type : str
        Type of item: 'likert' or 'crt'.
    scale : str, optional
        Name of the underlying scale/domain (e.g., 'extraversion').
    reverse_scored : bool, default False
        Whether the item is reverse-scored.
    correct_answer : str, optional
        Correct answer for CRT-style items.
    incorrect_answer : str, optional
        Typical intuitive but wrong answer for CRT-style items.
    """

    item_id: str
    test_name: str
    text: str
    item_type: str
    scale: Optional[str] = None
    reverse_scored: bool = False
    correct_answer: Optional[str] = None
    incorrect_answer: Optional[str] = None


@dataclass
class Test:
    """
    Representation of a psychometric test.

    Attributes
    ----------
    name : str
        Name of the test.
    items : list of TestItem
        Items in the test.
    """

    name: str
    items: List[TestItem]


def build_bfi_brief_test() -> Test:
    """
    Construct a small BFI-brief style test with 10 items (2 per Big Five domain).

    Returns
    -------
    Test
        BFI-brief style test object.
    """
    items: List[TestItem] = [
        TestItem(
            item_id="BFI1",
            test_name="BFI_brief",
            text="I see myself as someone who is talkative.",
            item_type="likert",
            scale="extraversion",
            reverse_scored=False,
        ),
        TestItem(
            item_id="BFI2",
            test_name="BFI_brief",
            text="I see myself as someone who tends to be quiet.",
            item_type="likert",
            scale="extraversion",
            reverse_scored=True,
        ),
        TestItem(
            item_id="BFI3",
            test_name="BFI_brief",
            text="I see myself as someone who is considerate and kind to almost everyone.",
            item_type="likert",
            scale="agreeableness",
            reverse_scored=False,
        ),
        TestItem(
            item_id="BFI4",
            test_name="BFI_brief",
            text="I see myself as someone who tends to find fault with others.",
            item_type="likert",
            scale="agreeableness",
            reverse_scored=True,
        ),
        TestItem(
            item_id="BFI5",
            test_name="BFI_brief",
            text="I see myself as someone who does a thorough job.",
            item_type="likert",
            scale="conscientiousness",
            reverse_scored=False,
        ),
        TestItem(
            item_id="BFI6",
            test_name="BFI_brief",
            text="I see myself as someone who tends to be disorganized.",
            item_type="likert",
            scale="conscientiousness",
            reverse_scored=True,
        ),
        TestItem(
            item_id="BFI7",
            test_name="BFI_brief",
            text="I see myself as someone who worries a lot.",
            item_type="likert",
            scale="neuroticism",
            reverse_scored=False,
        ),
        TestItem(
            item_id="BFI8",
            test_name="BFI_brief",
            text="I see myself as someone who is relaxed, handles stress well.",
            item_type="likert",
            scale="neuroticism",
            reverse_scored=True,
        ),
        TestItem(
            item_id="BFI9",
            test_name="BFI_brief",
            text="I see myself as someone who is original, comes up with new ideas.",
            item_type="likert",
            scale="openness",
            reverse_scored=False,
        ),
        TestItem(
            item_id="BFI10",
            test_name="BFI_brief",
            text="I see myself as someone who has few artistic interests.",
            item_type="likert",
            scale="openness",
            reverse_scored=True,
        ),
    ]
    return Test(name="BFI_brief", items=items)


def build_nfc18_short_test() -> Test:
    """
    Construct a short NFC-18-style test (6 items: 3 positive, 3 reverse-scored).

    Returns
    -------
    Test
        NFC-18 style test object.
    """
    items: List[TestItem] = [
        TestItem(
            item_id="NFC1",
            test_name="NFC18",
            text="I prefer complex to simple problems.",
            item_type="likert",
            scale="nfc",
            reverse_scored=False,
        ),
        TestItem(
            item_id="NFC2",
            test_name="NFC18",
            text="I like tasks that require a lot of thinking.",
            item_type="likert",
            scale="nfc",
            reverse_scored=False,
        ),
        TestItem(
            item_id="NFC3",
            test_name="NFC18",
            text="I enjoy thinking about abstract ideas.",
            item_type="likert",
            scale="nfc",
            reverse_scored=False,
        ),
        TestItem(
            item_id="NFC4",
            test_name="NFC18",
            text="I only think as hard as I have to.",
            item_type="likert",
            scale="nfc",
            reverse_scored=True,
        ),
        TestItem(
            item_id="NFC5",
            test_name="NFC18",
            text="I prefer tasks that do not require much thought once I have learned them.",
            item_type="likert",
            scale="nfc",
            reverse_scored=True,
        ),
        TestItem(
            item_id="NFC6",
            test_name="NFC18",
            text="Thinking is not my idea of fun.",
            item_type="likert",
            scale="nfc",
            reverse_scored=True,
        ),
    ]
    return Test(name="NFC18", items=items)


def build_crt2_test() -> Test:
    """
    Construct a small CRT2-style test (3 classic items).

    Returns
    -------
    Test
        CRT2-style test object.
    """
    items: List[TestItem] = [
        TestItem(
            item_id="CRT1",
            test_name="CRT2",
            text=(
                "A bat and a ball cost $1.10 in total. The bat costs $1.00 more "
                "than the ball. How much does the ball cost (in cents)?"
            ),
            item_type="crt",
            correct_answer="5",
            incorrect_answer="10",
        ),
        TestItem(
            item_id="CRT2",
            test_name="CRT2",
            text=(
                "If it takes 5 machines 5 minutes to make 5 widgets, how long "
                "would it take 100 machines to make 100 widgets (in minutes)?"
            ),
            item_type="crt",
            correct_answer="5",
            incorrect_answer="100",
        ),
        TestItem(
            item_id="CRT3",
            test_name="CRT2",
            text=(
                "In a lake, there is a patch of lily pads. Every day, the patch "
                "doubles in size. If it takes 48 days for the patch to cover the "
                "entire lake, how long would it take for the patch to cover half "
                "of the lake (in days)?"
            ),
            item_type="crt",
            correct_answer="47",
            incorrect_answer="24",
        ),
    ]
    return Test(name="CRT2", items=items)


def build_bcrt_test() -> Test:
    """
    Construct a small behavioral CRT (bCRT) style test.

    Returns
    -------
    Test
        bCRT-style test object.
    """
    items: List[TestItem] = [
        TestItem(
            item_id="BCRT1",
            test_name="bCRT",
            text=(
                "You are offered a lottery: 10% chance to win $100, otherwise "
                "nothing. What is the expected value in dollars?"
            ),
            item_type="crt",
            correct_answer="10",
            incorrect_answer="100",
        ),
        TestItem(
            item_id="BCRT2",
            test_name="bCRT",
            text=(
                "You flip a fair coin three times. What is the probability of "
                "getting exactly two heads (as a fraction)?"
            ),
            item_type="crt",
            correct_answer="3/8",
            incorrect_answer="1/2",
        ),
        TestItem(
            item_id="BCRT3",
            test_name="bCRT",
            text=(
                "A jar contains 2 red and 8 blue balls. If you draw one ball at "
                "random, what is the probability of drawing a red ball (as a "
                "percentage)?"
            ),
            item_type="crt",
            correct_answer="20",
            incorrect_answer="80",
        ),
    ]
    return Test(name="bCRT", items=items)


def get_all_tests() -> List[Test]:
    """
    Construct all tests used in the simulation.

    Returns
    -------
    list of Test
        Tests: BFI_brief, NFC18, CRT2, bCRT.
    """
    return [
        build_bfi_brief_test(),
        build_nfc18_short_test(),
        build_crt2_test(),
        build_bcrt_test(),
    ]


# ---------------------------------------------------------------------------
# Simulation parameters and calibration
# ---------------------------------------------------------------------------


@dataclass
class SimulationParameters:
    """
    Parameters controlling the stochastic response generation.

    Attributes
    ----------
    bfi_slope : float
        Slope mapping BFI trait scores to Likert responses.
    bfi_intercept : float
        Intercept for BFI Likert mapping.
    bfi_noise : float
        Standard deviation of BFI response noise.
    nfc_slope : float
        Slope mapping NFC trait scores to Likert responses.
    nfc_intercept : float
        Intercept for NFC Likert mapping.
    nfc_noise : float
        Standard deviation of NFC response noise.
    crt_alpha : float
        Slope of logistic mapping from CRT2 level to probability of a correct
        CRT response.
    crt_beta : float
        Intercept of the logistic mapping.
    """

    bfi_slope: float = 1.0
    bfi_intercept: float = 0.0
    bfi_noise: float = 0.4
    nfc_slope: float = 1.0
    nfc_intercept: float = 0.0
    nfc_noise: float = 0.5
    crt_alpha: float = 1.0
    crt_beta: float = -1.0


class Calibrator:
    """
    Calibration algorithm for fitting SimulationParameters.

    The goal is to choose parameters so that simulated test scores reproduce
    the target trait scores in the training data as closely as possible.

    This implementation uses simple closed-form approximations rather than
    running the full simulator on the training set, keeping the evaluation
    interface stable and the calibration module replaceable.
    """

    def __init__(self) -> None:
        """Initialize an unfitted Calibrator."""
        self.params: Optional[SimulationParameters] = None

    def fit(self, train_df: pd.DataFrame) -> "Calibrator":
        """
        Fit SimulationParameters on the training personas.

        Parameters
        ----------
        train_df : pandas.DataFrame
            Training personas dataframe.

        Returns
        -------
        Calibrator
            Self, with `params` attribute set.
        """
        bfi_cols = [c for c in train_df.columns if c.startswith("bfi_")]
        if not bfi_cols:
            raise ValueError(
                "Training data must contain BFI trait columns starting with 'bfi_'."
            )

        x_vals: List[float] = []
        y_vals: List[float] = []
        for col in bfi_cols:
            vals = train_df[col].astype(float).values
            x_vals.append(vals)
            y_vals.append(vals)
        x_all = np.concatenate(x_vals)
        y_all = np.concatenate(y_vals)
        bfi_slope, bfi_intercept = fit_linear(x_all, y_all)

        if "nfc_score" in train_df.columns:
            x_nfc = train_df["nfc_score"].astype(float).values
            y_nfc = train_df["nfc_score"].astype(float).values
            nfc_slope, nfc_intercept = fit_linear(x_nfc, y_nfc)
        else:
            logging.warning(
                "No 'nfc_score' column found in training data; using default NFC parameters."
            )
            nfc_slope, nfc_intercept = 1.0, 0.0

        if "crt2_score" in train_df.columns:
            levels = train_df["crt2_score"].astype(float).values
            max_level = max(4.0, float(levels.max() or 4.0))
            p = (levels + 0.5) / (max_level + 1.0)
            p = np.clip(p, 1e-3, 1.0 - 1e-3)
            z = np.log(p / (1.0 - p))
            crt_alpha, crt_beta = fit_linear(levels, z)
        else:
            logging.warning(
                "No 'crt2_score' column found in training data; using default CRT parameters."
            )
            crt_alpha, crt_beta = 1.0, -1.0

        params = SimulationParameters(
            bfi_slope=bfi_slope,
            bfi_intercept=bfi_intercept,
            bfi_noise=0.4,
            nfc_slope=nfc_slope,
            nfc_intercept=nfc_intercept,
            nfc_noise=0.5,
            crt_alpha=crt_alpha,
            crt_beta=crt_beta,
        )
        self.params = params
        logging.info(
            "Calibrated parameters: %s",
            params,
        )
        return self


# ---------------------------------------------------------------------------
# LLM-based Memory, Planning, and Reasoning Agents for writing tasks
# ---------------------------------------------------------------------------


class MemoryAgent:
    """
    Simple Memory Agent that provides user and item context for LLM prompts.
    """

    def build_user_context(self, persona: Persona, persona_prompt: str) -> str:
        """
        Build a textual summary of persona demographics and traits.

        Parameters
        ----------
        persona : Persona
            Persona object.
        persona_prompt : str
            Full persona prompt used for configuration.

        Returns
        -------
        str
            User context description.
        """
        lines: List[str] = [f"Persona ID: {persona.persona_id}"]
        if persona.demographics:
            lines.append("Demographics:")
            for k, v in persona.demographics.items():
                lines.append(f"- {k}: {v}")
        if persona.traits:
            lines.append("Trait scores:")
            for k, v in persona.traits.items():
                lines.append(f"- {k}: {v}")
        if persona_prompt:
            lines.append("Persona configuration prompt:")
            lines.append(persona_prompt)
        return "\n".join(lines)

    def build_item_context_for_writing(self, task_prompt: str) -> str:
        """
        Build item/task context for a reflection writing task.

        Parameters
        ----------
        task_prompt : str
            Writing task prompt.

        Returns
        -------
        str
            Item context description.
        """
        return f"Writing task prompt: {task_prompt}"


class PlanningAgent:
    """
    Simple Planning Agent that sketches how the persona should respond.
    """

    def build_writing_plan(self, persona: Persona, task_prompt: str) -> str:
        """
        Create a short plan for how the persona should approach the writing task.

        Parameters
        ----------
        persona : Persona
            Persona object.
        task_prompt : str
            Writing task prompt.

        Returns
        -------
        str
            Plan/steps description.
        """
        extraversion = persona.traits.get("bfi_extraversion", 3.0)
        nfc = persona.traits.get("nfc_score", 3.0)
        neuroticism = persona.traits.get("bfi_neuroticism", 3.0)
        openness = persona.traits.get("bfi_openness", 3.0)

        steps: List[str] = []
        steps.append(
            "1. Restate the task in your own words and think about a relevant recent experience."
        )
        if extraversion > 3.5:
            steps.append(
                "2. Use a fairly expressive and detailed narrative style when describing the experience."
            )
        else:
            steps.append(
                "2. Use a more concise and reserved style, focusing on internal thoughts."
            )
        if nfc > 3.5:
            steps.append(
                "3. Analyse the situation logically, explaining your reasoning and how you evaluated options."
            )
        else:
            steps.append(
                "3. Describe the most salient aspects without going into heavy analytic detail."
            )
        if neuroticism > 3.5:
            steps.append(
                "4. Include some mention of worries, doubts, or emotional tension you might have felt."
            )
        else:
            steps.append(
                "4. Emphasise how you stayed calm or managed any stress in a balanced way."
            )
        if openness > 3.5:
            steps.append(
                "5. Connect the experience to broader ideas, lessons learned, or creative insights."
            )
        else:
            steps.append(
                "5. Focus mainly on concrete details of what happened and the immediate outcome."
            )
        steps.append(
            "6. Write in the first person ('I') as if you are this persona, without referring to being an AI or simulation."
        )
        return "\n".join(steps)


class ReasoningAgent:
    """
    Reasoning Agent that uses an LLM to generate writing-task responses.

    It integrates context from the Memory Agent and a plan from the Planning
    Agent and performs its reasoning via an OpenAI GPT-5 Responses API call.
    """

    def __init__(
        self,
        memory_agent: MemoryAgent,
        planning_agent: PlanningAgent,
        model: str = "gpt-5",
        max_output_tokens: int = 800,
    ) -> None:
        self.memory_agent = memory_agent
        self.planning_agent = planning_agent
        self.model = model
        self.max_output_tokens = max_output_tokens

    def generate_writing_response(
        self, persona: Persona, persona_prompt: str, task_prompt: str
    ) -> str:
        """
        Generate a writing-task response for a persona using an LLM call.

        Parameters
        ----------
        persona : Persona
            Persona producing the writing.
        persona_prompt : str
            Full persona prompt that configures the LLM.
        task_prompt : str
            Writing task prompt.

        Returns
        -------
        str
            Generated response text. Falls back to a heuristic response if
            the LLM call fails (e.g., missing API key or network error).
        """
        user_context = self.memory_agent.build_user_context(
            persona, persona_prompt
        )
        item_context = self.memory_agent.build_item_context_for_writing(
            task_prompt
        )
        plan = self.planning_agent.build_writing_plan(persona, task_prompt)

        full_prompt = (
            "You are participating in a research simulation where a large language model "
            "emulates a human persona completing reflection-related writing tasks.\n\n"
            "USER CONTEXT (from Memory Agent):\n"
            f"{user_context}\n\n"
            "TASK CONTEXT (from Memory Agent):\n"
            f"{item_context}\n\n"
            "PLAN / STEPS (from Planning Agent):\n"
            f"{plan}\n\n"
            "Using the information above, write the persona's response to the writing task. "
            "Follow the plan, but DO NOT mention the plan, the Memory Agent, the Planning Agent, "
            "or that this is a simulation. Write in the first person ('I'), as a coherent "
            "short essay of one to three paragraphs directly answering the task prompt.\n\n"
            "Persona's written response:"
        )

        try:
            response_text = call_gpt5_with_responses_api(
                prompt=full_prompt,
                model=self.model,
                max_output_tokens=self.max_output_tokens,
            )
            text = (response_text or "").strip()
            if not text:
                raise ValueError("Empty response from LLM.")
            return text
        except Exception as exc:
            logging.warning(
                "LLM call for writing task failed (%s); falling back to heuristic text.",
                exc,
            )
            return heuristic_writing_response(persona, task_prompt)


# ---------------------------------------------------------------------------
# Response generation (pseudo LLM for structured test items)
# ---------------------------------------------------------------------------


class ResponseGenerator:
    """
    Stochastic response generator that mimics an LLM conditioned on persona
    prompts and traits, using calibrated SimulationParameters.

    Responses are deterministic given the global random seed and persona data
    for structured psychometric items. Writing tasks may be delegated to an
    actual LLM via the ReasoningAgent.
    """

    def __init__(self, params: SimulationParameters, seed: int) -> None:
        """
        Initialize the response generator.

        Parameters
        ----------
        params : SimulationParameters
            Calibrated simulation parameters.
        seed : int
            Global seed used to initialize an internal random generator.
        """
        self.params = params
        self.rng = np.random.RandomState(seed)

    def _likert_from_trait(
        self,
        trait_value: float,
        scale_min: int,
        scale_max: int,
        slope: float,
        intercept: float,
        noise: float,
    ) -> float:
        """
        Generate a Likert-style response given a trait value.

        Parameters
        ----------
        trait_value : float
            Underlying trait score (e.g., 1-5).
        scale_min : int
            Minimum Likert value.
        scale_max : int
            Maximum Likert value.
        slope : float
            Linear mapping slope.
        intercept : float
            Linear mapping intercept.
        noise : float
            Standard deviation of Gaussian noise.

        Returns
        -------
        float
            Simulated Likert response (clipped to [scale_min, scale_max]).
        """
        mean = slope * trait_value + intercept
        resp = mean + self.rng.normal(0.0, noise)
        return float(np.clip(resp, scale_min, scale_max))

    def _crt_prob_correct(self, crt_level: float) -> float:
        """
        Compute probability of a correct CRT response given CRT level.

        Parameters
        ----------
        crt_level : float
            CRT2 level in [0, 4].

        Returns
        -------
        float
            Probability in [0, 1].
        """
        z = self.params.crt_alpha * crt_level + self.params.crt_beta
        p = 1.0 / (1.0 + np.exp(-z))
        return float(np.clip(p, 0.01, 0.99))

    def simulate_bfi_item(self, persona: Persona, item: TestItem) -> float:
        """
        Simulate a single BFI-brief Likert item response.

        Parameters
        ----------
        persona : Persona
            The persona answering the item.
        item : TestItem
            The item being answered (must belong to BFI_brief).

        Returns
        -------
        float
            Simulated Likert response (1-5).
        """
        if item.scale is None:
            raise ValueError("BFI item must have a 'scale' attribute.")
        trait_col = f"bfi_{item.scale}"
        trait_val = persona.traits.get(trait_col)
        if trait_val is None:
            raise ValueError(
                f"Persona {persona.persona_id} is missing trait {trait_col} "
                "needed for BFI simulation."
            )
        resp = self._likert_from_trait(
            trait_value=trait_val,
            scale_min=1,
            scale_max=5,
            slope=self.params.bfi_slope,
            intercept=self.params.bfi_intercept,
            noise=self.params.bfi_noise,
        )
        if item.reverse_scored:
            resp = 6.0 - resp
        return resp

    def simulate_nfc_item(self, persona: Persona, item: TestItem) -> float:
        """
        Simulate a single NFC Likert item response.

        Parameters
        ----------
        persona : Persona
            The persona answering the item.
        item : TestItem
            The NFC item.

        Returns
        -------
        float
            Simulated Likert response (1-7).
        """
        trait_val = persona.traits.get("nfc_score")
        if trait_val is None:
            raise ValueError(
                f"Persona {persona.persona_id} is missing 'nfc_score' trait."
            )
        resp = self._likert_from_trait(
            trait_value=trait_val,
            scale_min=1,
            scale_max=7,
            slope=self.params.nfc_slope,
            intercept=self.params.nfc_intercept,
            noise=self.params.nfc_noise,
        )
        if item.reverse_scored:
            resp = 8.0 - resp
        return resp

    def simulate_crt_item(self, persona: Persona, item: TestItem) -> Tuple[str, bool]:
        """
        Simulate a single CRT-style item response (answer string and correctness).

        Parameters
        ----------
        persona : Persona
            The persona answering the item.
        item : TestItem
            The CRT item.

        Returns
        -------
        (answer, is_correct) : tuple
            Simulated answer string and correctness flag.
        """
        crt_level = persona.traits.get("crt2_score", 2.0)
        prob_correct = self._crt_prob_correct(crt_level)
        is_correct = self.rng.rand() < prob_correct
        if is_correct or not item.incorrect_answer:
            answer = item.correct_answer or ""
        else:
            answer = item.incorrect_answer
        return answer, bool(is_correct)

    def simulate_writing_task(
        self,
        persona: Persona,
        prompt: str,
        reasoning_agent: Optional[ReasoningAgent] = None,
        persona_prompt: str = "",
    ) -> str:
        """
        Simulate a short written response to a reflection-related writing task.

        If a ReasoningAgent is provided, it will perform an LLM-based call
        combining Memory Agent and Planning Agent outputs. Otherwise, a
        heuristic text is generated locally.

        Parameters
        ----------
        persona : Persona
            Persona producing the writing.
        prompt : str
            Writing task prompt.
        reasoning_agent : ReasoningAgent, optional
            Agent that uses an LLM to generate responses.
        persona_prompt : str, optional
            Full persona prompt used for configuration (for the ReasoningAgent).

        Returns
        -------
        str
            Simulated written response.
        """
        if reasoning_agent is not None:
            return reasoning_agent.generate_writing_response(
                persona=persona,
                persona_prompt=persona_prompt,
                task_prompt=prompt,
            )
        return heuristic_writing_response(persona, prompt)


# ---------------------------------------------------------------------------
# Simple LIWC-style analysis for writing
# ---------------------------------------------------------------------------


def heuristic_writing_response(persona: Persona, prompt: str) -> str:
    """
    Heuristic (non-LLM) generation of a short written response to a reflection task.

    Parameters
    ----------
    persona : Persona
        Persona producing the writing.
    prompt : str
        Writing task prompt.

    Returns
    -------
    str
        Heuristically generated text.
    """
    extraversion = persona.traits.get("bfi_extraversion", 3.0)
    nfc = persona.traits.get("nfc_score", 3.0)
    neuroticism = persona.traits.get("bfi_neuroticism", 3.0)
    openness = persona.traits.get("bfi_openness", 3.0)

    tone = []
    if extraversion > 3.5:
        tone.append("I enjoy talking about my experiences in detail.")
    else:
        tone.append("I prefer to reflect quietly on what happened.")
    if nfc > 3.5:
        tone.append(
            "I carefully analyse the situation, considering multiple perspectives."
        )
    else:
        tone.append("I focus on the most obvious aspects without overthinking.")
    if neuroticism > 3.5:
        tone.append(
            "This makes me a bit anxious, and I often worry about possible mistakes."
        )
    else:
        tone.append(
            "I stay relatively calm and do not let worries distract me too much."
        )
    if openness > 3.5:
        tone.append(
            "I connect this experience to broader ideas and creative possibilities."
        )
    else:
        tone.append("I mostly think about the concrete details.")

    text = f"Prompt: {prompt}\nAs this persona, I respond: " + " ".join(tone)
    return text


def analyse_writing(text: str) -> Dict[str, float]:
    """
    Compute simple LIWC-like features from a text.

    Features include:
    - word_count
    - analytic_ratio: fraction of words that are analytic markers
    - insight_ratio: fraction of words that are insight markers
    - affect_ratio: fraction of words that are affect markers

    Parameters
    ----------
    text : str
        Input text.

    Returns
    -------
    dict
        Mapping from feature name to value.
    """
    words = [w.strip(".,!?;:").lower() for w in text.split() if w.strip()]
    n = len(words) or 1
    analytic_markers = {"because", "therefore", "hence", "analysis", "reason"}
    insight_markers = {"realise", "realize", "understand", "insight", "learned"}
    affect_markers = {"happy", "sad", "anxious", "worried", "excited", "afraid"}
    analytic = sum(w in analytic_markers for w in words) / n
    insight = sum(w in insight_markers for w in words) / n
    affect = sum(w in affect_markers for w in words) / n
    return {
        "word_count": float(len(words)),
        "analytic_ratio": float(analytic),
        "insight_ratio": float(insight),
        "affect_ratio": float(affect),
    }


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------


class Simulator:
    """
    Forward simulator that rolls out tests and writing tasks for personas
    under a given configuration and calibrated parameters.
    """

    def __init__(
        self,
        config: SimulationConfig,
        calibrator: Calibrator,
    ) -> None:
        """
        Initialize the simulator.

        Parameters
        ----------
        config : SimulationConfig
            Simulation configuration.
        calibrator : Calibrator
            Fitted calibrator providing SimulationParameters.
        """
        if calibrator.params is None:
            raise ValueError(
                "Calibrator must be fitted before creating a Simulator "
                "(calibrator.params is None)."
            )
        self.config = config
        self.params = calibrator.params

    def _get_ordered_tests(self) -> List[Test]:
        """
        Get tests in configured order, optionally shuffled.

        Returns
        -------
        list of Test
            Ordered tests.
        """
        tests = get_all_tests()
        if self.config.shuffle_test_order:
            random.shuffle(tests)
        return tests

    def _maybe_shuffle_items(self, test: Test) -> List[TestItem]:
        """
        Optionally shuffle items within a test based on configuration.

        Parameters
        ----------
        test : Test
            The test whose items may be shuffled.

        Returns
        -------
        list of TestItem
            Ordered (possibly shuffled) items.
        """
        items = list(test.items)
        if self.config.shuffle_items_within_tests:
            random.shuffle(items)
        return items

    def rollout(
        self, valid_df: pd.DataFrame, network: InteractionNetwork
    ) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        """
        Run the simulator on the validation personas.

        Parameters
        ----------
        valid_df : pandas.DataFrame
            Validation personas.
        network : InteractionNetwork
            Interaction network over validation personas. It is used to
            propagate simple scalar signals (CRT2 and NFC levels) prior to
            response generation, modelling basic social/informational influence.

        Returns
        -------
        (sim_df, raw_records) : tuple
            sim_df : pandas.DataFrame
                Row-per-persona dataset with persona attributes, aggregated
                test scores, item-level responses, writing metrics, and
                metadata (e.g., num_api_calls).
            raw_records : list of dict
                More detailed JSON-serializable records including nested
                raw responses.

        Notes
        -----
        Structured psychometric items are always simulated locally using
        ResponseGenerator, regardless of `admin_mode`. The `admin_mode`
        parameter affects how many *conceptual* API calls are counted in
        `num_api_calls` for budgeting analyses. Actual OpenAI LLM calls
        occur only for reflection-related writing tasks via ReasoningAgent.
        """
        response_gen = ResponseGenerator(self.params, seed=self.config.seed)
        tests = self._get_ordered_tests()

        memory_agent = MemoryAgent()
        planning_agent = PlanningAgent()
        reasoning_agent = ReasoningAgent(
            memory_agent=memory_agent,
            planning_agent=planning_agent,
            model="gpt-5",
            max_output_tokens=800,
        )

        writing_tasks = [
            "Describe a recent challenging problem you faced and how you handled it.",
            "Reflect on a decision you made that required careful thinking.",
        ]

        # ------------------------------------------------------------------
        # Network-based trait propagation: influence CRT2 and NFC traits
        # ------------------------------------------------------------------
        try:
            if network.agents:
                initial_crt = {
                    pid: agent.traits.get("crt2_score", 2.0)
                    for pid, agent in network.agents.items()
                }
                propagated_crt = network.propagate_signal(
                    layer="information", initial_values=initial_crt, n_steps=2
                )

                initial_nfc = {
                    pid: agent.traits.get("nfc_score", 3.0)
                    for pid, agent in network.agents.items()
                }
                propagated_nfc = network.propagate_signal(
                    layer="social", initial_values=initial_nfc, n_steps=2
                )

                for pid, agent in network.agents.items():
                    if "crt2_score" in agent.traits:
                        base = float(agent.traits["crt2_score"])
                        influenced = float(propagated_crt.get(pid, base))
                        agent.traits["crt2_score"] = float(
                            0.7 * base + 0.3 * influenced
                        )
                    if "nfc_score" in agent.traits:
                        base = float(agent.traits["nfc_score"])
                        influenced = float(propagated_nfc.get(pid, base))
                        agent.traits["nfc_score"] = float(
                            0.7 * base + 0.3 * influenced
                        )
        except Exception as exc:
            logging.warning(
                "Network-based trait propagation failed (%s); proceeding without network influence.",
                exc,
            )

        records: List[Dict[str, Any]] = []
        raw_records: List[Dict[str, Any]] = []

        # Use the network's Persona objects where available to avoid duplicate
        # construction and to ensure we use network-influenced traits.
        example_markers: Dict[str, Dict[str, List[str]]] = {}
        if network.agents:
            example_markers = next(iter(network.agents.values())).markers

        for _, row in valid_df.iterrows():
            persona_id = str(row.get("persona_id", ""))
            persona = network.agents.get(persona_id)
            if persona is None:
                # Fallback: construct persona directly from row if not present in network.
                persona = Persona.from_series(row, example_markers)

            persona_prompt = persona.build_full_persona_prompt(
                crt_numeric_only=self.config.crt_numeric_only,
                bfi_mode=self.config.bfi_prompt_mode,
            )
            num_api_calls = 0

            agg_scores: Dict[str, Any] = {}
            item_scores: Dict[str, Any] = {}
            raw_details: Dict[str, Any] = {
                "persona_id": persona.persona_id,
                "persona_prompt": persona_prompt,
                "tests": {},
                "writing_tasks": {},
            }

            if self.config.admin_mode == "all_tests":
                num_api_calls += 1
                for test in tests:
                    raw_details["tests"][test.name] = {"items": []}
                    for item in self._maybe_shuffle_items(test):
                        (
                            resp_value,
                            correct_flag,
                            item_col_name,
                        ) = self._simulate_item(persona, response_gen, item)
                        raw_details["tests"][test.name]["items"].append(
                            {
                                "item_id": item.item_id,
                                "text": item.text,
                                "response": resp_value,
                                "is_correct": correct_flag,
                            }
                        )
                        item_scores[item_col_name] = (
                            resp_value
                            if isinstance(resp_value, (int, float))
                            else None
                        )
                        self._update_agg_scores(
                            agg_scores, test, item, resp_value, correct_flag
                        )
            elif self.config.admin_mode == "per_test":
                for test in tests:
                    num_api_calls += 1
                    raw_details["tests"][test.name] = {"items": []}
                    for item in self._maybe_shuffle_items(test):
                        (
                            resp_value,
                            correct_flag,
                            item_col_name,
                        ) = self._simulate_item(persona, response_gen, item)
                        raw_details["tests"][test.name]["items"].append(
                            {
                                "item_id": item.item_id,
                                "text": item.text,
                                "response": resp_value,
                                "is_correct": correct_flag,
                            }
                        )
                        item_scores[item_col_name] = (
                            resp_value
                            if isinstance(resp_value, (int, float))
                            else None
                        )
                        self._update_agg_scores(
                            agg_scores, test, item, resp_value, correct_flag
                        )
            elif self.config.admin_mode == "per_item":
                for test in tests:
                    raw_details["tests"][test.name] = {"items": []}
                    for item in self._maybe_shuffle_items(test):
                        num_api_calls += 1
                        (
                            resp_value,
                            correct_flag,
                            item_col_name,
                        ) = self._simulate_item(persona, response_gen, item)
                        raw_details["tests"][test.name]["items"].append(
                            {
                                "item_id": item.item_id,
                                "text": item.text,
                                "response": resp_value,
                                "is_correct": correct_flag,
                            }
                        )
                        item_scores[item_col_name] = (
                            resp_value
                            if isinstance(resp_value, (int, float))
                            else None
                        )
                        self._update_agg_scores(
                            agg_scores, test, item, resp_value, correct_flag
                        )
            else:
                raise ValueError(f"Unknown admin_mode: {self.config.admin_mode!r}")

            # Simulate writing tasks using the ReasoningAgent (LLM-based where available)
            for wt_idx, wt_prompt in enumerate(writing_tasks, start=1):
                num_api_calls += 1
                task_id = f"writing_task_{wt_idx}"
                text = response_gen.simulate_writing_task(
                    persona,
                    wt_prompt,
                    reasoning_agent=reasoning_agent,
                    persona_prompt=persona_prompt,
                )
                metrics = analyse_writing(text)
                raw_details["writing_tasks"][task_id] = {
                    "prompt": wt_prompt,
                    "text": text,
                    "metrics": metrics,
                }
                for k, v in metrics.items():
                    agg_scores[f"{task_id}_{k}"] = v

            agg_scores["num_api_calls"] = num_api_calls

            persona_data = row.to_dict()
            persona_record = {**persona_data, **agg_scores, **item_scores}
            records.append(persona_record)
            raw_records.append({**persona_record, "raw_responses": raw_details})

        sim_df = pd.DataFrame(records)
        logging.info(
            "Simulation rollout produced %d persona records with %d columns.",
            len(sim_df),
            sim_df.shape[1],
        )
        return sim_df, raw_records

    def _simulate_item(
        self,
        persona: Persona,
        response_gen: ResponseGenerator,
        item: TestItem,
    ) -> Tuple[Any, Optional[bool], str]:
        """
        Simulate a single item response and return structured information.

        Parameters
        ----------
        persona : Persona
            Persona answering.
        response_gen : ResponseGenerator
            Response generator.
        item : TestItem
            Item to simulate.

        Returns
        -------
        (resp_value, correct_flag, item_col_name) : tuple
            resp_value : numeric value or string answer.
            correct_flag : bool or None for non-CRT items.
            item_col_name : str
                Column name used for item-level numeric scores in the output.
        """
        if item.test_name == "BFI_brief":
            resp_value = response_gen.simulate_bfi_item(persona, item)
            correct_flag = None
            item_col_name = f"BFI_{item.item_id}"
        elif item.test_name == "NFC18":
            resp_value = response_gen.simulate_nfc_item(persona, item)
            correct_flag = None
            item_col_name = f"NFC_{item.item_id}"
        elif item.test_name in ("CRT2", "bCRT"):
            answer, is_correct = response_gen.simulate_crt_item(persona, item)
            resp_value = answer
            correct_flag = is_correct
            item_col_name = f"{item.test_name}_{item.item_id}_correct"
        else:
            raise ValueError(f"Unknown test name: {item.test_name!r}")
        return resp_value, correct_flag, item_col_name

    @staticmethod
    def _update_agg_scores(
        agg_scores: Dict[str, Any],
        test: Test,
        item: TestItem,
        resp_value: Any,
        correct_flag: Optional[bool],
    ) -> None:
        """
        Update aggregated scale/test scores based on a single item response.

        Parameters
        ----------
        agg_scores : dict
            Aggregated scores dictionary to update in-place.
        test : Test
            Test the item belongs to.
        item : TestItem
            Item.
        resp_value : Any
            Response value.
        correct_flag : bool or None
            Correctness for CRT items.
        """
        if test.name == "BFI_brief" and isinstance(resp_value, (int, float)):
            if item.scale is None:
                return
            key = f"bfi_{item.scale}_sim_sum"
            count_key = f"bfi_{item.scale}_sim_count"
            agg_scores[key] = agg_scores.get(key, 0.0) + float(resp_value)
            agg_scores[count_key] = agg_scores.get(count_key, 0) + 1
            mean_key = f"bfi_{item.scale}_sim"
            agg_scores[mean_key] = agg_scores[key] / agg_scores[count_key]

        elif test.name == "NFC18" and isinstance(resp_value, (int, float)):
            key = "nfc_sim_sum"
            count_key = "nfc_sim_count"
            agg_scores[key] = agg_scores.get(key, 0.0) + float(resp_value)
            agg_scores[count_key] = agg_scores.get(count_key, 0) + 1
            mean_key = "nfc_score_sim"
            agg_scores[mean_key] = agg_scores[key] / agg_scores[count_key]

        elif test.name in ("CRT2", "bCRT"):
            if correct_flag is None:
                return
            key = f"{test.name}_correct_sum"
            agg_scores[key] = agg_scores.get(key, 0.0) + (
                1.0 if correct_flag else 0.0
            )
            if test.name == "CRT2":
                agg_scores["crt2_score_sim"] = agg_scores[key]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


class Evaluator:
    """
    Evaluation of simulator outputs against ground-truth persona traits.

    Metrics cover:
    - Trait fidelity (correlations, MAE, monotonicity).
    - Behavioural alignment (relationships between traits and writing metrics,
      including simple regressions).
    - Response consistency (internal reliability via Cronbach's alpha).
    """

    def compute_metrics(
        self, valid_df: pd.DataFrame, sim_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Compute all evaluation metrics.

        Parameters
        ----------
        valid_df : pandas.DataFrame
            Validation personas (ground truth traits).
        sim_df : pandas.DataFrame
            Simulation results for validation personas.

        Returns
        -------
        dict
            Nested metrics dictionary.
        """
        metrics: Dict[str, Any] = {}
        metrics["trait_fidelity"] = self._trait_fidelity(valid_df, sim_df)
        metrics["behavioural_alignment"] = self._behavioural_alignment(
            valid_df, sim_df
        )
        metrics["response_consistency"] = self._response_consistency(sim_df)
        logging.info("Computed evaluation metrics.")
        return metrics

    def _trait_fidelity(
        self, valid_df: pd.DataFrame, sim_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Compute trait fidelity metrics.

        Parameters
        ----------
        valid_df : pandas.DataFrame
            Validation personas.
        sim_df : pandas.DataFrame
            Simulation results.

        Returns
        -------
        dict
            Trait fidelity metrics.
        """
        results: Dict[str, Any] = {"by_trait": {}}

        for domain in [
            "extraversion",
            "agreeableness",
            "conscientiousness",
            "neuroticism",
            "openness",
        ]:
            target_col = f"bfi_{domain}"
            sim_col = f"bfi_{domain}_sim"
            if target_col in valid_df.columns and sim_col in sim_df.columns:
                target = valid_df[target_col].astype(float).values
                sim = sim_df[sim_col].astype(float).values
                r_pearson = pearsonr(target, sim)
                r_spearman = spearmanr(target, sim)
                mae = float(np.mean(np.abs(target - sim)))
                order = np.argsort(target)
                sim_sorted = sim[order]
                diffs = np.diff(sim_sorted)
                violations = np.sum(diffs < 0)
                monotonicity = 1.0 - violations / max(1, len(diffs))
                results["by_trait"][domain] = {
                    "pearson_r": r_pearson,
                    "spearman_r": r_spearman,
                    "mae": mae,
                    "monotonicity_score": float(monotonicity),
                }

        if "nfc_score" in valid_df.columns and "nfc_score_sim" in sim_df.columns:
            target = valid_df["nfc_score"].astype(float).values
            sim = sim_df["nfc_score_sim"].astype(float).values
            r_pearson = pearsonr(target, sim)
            r_spearman = spearmanr(target, sim)
            mae = float(np.mean(np.abs(target - sim)))
            order = np.argsort(target)
            sim_sorted = sim[order]
            diffs = np.diff(sim_sorted)
            violations = np.sum(diffs < 0)
            monotonicity = 1.0 - violations / max(1, len(diffs))
            results["by_trait"]["nfc"] = {
                "pearson_r": r_pearson,
                "spearman_r": r_spearman,
                "mae": mae,
                "monotonicity_score": float(monotonicity),
            }

        if "crt2_score" in valid_df.columns and "crt2_score_sim" in sim_df.columns:
            target = valid_df["crt2_score"].astype(float).values
            sim = sim_df["crt2_score_sim"].astype(float).values
            r_pearson = pearsonr(target, sim)
            r_spearman = spearmanr(target, sim)
            mae = float(np.mean(np.abs(target - sim)))
            order = np.argsort(target)
            sim_sorted = sim[order]
            diffs = np.diff(sim_sorted)
            violations = np.sum(diffs < 0)
            monotonicity = 1.0 - violations / max(1, len(diffs))
            results["by_trait"]["crt2"] = {
                "pearson_r": r_pearson,
                "spearman_r": r_spearman,
                "mae": mae,
                "monotonicity_score": float(monotonicity),
            }

        return results

    def _behavioural_alignment(
        self, valid_df: pd.DataFrame, sim_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Compute behavioural alignment metrics via correlations and simple
        linear regressions between traits and writing-task features.

        Notes
        -----
        This implementation focuses on trait–feature associations and does
        not yet perform explicit cross-condition comparisons between different
        BFI persona prompt modes (e.g., coarse vs granular prompts).
        """
        results: Dict[str, Any] = {"correlations": {}, "regressions": {}}
        writing_features = [
            "writing_task_1_analytic_ratio",
            "writing_task_1_insight_ratio",
            "writing_task_1_affect_ratio",
            "writing_task_2_analytic_ratio",
            "writing_task_2_insight_ratio",
            "writing_task_2_affect_ratio",
        ]
        trait_cols = [
            "bfi_extraversion",
            "bfi_neuroticism",
            "nfc_score",
            "crt2_score",
        ]

        for trait_col in trait_cols:
            if trait_col not in valid_df.columns:
                continue
            trait = valid_df[trait_col].astype(float).values
            for feat in writing_features:
                if feat not in sim_df.columns:
                    continue
                feat_vals = sim_df[feat].astype(float).values
                r = pearsonr(trait, feat_vals)
                results["correlations"].setdefault(trait_col, {})[feat] = r

                a, b, r2 = simple_linear_regression_metrics(trait, feat_vals)
                results["regressions"].setdefault(trait_col, {})[feat] = {
                    "slope": a,
                    "intercept": b,
                    "r2": r2,
                }

        return results

    def _response_consistency(self, sim_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Compute response consistency metrics (Cronbach's alpha) for BFI and NFC.

        Notes
        -----
        The current simulator runs a single administration per persona, so
        test-retest or within-session order-shuffle consistency cannot be
        directly estimated from this dataset. Those metrics are therefore
        not included here; we report only internal reliability indices.
        """
        results: Dict[str, Any] = {}

        bfi_item_cols = [c for c in sim_df.columns if c.startswith("BFI_")]
        if bfi_item_cols:
            item_scores = sim_df[bfi_item_cols].astype(float).values
            alpha = cronbach_alpha(item_scores)
            results["bfi_cronbach_alpha"] = alpha

        nfc_item_cols = [c for c in sim_df.columns if c.startswith("NFC_")]
        if nfc_item_cols:
            item_scores = sim_df[nfc_item_cols].astype(float).values
            alpha = cronbach_alpha(item_scores)
            results["nfc_cronbach_alpha"] = alpha

        results["limitations"] = (
            "Test-retest and within-session order-shuffle consistency metrics "
            "are not available in this version because only a single "
            "administration per persona is simulated."
        )

        return results


# ---------------------------------------------------------------------------
# Saving results
# ---------------------------------------------------------------------------


def save_results(
    sim_df: pd.DataFrame, raw_records: List[Dict[str, Any]], metrics: Dict[str, Any]
) -> None:
    """
    Save simulation results and evaluation metrics to disk.

    Files created:
    - simulation_results.csv : flat, analysis-ready dataset.
    - simulation_results.json : JSON list of persona records including nested
      raw responses.
    - evaluation_metrics.json : JSON dictionary of metrics.

    Parameters
    ----------
    sim_df : pandas.DataFrame
        Simulation dataframe.
    raw_records : list of dict
        JSON-serializable detailed records.
    metrics : dict
        Evaluation metrics.
    """
    ensure_data_dir()
    results_csv = os.path.join(DATA_DIR, "simulation_results.csv")
    results_json = os.path.join(DATA_DIR, "simulation_results.json")
    metrics_json = os.path.join(DATA_DIR, "evaluation_metrics.json")

    try:
        sim_df.to_csv(results_csv, index=False)
        logging.info("Saved flat simulation results to %s.", results_csv)
    except OSError as e:
        logging.error(
            "Failed to save flat simulation results to %s: %s", results_csv, e
        )
        raise

    try:
        with open(results_json, "w", encoding="utf-8") as f:
            json.dump(raw_records, f, indent=2, ensure_ascii=False)
        logging.info("Saved detailed simulation results to %s.", results_json)
    except OSError as e:
        logging.error(
            "Failed to save detailed simulation results to %s: %s",
            results_json,
            e,
        )
        raise

    try:
        with open(metrics_json, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        logging.info("Saved evaluation metrics to %s.", metrics_json)
    except OSError as e:
        logging.error(
            "Failed to save evaluation metrics to %s: %s", metrics_json, e
        )
        raise


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


def main() -> None:
    """
    Main entry point orchestrating the entire simulation pipeline.

    Order of operations (as required):
    - parse_cli()
    - load_data()
    - holdout_split()
    - build_network_and_agents() on the validation subset
    - calibrator.fit()
    - simulator.rollout()
    - evaluator.compute_metrics()
    - save_results()
    """
    config = parse_cli()
    set_global_seed(config.seed)

    personas_df, markers_df = load_data()
    train_df, valid_df = holdout_split(personas_df, config)
    network = build_network_and_agents(valid_df, markers_df, config)

    calibrator = Calibrator()
    calibrator.fit(train_df)

    simulator = Simulator(config=config, calibrator=calibrator)
    sim_df, raw_records = simulator.rollout(valid_df, network)

    evaluator = Evaluator()
    metrics = evaluator.compute_metrics(valid_df, sim_df)

    save_results(sim_df, raw_records, metrics)


# Execute main for both direct execution and sandbox wrapper invocation
main()