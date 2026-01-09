#!/usr/bin/env python
"""
simulate.py

End-to-end multi-agent simulator for LLM-based personas parameterised by
human trait scores (BFI, Need for Cognition, CRT2).

The script:

1. Parses CLI arguments (parse_cli).
2. Loads input data (load_data) from:
   - synthetic_personas.csv
   - serapio_goldberg_markers.csv
   - bfi10_items.yaml
   - nfc18_items.yaml
   - crt2_items.yaml
   - bcrt_items.yaml
   - writing_tasks.yaml
3. Builds agents and a minimal multilayer social network (build_network_and_agents).
4. Performs a temporal-like holdout split into train/validation (holdout_split).
5. Calibrates simulator parameters on the training personas (Calibrator.fit).
6. Runs a forward simulation (Simulator.rollout) on the validation personas.
7. Evaluates simulation outputs against target traits (Evaluator.compute_metrics).
8. Saves results and metrics to disk (save_results).

Environment variables required for data paths:

    PROJECT_ROOT=/absolute/path/to/project/root
    DATA_PATH=relative/or/absolute/path/to/data

Data files are resolved as:

    DATA_DIR = os.path.join(PROJECT_ROOT, DATA_PATH)
    synthetic_personas.csv -> os.path.join(DATA_DIR, "synthetic_personas.csv")
    serapio_goldberg_markers.csv -> os.path.join(DATA_DIR, "serapio_goldberg_markers.csv")

The code is deterministic given a global random seed (for all internal randomness).
Psychometric test responses (BFI, NFC, CRT2, bCRT) are currently simulated using
deterministic/statistical logic; the OpenAI LLM is used only for the writing
task. The `admin_mode` configuration therefore affects only how simulations are
grouped and how administration metadata and ordering are recorded; it does not
change LLM usage for psychometric tests.
"""

import argparse
import json
import logging
import math
import os
import random
import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

# Optional OpenAI integration
try:
    from openai import OpenAI  # type: ignore

    OPENAI_AVAILABLE = True
except ImportError:  # pragma: no cover - handled gracefully at runtime
    OpenAI = None  # type: ignore
    OPENAI_AVAILABLE = False

# ---------------------------------------------------------------------------
# Global constants and random seed
# ---------------------------------------------------------------------------

GLOBAL_RANDOM_SEED = 42

random.seed(GLOBAL_RANDOM_SEED)
np.random.seed(GLOBAL_RANDOM_SEED)

# Canonical test-name constants
TEST_BFI = "BFI"
TEST_NFC = "NFC"
TEST_CRT2 = "CRT2"
TEST_BCRT = "bCRT"
TEST_WRITING = "Writing"

# ---------------------------------------------------------------------------
# Required environment-based data directory
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.environ.get("PROJECT_ROOT")
DATA_PATH = os.environ.get("DATA_PATH")
DATA_DIR = os.path.join(PROJECT_ROOT, DATA_PATH) if PROJECT_ROOT and DATA_PATH else None


# ---------------------------------------------------------------------------
# OpenAI integration helpers
# ---------------------------------------------------------------------------


def get_openai_api_key() -> str:
    """
    Retrieve the OpenAI API key from the environment.

    Returns
    -------
    str
        The API key string.

    Raises
    ------
    RuntimeError
        If the OpenAI Python package is not installed.
    ValueError
        If OPENAI_API_KEY is not set in the environment.
    """
    if not OPENAI_AVAILABLE:
        raise RuntimeError(
            "OpenAI Python package is not installed. Install 'openai' to enable "
            "LLM-backed writing task generation."
        )
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        return api_key
    raise ValueError("OpenAI API key not found in environment")


def call_gpt5_with_responses_api(
    prompt: str,
    model: str = "gpt-5",
    max_output_tokens: int = 4000,
) -> str:
    """
    Call an OpenAI LLM using the Responses API and return the generated text.

    Parameters
    ----------
    prompt:
        Full text prompt for the model.
    model:
        Model name, e.g. "gpt-5".
    max_output_tokens:
        Maximum number of output tokens.

    Returns
    -------
    str
        Extracted text response from the model. If extraction fails, a string
        representation of the raw response object is returned.

    Raises
    ------
    Exception
        Any exception raised by the OpenAI client is propagated to the caller.
    """
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
# Utility and config classes
# ---------------------------------------------------------------------------


@dataclass
class SimulationConfig:
    """
    Configuration parameters for the simulation experiment.

    Notes
    -----
    - `admin_mode` is designed to mirror different API call granularities
      (per_item, per_test, all_tests). In the current implementation, all
      psychometric tests are simulated analytically, so `admin_mode` only
      affects grouping and metadata (e.g., recorded orders), not LLM usage.
    """

    random_seed: int = GLOBAL_RANDOM_SEED
    admin_mode: str = "per_test"
    shuffle_items: bool = False
    shuffle_tests: bool = False
    fixed_order: bool = True
    holdout_fraction: float = 0.2
    output_dir: Optional[str] = None
    bfi_prompt_mode: str = "granular_serapio"
    nfc_prompt_mode: str = "granular_9_level"
    crt_prompt_mode: str = "descriptive"
    test_retest_fraction: float = 0.0
    log_level: str = "INFO"

    def validate(self) -> None:
        """
        Validate configuration values.
        """
        if self.admin_mode not in {"per_item", "per_test", "all_tests"}:
            raise ValueError(
                f"Invalid admin_mode '{self.admin_mode}'. "
                f"Use one of 'per_item', 'per_test', 'all_tests'."
            )

        if not (0.0 < self.holdout_fraction < 1.0):
            raise ValueError(
                "holdout_fraction must be in the open interval (0, 1); "
                f"got {self.holdout_fraction}."
            )

        if not (0.0 <= self.test_retest_fraction <= 1.0):
            raise ValueError(
                "test_retest_fraction must be in [0, 1]; "
                f"got {self.test_retest_fraction}."
            )

        if self.bfi_prompt_mode not in {
            "coarse_numeric",
            "coarse_descriptive",
            "granular_serapio",
        }:
            raise ValueError(
                "Invalid bfi_prompt_mode. Use one of "
                "'coarse_numeric', 'coarse_descriptive', 'granular_serapio'."
            )

        if self.nfc_prompt_mode != "granular_9_level":
            raise ValueError(
                "Invalid nfc_prompt_mode. Currently only 'granular_9_level' "
                "is supported."
            )

        if self.crt_prompt_mode not in {"numeric", "descriptive"}:
            raise ValueError(
                "Invalid crt_prompt_mode. Use 'numeric' or 'descriptive'."
            )

        if self.log_level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError(
                f"Invalid log_level '{self.log_level}'. "
                "Use one of 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'."
            )


@dataclass
class PsychometricItem:
    """
    Representation of a single psychometric test item.
    """

    item_id: str
    text: str
    trait: str
    reverse_scored: bool = False
    correct_answer: Optional[str] = None


@dataclass
class PsychometricTest:
    """
    Representation of a psychometric test (e.g., BFI-brief, NFC-18, CRT2, bCRT).
    """

    name: str
    items: List[PsychometricItem]
    scale_type: str
    instrument_id: Optional[str] = None


@dataclass
class DataBundle:
    """
    Container for all data loaded from disk or constructed in memory.
    """

    personas_df: pd.DataFrame
    markers_df: pd.DataFrame
    tests: Dict[str, PsychometricTest]
    writing_tasks: List[Dict[str, Any]]


@dataclass
class Persona:
    """
    In-memory representation of a single synthetic persona.
    """

    persona_id: str
    age: int
    gender: str
    bfi_extraversion: float
    bfi_agreeableness: float
    bfi_conscientiousness: float
    bfi_neuroticism: float
    bfi_openness: float
    nfc_score: float
    crt2_level: int
    extra_attributes: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_series(row: pd.Series) -> "Persona":
        """
        Create a Persona instance from a pandas Series (row).
        """
        required = [
            "persona_id",
            "age",
            "gender",
            "bfi_extraversion",
            "bfi_agreeableness",
            "bfi_conscientiousness",
            "bfi_neuroticism",
            "bfi_openness",
            "nfc_score",
            "crt2_level",
        ]
        for col in required:
            if col not in row:
                raise KeyError(
                    f"Missing required persona column '{col}' in personas "
                    f"DataFrame row: available columns={list(row.index)}"
                )

        extra = {k: v for k, v in row.items() if k not in required}

        return Persona(
            persona_id=str(row["persona_id"]),
            age=int(row["age"]),
            gender=str(row["gender"]),
            bfi_extraversion=float(row["bfi_extraversion"]),
            bfi_agreeableness=float(row["bfi_agreeableness"]),
            bfi_conscientiousness=float(row["bfi_conscientiousness"]),
            bfi_neuroticism=float(row["bfi_neuroticism"]),
            bfi_openness=float(row["bfi_openness"]),
            nfc_score=float(row["nfc_score"]),
            crt2_level=int(row["crt2_level"]),
            extra_attributes=extra,
        )


@dataclass
class SocialNetwork:
    """
    Simple multilayer social network over personas.

    Notes
    -----
    The default construction creates a ring in the "friendship" layer and a
    fully connected directed graph in the "information" layer. For very large
    persona sets this may be memory intensive; consider adapting the topology
    if scalability becomes an issue.
    """

    nodes: List[str]
    edges_by_layer: Dict[str, Dict[str, List[str]]] = field(default_factory=dict)

    def add_layer(self, layer_name: str) -> None:
        if layer_name not in self.edges_by_layer:
            self.edges_by_layer[layer_name] = {node: [] for node in self.nodes}

    def add_edge(self, layer_name: str, source: str, target: str) -> None:
        if layer_name not in self.edges_by_layer:
            raise KeyError(f"Layer '{layer_name}' does not exist in the network.")
        if source not in self.edges_by_layer[layer_name]:
            raise KeyError(f"Source '{source}' not present in layer '{layer_name}'.")
        if target not in self.edges_by_layer[layer_name]:
            raise KeyError(f"Target '{target}' not present in layer '{layer_name}'.")
        self.edges_by_layer[layer_name][source].append(target)


@dataclass
class ExogenousSignal:
    """
    Simple representation of an exogenous time-varying signal that can
    influence agent behaviour.
    """

    name: str
    values_by_time: Dict[int, float]

    def get(self, t: int) -> float:
        if not self.values_by_time:
            return 0.0
        if t in self.values_by_time:
            return self.values_by_time[t]
        valid_times = [time for time in self.values_by_time if time <= t]
        if not valid_times:
            return 0.0
        last_time = max(valid_times)
        return self.values_by_time[last_time]


@dataclass
class SimulationState:
    """
    Bundles together agents, network, exogenous signals, and raw data.
    """

    personas: Dict[str, Persona]
    network: SocialNetwork
    exogenous_signals: Dict[str, ExogenousSignal]
    data_bundle: DataBundle
    config: SimulationConfig


@dataclass
class CalibrationParameters:
    """
    Parameters learned (or set) during calibration.
    """

    crt2_intercept: float = -1.0
    crt2_slope: float = 0.8
    bfi_response_sd: float = 0.5
    nfc_response_sd: float = 0.5
    # Optional per-item difficulty offsets (logit scale)
    crt2_item_deltas: Dict[str, float] = field(default_factory=dict)
    bcrt_item_deltas: Dict[str, float] = field(default_factory=dict)


class Calibrator:
    """
    Pluggable calibration algorithm for the simulator.
    """

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.params = CalibrationParameters()

    def fit(self, state: SimulationState, train_ids: Iterable[str]) -> None:
        """
        Fit calibration parameters using the training subset of personas.
        """
        df = state.data_bundle.personas_df
        df = df[df["persona_id"].isin(list(train_ids))]

        if df.empty:
            raise ValueError(
                "Training set is empty after applying train_ids. Ensure that "
                "holdout_fraction is configured correctly and that persona_id "
                "values match between the state and DataFrame."
            )

        # --- Calibrate CRT2 logistic parameters ---
        levels = df["crt2_level"].astype(float).values
        p_targets = np.clip(levels / 4.0, 1e-3, 1.0 - 1e-3)
        log_odds = np.log(p_targets / (1.0 - p_targets))

        X = np.vstack([np.ones_like(levels), levels]).T
        beta, _, _, _ = np.linalg.lstsq(X, log_odds, rcond=None)
        intercept, slope = float(beta[0]), float(beta[1])

        self.params.crt2_intercept = intercept
        self.params.crt2_slope = slope

        # --- Calibrate noise scales for BFI and NFC ---
        bfi_cols = [
            "bfi_extraversion",
            "bfi_agreeableness",
            "bfi_conscientiousness",
            "bfi_neuroticism",
            "bfi_openness",
        ]
        bfi_std = float(df[bfi_cols].stack().std(ddof=1))
        self.params.bfi_response_sd = max(0.2, bfi_std / 4.0)

        nfc_std = float(df["nfc_score"].std(ddof=1))
        self.params.nfc_response_sd = max(0.2, nfc_std / 4.0)

        # Initialize per-item difficulty offsets to zero (can be extended later)
        tests = state.data_bundle.tests
        if TEST_CRT2 in tests:
            self.params.crt2_item_deltas = {
                item.item_id: 0.0 for item in tests[TEST_CRT2].items
            }
        if TEST_BCRT in tests:
            self.params.bcrt_item_deltas = {
                item.item_id: 0.0 for item in tests[TEST_BCRT].items
            }

        logging.info(
            "Calibration complete. CRT2 intercept=%.3f, slope=%.3f, "
            "BFI sd=%.3f, NFC sd=%.3f",
            self.params.crt2_intercept,
            self.params.crt2_slope,
            self.params.bfi_response_sd,
            self.params.nfc_response_sd,
        )


@dataclass
class SimulationResult:
    """
    Container for the simulated outputs for a single persona.
    """

    persona_id: str
    persona_prompt: str
    test_item_responses: Dict[str, Dict[str, Any]]
    test_scores: Dict[str, Dict[str, float]]
    writing_features: Dict[str, Any]
    writing_text: str
    retest_scores: Optional[Dict[str, Dict[str, float]]] = None
    admin_metadata: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Prompt construction logic
# ---------------------------------------------------------------------------


class PersonaPromptBuilder:
    """
    Build persona prompts based on BFI, NFC, and CRT2 specifications.
    """

    QUALIFIER_SCALE = [
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

    CRT_LEVELS = {
        0: {
            "label": "very low reflection",
            "sentence": (
                "I almost always trust my first impression, answer quickly "
                "without re-checking, and rarely notice when a question "
                "might be tricky."
            ),
        },
        1: {
            "label": "low reflection",
            "sentence": (
                "I often go with my first impression and only occasionally "
                "stop to reconsider whether it might be misleading."
            ),
        },
        2: {
            "label": "mixed reflection",
            "sentence": (
                "I sometimes pause to reconsider my first impression before "
                "answering, but I am inconsistent and often stick with the "
                "obvious answer."
            ),
        },
        3: {
            "label": "high reflection",
            "sentence": (
                "I usually pause to check whether an obvious answer could "
                "be a trap, and I am willing to change my mind after "
                "thinking things through."
            ),
        },
        4: {
            "label": "very high reflection",
            "sentence": (
                "I almost always look for hidden assumptions, carefully check "
                "for tricks, and verify my answers with calculations before "
                "responding."
            ),
        },
    }

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

    def __init__(self, markers_df: pd.DataFrame, config: SimulationConfig):
        """
        Initialize the builder with adjective markers and configuration.
        """
        self.markers_df = markers_df
        self.config = config
        # Pre-index markers for efficiency and determinism
        self._markers_index: Dict[Tuple[str, str], List[str]] = {}
        for _, row in markers_df.iterrows():
            trait = str(row["trait"]).lower()
            pole = str(row["pole"]).lower()
            adj = str(row["adjective"])
            key = (trait, pole)
            self._markers_index.setdefault(key, []).append(adj)
        for key in self._markers_index:
            self._markers_index[key] = sorted(self._markers_index[key])

    @staticmethod
    def _map_bfi_score_to_bin(score: float) -> str:
        if score <= 2.5:
            return "low"
        if score >= 3.5:
            return "high"
        return "neutral"

    @staticmethod
    def _continuous_to_9_level(score: float) -> int:
        level = 2 * (score - 1.0) + 1.0
        return int(min(9, max(1, round(level))))

    def _sample_markers(self, trait: str, pole: str, n: int = 2) -> List[str]:
        """
        Sample adjective markers for a given trait and pole using a
        deterministic RNG seeded from config.random_seed and a stable hash.
        """
        key = (trait.lower(), pole.lower())
        adjectives = self._markers_index.get(key, [])
        if not adjectives:
            return []
        k = min(n, len(adjectives))
        digest = hashlib.md5(f"{trait.lower()}_{pole.lower()}".encode("utf-8")).hexdigest()
        offset = int(digest, 16) % (2**32)
        rng = random.Random(self.config.random_seed + offset)
        shuffled = list(adjectives)
        rng.shuffle(shuffled)
        return shuffled[:k]

    def _build_bfi_fragment(self, persona: Persona) -> str:
        mode = self.config.bfi_prompt_mode
        parts: List[str] = []

        trait_map = {
            "extraversion": persona.bfi_extraversion,
            "agreeableness": persona.bfi_agreeableness,
            "conscientiousness": persona.bfi_conscientiousness,
            "neuroticism": persona.bfi_neuroticism,
            "openness": persona.bfi_openness,
        }

        if mode == "coarse_numeric":
            for trait_name, score in trait_map.items():
                bin_label = self._map_bfi_score_to_bin(score)
                if bin_label == "neutral":
                    polarity_desc = "moderate"
                else:
                    polarity_desc = bin_label
                parts.append(
                    f"You are a person with {polarity_desc} {trait_name}."
                )

        elif mode == "coarse_descriptive":
            for trait_name, score in trait_map.items():
                bin_label = self._map_bfi_score_to_bin(score)
                if bin_label == "neutral":
                    low_markers = self._sample_markers(trait_name, "low", n=2)
                    high_markers = self._sample_markers(trait_name, "high", n=2)
                    desc_low = ", ".join(low_markers) if low_markers else "reserved"
                    desc_high = ", ".join(high_markers) if high_markers else "outgoing"
                    parts.append(
                        "You are a person who is neither "
                        f"{desc_low} nor {desc_high} in terms of "
                        f"{trait_name}."
                    )
                else:
                    pole = "high" if bin_label == "high" else "low"
                    markers = self._sample_markers(trait_name, pole, n=3)
                    if markers:
                        descriptor_list = ", ".join(markers)
                    else:
                        descriptor_list = (
                            "very " + ("outgoing" if pole == "high" else "reserved")
                        )
                    parts.append(
                        f"You are a person who is {descriptor_list}."
                    )

        elif mode == "granular_serapio":
            domain_fragments: List[str] = []
            for trait_name, score in trait_map.items():
                level = self._continuous_to_9_level(score)
                qualifier_template = self.QUALIFIER_SCALE[level - 1]

                if level <= 4:
                    pole = "low"
                elif level >= 6:
                    pole = "high"
                else:
                    pole = "neutral"

                if pole == "neutral":
                    low_markers = self._sample_markers(trait_name, "low", n=1)
                    high_markers = self._sample_markers(trait_name, "high", n=1)
                    low_adj = low_markers[0] if low_markers else "reserved"
                    high_adj = high_markers[0] if high_markers else "outgoing"
                    fragment = qualifier_template.format(
                        low_adjective=low_adj,
                        high_adjective=high_adj,
                    )
                else:
                    markers = self._sample_markers(trait_name, pole, n=3)
                    base_adj = ", ".join(markers) if markers else pole
                    if pole == "low":
                        fragment = qualifier_template.format(
                            low_adjective=base_adj,
                            high_adjective="",
                        )
                    else:
                        fragment = qualifier_template.format(
                            low_adjective="",
                            high_adjective=base_adj,
                        )
                domain_fragments.append(f"{trait_name}: {fragment}")
            qualified_descriptors = "; ".join(domain_fragments)
            parts.append(
                'For the following tasks, respond as a person described as: '
                f'"I am {qualified_descriptors}."'
            )
        else:
            raise ValueError(f"Unsupported bfi_prompt_mode: {mode}")

        return " ".join(parts)

    def _build_nfc_fragment(self, persona: Persona) -> str:
        level = self._continuous_to_9_level(persona.nfc_score)
        qualifier_template = self.QUALIFIER_SCALE[level - 1]

        if level <= 4:
            base_descriptors = self.NFC_LOW_DESCRIPTORS
            rng = random.Random(self.config.random_seed + 17)
            low_choices = rng.sample(base_descriptors, k=min(3, len(base_descriptors)))
            low_adj = ", ".join(low_choices)
            qualified = qualifier_template.format(
                low_adjective=low_adj,
                high_adjective="",
            )
        elif level >= 6:
            base_descriptors = self.NFC_HIGH_DESCRIPTORS
            rng = random.Random(self.config.random_seed + 23)
            high_choices = rng.sample(
                base_descriptors, k=min(3, len(base_descriptors))
            )
            high_adj = ", ".join(high_choices)
            qualified = qualifier_template.format(
                low_adjective="",
                high_adjective=high_adj,
            )
        else:
            rng = random.Random(self.config.random_seed + 31)
            low_adj = rng.choice(self.NFC_LOW_DESCRIPTORS)
            high_adj = rng.choice(self.NFC_HIGH_DESCRIPTORS)
            qualified = qualifier_template.format(
                low_adjective=low_adj,
                high_adjective=high_adj,
            )

        return (
            'For the following tasks, respond as a person described as: '
            f'"I {qualified}."'
        )

    def _build_crt_fragment(self, persona: Persona) -> str:
        level = int(persona.crt2_level)
        if level not in self.CRT_LEVELS:
            raise ValueError(
                f"crt2_level must be an integer 0–4; got {level} "
                f"for persona_id={persona.persona_id}"
            )
        meta = self.CRT_LEVELS[level]
        numeric_tag = (
            f"This persona has a CRT2 ability level of {level} on a 0–4 scale."
        )

        if self.config.crt_prompt_mode == "numeric":
            return numeric_tag

        sentence = meta["sentence"]
        return (
            f"{numeric_tag} For the following CRT-style questions, respond as "
            f'a person described as: "{sentence}"'
        )

    def build_full_prompt(self, persona: Persona) -> str:
        parts = [
            self._build_bfi_fragment(persona),
            self._build_nfc_fragment(persona),
            self._build_crt_fragment(persona),
        ]
        return " ".join(parts)


# ---------------------------------------------------------------------------
# Memory and planning agents for LLM prompting
# ---------------------------------------------------------------------------


class MemoryAgent:
    """
    Simple memory agent that provides user and item/task context for LLM calls.
    """

    def __init__(self, state: SimulationState):
        self.state = state

    def get_user_context(self, persona: Persona, persona_prompt: str) -> str:
        """
        Construct a concise user context string from persona attributes and
        the persona-level prompt.
        """
        traits = (
            f"BFI traits: extraversion={persona.bfi_extraversion}, "
            f"agreeableness={persona.bfi_agreeableness}, "
            f"conscientiousness={persona.bfi_conscientiousness}, "
            f"neuroticism={persona.bfi_neuroticism}, "
            f"openness={persona.bfi_openness}; "
            f"NFC={persona.nfc_score}; CRT2 level={persona.crt2_level}."
        )
        demo = f"Persona {persona.persona_id}, age {persona.age}, gender {persona.gender}."
        return demo + "\n" + traits + "\nPersona prompt:\n" + persona_prompt

    def get_writing_task_context(self, writing_task: Optional[Dict[str, Any]]) -> str:
        """
        Build item/task context for a writing task from the test bank.
        """
        if writing_task is None:
            return (
                "Writing task: Reflect on a recent decision and describe how "
                "you approached it, including your thinking process."
            )
        task_id = writing_task.get("id") or writing_task.get("task_id") or "writing_task"
        title = (
            writing_task.get("title")
            or writing_task.get("name")
            or "Reflection writing task"
        )
        prompt = (
            writing_task.get("prompt")
            or writing_task.get("instruction")
            or ""
        )
        return f"Writing task [{task_id}] - {title}.\nInstruction: {prompt}"


class PlanningAgent:
    """
    Simple planning agent that produces a high-level plan for responding to
    a writing task.
    """

    def plan_writing(
        self, persona: Persona, writing_task: Optional[Dict[str, Any]]
    ) -> str:
        """
        Produce a short natural-language plan/steps for the writing task.
        """
        steps = [
            "1. Restate the task in your own words.",
            "2. Briefly describe the situation or decision you are reflecting on.",
            "3. Explain how you approached the decision, including any initial reactions.",
            "4. Describe how much you thought about alternative options or possible pitfalls.",
            "5. Conclude with what you learned about your own thinking style.",
        ]
        return "\n".join(steps)


# ---------------------------------------------------------------------------
# Response generation (simulated LLM)
# ---------------------------------------------------------------------------


class ResponseGenerator:
    """
    Simulated LLM response generator based on persona traits and calibrated
    parameters, with optional real LLM calls for writing tasks.
    """

    def __init__(
        self,
        config: SimulationConfig,
        calibrator: Calibrator,
        tests: Dict[str, PsychometricTest],
        exogenous_signals: Dict[str, ExogenousSignal],
        memory_agent: MemoryAgent,
        planning_agent: PlanningAgent,
        writing_tasks: List[Dict[str, Any]],
    ):
        self.config = config
        self.calibrator = calibrator
        self.tests = tests
        self.exogenous_signals = exogenous_signals
        self.memory_agent = memory_agent
        self.planning_agent = planning_agent
        self.writing_tasks = writing_tasks

    @staticmethod
    def _truncate_likert(value: float) -> int:
        return int(min(5, max(1, round(value))))

    def _simulate_bfi(
        self, persona: Persona
    ) -> Tuple[Dict[str, int], Dict[str, float]]:
        test = self.tests[TEST_BFI]

        trait_values = {
            "extraversion": persona.bfi_extraversion,
            "agreeableness": persona.bfi_agreeableness,
            "conscientiousness": persona.bfi_conscientiousness,
            "neuroticism": persona.bfi_neuroticism,
            "openness": persona.bfi_openness,
        }

        responses: Dict[str, int] = {}
        domain_items: Dict[str, List[int]] = {
            "extraversion": [],
            "agreeableness": [],
            "conscientiousness": [],
            "neuroticism": [],
            "openness": [],
        }

        sd = self.calibrator.params.bfi_response_sd

        for item in test.items:
            base_mean = trait_values.get(item.trait, 3.0)
            if item.reverse_scored:
                base_mean = 6.0 - base_mean
            noisy_value = np.random.normal(loc=base_mean, scale=sd)
            likert = self._truncate_likert(noisy_value)
            responses[item.item_id] = likert
            if item.trait in domain_items:
                domain_items[item.trait].append(likert)

        domain_scores: Dict[str, float] = {}
        for trait, values in domain_items.items():
            if values:
                domain_scores[trait] = float(np.mean(values))
            else:
                domain_scores[trait] = float("nan")

        return responses, domain_scores

    def _simulate_nfc(
        self, persona: Persona
    ) -> Tuple[Dict[str, int], Dict[str, float]]:
        test = self.tests[TEST_NFC]
        sd = self.calibrator.params.nfc_response_sd
        base_mean = persona.nfc_score

        responses: Dict[str, int] = {}
        all_values: List[int] = []

        for item in test.items:
            mean_for_item = 6.0 - base_mean if item.reverse_scored else base_mean
            noisy_value = np.random.normal(loc=mean_for_item, scale=sd)
            likert = self._truncate_likert(noisy_value)
            responses[item.item_id] = likert
            all_values.append(likert)

        scores = {
            "nfc_total": float(np.mean(all_values)) if all_values else float("nan")
        }
        return responses, scores

    def _logistic(self, x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))

    def _generate_crt_answer(self, item: PsychometricItem, is_correct: bool) -> str:
        """
        Generate a concrete answer string for a CRT item based on the desired
        correctness flag and the item's correct_answer field.
        """
        correct = (item.correct_answer or "").strip()
        if is_correct or not correct:
            return correct if correct else "CORRECT"
        # Simple placeholder incorrect answer that differs from the correct key.
        # Downstream scoring should be based on equality with item.correct_answer.
        return "INCORRECT"

    def _simulate_crt2(
        self, persona: Persona
    ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, float]]:
        """
        Simulate responses for the CRT2 test.

        For each item, we sample whether the response is correct using a
        logistic model of CRT2 level, then generate an answer string that is
        either equal to the item's correct_answer (if correct) or a generic
        incorrect response. The returned item responses contain both the raw
        answer and an is_correct flag, and summary scores are derived from
        these flags.
        """
        test = self.tests[TEST_CRT2]

        lv = float(persona.crt2_level)
        base_logit = (
            self.calibrator.params.crt2_intercept
            + self.calibrator.params.crt2_slope * lv
        )

        responses: Dict[str, Dict[str, Any]] = {}
        correct_flags: List[int] = []

        for item in test.items:
            delta = self.calibrator.params.crt2_item_deltas.get(item.item_id, 0.0)
            logit_p = base_logit + delta
            p_correct = self._logistic(logit_p)
            is_correct = 1 if random.random() < p_correct else 0
            answer = self._generate_crt_answer(item, bool(is_correct))
            # Ensure internal consistency between answer and correctness flag
            derived_is_correct = int(
                bool(item.correct_answer)
                and str(answer).strip() == str(item.correct_answer).strip()
            )
            # If for some reason they disagree, trust the equality-based result
            is_correct = derived_is_correct
            responses[item.item_id] = {
                "answer": answer,
                "is_correct": is_correct,
            }
            correct_flags.append(is_correct)

        correct_count = int(sum(correct_flags))
        mean_correct = (
            correct_count / len(test.items) if test.items else float("nan")
        )
        scores = {
            "correct_count": float(correct_count),
            "mean_correct": float(mean_correct),
        }
        return responses, scores

    def _simulate_bcrt(
        self, persona: Persona
    ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, float]]:
        """
        Simulate responses for the bCRT test (binary correctness).

        We reuse the calibrated CRT2 logistic mapping as a proxy, optionally
        with per-item difficulty offsets.
        """
        test = self.tests.get(TEST_BCRT)
        if test is None:
            return {}, {}

        lv = float(persona.crt2_level)
        base_logit = (
            self.calibrator.params.crt2_intercept
            + self.calibrator.params.crt2_slope * lv
        )

        responses: Dict[str, Dict[str, Any]] = {}
        correct_flags: List[int] = []

        for item in test.items:
            delta = self.calibrator.params.bcrt_item_deltas.get(item.item_id, 0.0)
            logit_p = base_logit + delta
            p_correct = self._logistic(logit_p)
            is_correct = 1 if random.random() < p_correct else 0
            answer = self._generate_crt_answer(item, bool(is_correct))
            derived_is_correct = int(
                bool(item.correct_answer)
                and str(answer).strip() == str(item.correct_answer).strip()
            )
            is_correct = derived_is_correct
            responses[item.item_id] = {
                "answer": answer,
                "is_correct": is_correct,
            }
            correct_flags.append(is_correct)

        correct_count = int(sum(correct_flags))
        mean_correct = (
            correct_count / len(test.items) if test.items else float("nan")
        )
        scores = {
            "correct_count": float(correct_count),
            "mean_correct": float(mean_correct),
        }
        return responses, scores

    def _simulate_writing_deterministic(
        self, persona: Persona, t: int = 1
    ) -> Tuple[str, Dict[str, float]]:
        """
        Deterministic fallback writing-task generator used when LLM calls
        are not available.
        """
        signal = self.exogenous_signals.get("reflection_salience", None)
        signal_val = signal.get(t) if signal is not None else 0.0

        analytic = (
            40.0
            + 10.0 * persona.nfc_score
            + 5.0 * persona.crt2_level
            + 2.0 * signal_val
        )
        insight = (
            30.0
            + 5.0 * persona.nfc_score
            + 3.0 * persona.bfi_openness
            + 1.5 * signal_val
        )
        affect = (
            50.0 - 5.0 * persona.bfi_neuroticism + 2.0 * persona.bfi_extraversion
        )

        base_length = 80
        length = int(
            base_length
            + 8.0 * persona.bfi_openness
            + 6.0 * persona.bfi_extraversion
            + 2.0 * persona.nfc_score
        )

        clauses = []
        clauses.append(
            "I am reflecting on a recent decision and how I approached it."
        )
        if persona.nfc_score >= 3.5:
            clauses.append(
                "I considered multiple alternatives and enjoyed thinking "
                "through the complexities of the situation."
            )
        else:
            clauses.append(
                "I preferred to settle on a simple option without spending "
                "too much time thinking."
            )
        if persona.crt2_level >= 3:
            clauses.append(
                "I tried to check whether my first impression might be "
                "misleading before committing to an answer."
            )
        else:
            clauses.append(
                "I mostly trusted my initial impression and did not examine "
                "potential hidden assumptions in depth."
            )
        if persona.bfi_openness >= 3.5:
            clauses.append(
                "I am curious about new perspectives and I like to explore "
                "different possibilities in my writing."
            )

        text = " ".join(clauses)
        filler_sentence = (
            " This reflection helps me understand my own thinking style "
            "and how it shapes my choices."
        )
        while len(text.split()) < length:
            text += filler_sentence

        features = {
            "analytic": float(analytic),
            "insight": float(insight),
            "affect": float(affect),
            "length_words": float(len(text.split())),
        }
        return text, features

    def _simulate_writing_via_llm(
        self, persona: Persona, persona_prompt: str, t: int = 1
    ) -> Tuple[str, Dict[str, float]]:
        """
        Simulate a writing task by calling an OpenAI LLM using Memory and
        Planning agents to construct the prompt. Falls back to deterministic
        generation if the LLM call is unavailable or fails.
        """
        # If OpenAI is not available, immediately fall back deterministically
        if not OPENAI_AVAILABLE:
            logging.info(
                "OpenAI package not available; using deterministic writing generator."
            )
            return self._simulate_writing_deterministic(persona, t=t)

        writing_task = self.writing_tasks[0] if self.writing_tasks else None
        user_ctx = self.memory_agent.get_user_context(persona, persona_prompt)
        item_ctx = self.memory_agent.get_writing_task_context(writing_task)
        plan = self.planning_agent.plan_writing(persona, writing_task)

        prompt = (
            "You are an AI simulating a human persona.\n\n"
            "USER CONTEXT:\n"
            f"{user_ctx}\n\n"
            "TASK CONTEXT:\n"
            f"{item_ctx}\n\n"
            "PLAN / STEPS TO FOLLOW:\n"
            f"{plan}\n\n"
            "Now write the full response in the voice of this persona. "
            "Do not mention that you are an AI or that this is a simulation.\n"
        )

        try:
            text = call_gpt5_with_responses_api(
                prompt=prompt, model="gpt-5", max_output_tokens=2000
            )
            deterministic_text, features = self._simulate_writing_deterministic(
                persona, t=t
            )
            # Replace text with LLM output, keep deterministic features except
            # for length, which is recomputed from the LLM-generated text.
            _ = deterministic_text  # unused, kept for clarity
            features["length_words"] = float(len(text.split()))
            return text, features
        except Exception as exc:
            logging.warning(
                "LLM writing call failed (%s). Falling back to deterministic writing.",
                exc,
            )
            return self._simulate_writing_deterministic(persona, t=t)

    def simulate_all_tests(
        self, persona: Persona, persona_prompt: str
    ) -> Tuple[
        Dict[str, Dict[str, Any]],
        Dict[str, Dict[str, float]],
        str,
        Dict[str, float],
    ]:
        """
        Simulate all psychometric tests and writing tasks for a persona.
        """
        bfi_items, bfi_scores = self._simulate_bfi(persona)
        nfc_items, nfc_scores = self._simulate_nfc(persona)
        crt_items, crt_scores = self._simulate_crt2(persona)
        bcrt_items, bcrt_scores = self._simulate_bcrt(persona)
        writing_text, writing_features = self._simulate_writing_via_llm(
            persona, persona_prompt, t=1
        )

        item_responses: Dict[str, Dict[str, Any]] = {
            TEST_BFI: bfi_items,
            TEST_NFC: nfc_items,
            TEST_CRT2: crt_items,
        }
        if bcrt_items:
            item_responses[TEST_BCRT] = bcrt_items

        test_scores: Dict[str, Dict[str, float]] = {
            TEST_BFI: bfi_scores,
            TEST_NFC: nfc_scores,
            TEST_CRT2: crt_scores,
        }
        if bcrt_scores:
            test_scores[TEST_BCRT] = bcrt_scores

        return item_responses, test_scores, writing_text, writing_features


# ---------------------------------------------------------------------------
# Test administration abstraction
# ---------------------------------------------------------------------------


class TestAdministrator:
    """
    Abstraction for administering psychometric tests to personas under
    different "API call granularity" modes, and with configurable
    item/test shuffling for order-effects studies.

    Notes
    -----
    In this implementation, all psychometric tests are simulated using
    deterministic/statistical models. The `admin_mode` branches therefore
    correspond to different groupings and metadata about ordering, rather
    than to actual differences in LLM API usage for test items.
    """

    def __init__(
        self,
        config: SimulationConfig,
        response_generator: ResponseGenerator,
    ):
        self.config = config
        self.generator = response_generator

    def _get_test_order(self) -> List[str]:
        order = [TEST_BFI, TEST_NFC, TEST_CRT2, TEST_BCRT]
        order = [t for t in order if t in self.generator.tests]
        if self.config.fixed_order or not self.config.shuffle_tests:
            return order
        rng = random.Random(self.config.random_seed + 101)
        shuffled = list(order)
        rng.shuffle(shuffled)
        return shuffled

    def _get_item_order(self, test_name: str) -> List[str]:
        test = self.generator.tests[test_name]
        item_ids = [it.item_id for it in test.items]
        if self.config.fixed_order or not self.config.shuffle_items:
            return item_ids
        digest = hashlib.md5(test_name.encode("utf-8")).hexdigest()
        offset = int(digest, 16) % (2**32)
        rng = random.Random(self.config.random_seed + offset)
        shuffled = list(item_ids)
        rng.shuffle(shuffled)
        return shuffled

    def administer(self, persona: Persona, persona_prompt: str) -> SimulationResult:
        """
        Administer all configured tests and writing tasks to a persona.
        """
        admin_metadata: Dict[str, Any] = {}

        test_item_responses: Dict[str, Dict[str, Any]] = {}
        test_scores: Dict[str, Dict[str, float]] = {}

        test_order = self._get_test_order()
        admin_metadata["test_order"] = test_order
        item_orders: Dict[str, List[str]] = {}
        for tname in test_order:
            item_orders[tname] = self._get_item_order(tname)
        admin_metadata["item_order"] = item_orders

        if self.config.admin_mode == "all_tests":
            item_responses, scores, writing_text, writing_features = (
                self.generator.simulate_all_tests(persona, persona_prompt)
            )
            test_item_responses = item_responses
            test_scores = scores

        elif self.config.admin_mode == "per_test":
            if TEST_BFI in test_order:
                bfi_items, bfi_scores = self.generator._simulate_bfi(persona)
                test_item_responses[TEST_BFI] = bfi_items
                test_scores[TEST_BFI] = bfi_scores
            if TEST_NFC in test_order:
                nfc_items, nfc_scores = self.generator._simulate_nfc(persona)
                test_item_responses[TEST_NFC] = nfc_items
                test_scores[TEST_NFC] = nfc_scores
            if TEST_CRT2 in test_order:
                crt_items, crt_scores = self.generator._simulate_crt2(persona)
                test_item_responses[TEST_CRT2] = crt_items
                test_scores[TEST_CRT2] = crt_scores
            if TEST_BCRT in test_order:
                bcrt_items, bcrt_scores = self.generator._simulate_bcrt(persona)
                if bcrt_items:
                    test_item_responses[TEST_BCRT] = bcrt_items
                if bcrt_scores:
                    test_scores[TEST_BCRT] = bcrt_scores
            writing_text, writing_features = self.generator._simulate_writing_via_llm(
                persona, persona_prompt, t=1
            )

        else:  # per_item
            if TEST_BFI in test_order:
                bfi_items, bfi_scores = self.generator._simulate_bfi(persona)
                test_item_responses[TEST_BFI] = bfi_items
                test_scores[TEST_BFI] = bfi_scores
            if TEST_NFC in test_order:
                nfc_items, nfc_scores = self.generator._simulate_nfc(persona)
                test_item_responses[TEST_NFC] = nfc_items
                test_scores[TEST_NFC] = nfc_scores
            if TEST_CRT2 in test_order:
                crt_items, crt_scores = self.generator._simulate_crt2(persona)
                test_item_responses[TEST_CRT2] = crt_items
                test_scores[TEST_CRT2] = crt_scores
            if TEST_BCRT in test_order:
                bcrt_items, bcrt_scores = self.generator._simulate_bcrt(persona)
                if bcrt_items:
                    test_item_responses[TEST_BCRT] = bcrt_items
                if bcrt_scores:
                    test_scores[TEST_BCRT] = bcrt_scores
            writing_text, writing_features = self.generator._simulate_writing_via_llm(
                persona, persona_prompt, t=1
            )

        retest_scores: Optional[Dict[str, Dict[str, float]]] = None
        if random.random() < self.config.test_retest_fraction:
            _, retest_scores, _, _ = self.generator.simulate_all_tests(
                persona, persona_prompt
            )

        return SimulationResult(
            persona_id=persona.persona_id,
            persona_prompt=persona_prompt,
            test_item_responses=test_item_responses,
            test_scores=test_scores,
            writing_features=writing_features,
            writing_text=writing_text,
            retest_scores=retest_scores,
            admin_metadata=admin_metadata,
        )


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------


class Simulator:
    """
    Forward simulator that rolls out test administrations for a set of
    personas over a (small) time horizon.
    """

    def __init__(
        self,
        state: SimulationState,
        calibrator: Calibrator,
    ):
        self.state = state
        self.calibrator = calibrator

        self.prompt_builder = PersonaPromptBuilder(
            markers_df=state.data_bundle.markers_df,
            config=state.config,
        )
        self.memory_agent = MemoryAgent(state)
        self.planning_agent = PlanningAgent()
        self.response_generator = ResponseGenerator(
            config=state.config,
            calibrator=self.calibrator,
            tests=state.data_bundle.tests,
            exogenous_signals=state.exogenous_signals,
            memory_agent=self.memory_agent,
            planning_agent=self.planning_agent,
            writing_tasks=state.data_bundle.writing_tasks,
        )
        self.administrator = TestAdministrator(
            config=state.config,
            response_generator=self.response_generator,
        )

    def rollout(self, persona_ids: Iterable[str]) -> List[SimulationResult]:
        results: List[SimulationResult] = []
        for pid in persona_ids:
            if pid not in self.state.personas:
                raise KeyError(
                    f"Persona ID '{pid}' is not present in SimulationState.personas."
                )
            persona = self.state.personas[pid]
            persona_prompt = self.prompt_builder.build_full_prompt(persona)
            sim_result = self.administrator.administer(persona, persona_prompt)
            results.append(sim_result)
        return results


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------


class Evaluator:
    """
    Compute evaluation metrics for the simulation.
    """

    def __init__(self, config: SimulationConfig, tests: Dict[str, PsychometricTest]):
        self.config = config
        self.tests = tests

    @staticmethod
    def _pearson_spearman(x: List[float], y: List[float]) -> Tuple[float, float]:
        if len(x) == 0 or len(x) != len(y):
            return float("nan"), float("nan")
        s1 = pd.Series(x)
        s2 = pd.Series(y)
        pearson = float(s1.corr(s2, method="pearson"))
        spearman = float(s1.corr(s2, method="spearman"))
        return pearson, spearman

    @staticmethod
    def _mae(x: List[float], y: List[float]) -> float:
        if len(x) == 0 or len(x) != len(y):
            return float("nan")
        arr_x = np.asarray(x, dtype=float)
        arr_y = np.asarray(y, dtype=float)
        return float(np.mean(np.abs(arr_x - arr_y)))

    @staticmethod
    def _cronbach_alpha(matrix: np.ndarray) -> float:
        """
        Compute Cronbach's alpha for a 2D matrix of item responses, with
        basic handling of missing data (NaNs).

        - Items with >30% missingness are dropped.
        - Remaining missing entries are imputed with item means.
        """
        if matrix.ndim != 2:
            raise ValueError(
                "matrix must be 2D (n_personas, n_items) to compute "
                "Cronbach's alpha."
            )

        n_personas, n_items = matrix.shape
        if n_items < 2 or n_personas < 2:
            return float("nan")

        mat = np.array(matrix, dtype=float)

        missing_mask = np.isnan(mat)
        missing_frac = missing_mask.sum(axis=0) / float(n_personas)
        keep_mask = missing_frac <= 0.3
        if keep_mask.sum() < 2:
            return float("nan")
        mat = mat[:, keep_mask]

        col_means = np.nanmean(mat, axis=0)
        inds = np.where(np.isnan(mat))
        mat[inds] = np.take(col_means, inds[1])

        n_personas, n_items = mat.shape
        item_vars = mat.var(axis=0, ddof=1)
        total_scores = mat.sum(axis=1)
        total_var = total_scores.var(ddof=1)

        if total_var <= 0:
            return float("nan")

        alpha = (n_items / (n_items - 1.0)) * (
            1.0 - float(item_vars.sum()) / float(total_var)
        )
        return float(alpha)

    @staticmethod
    def _monotonicity_summary(
        real_vals: List[float],
        sim_vals: List[float],
        n_bins: int = 5,
    ) -> Dict[str, Any]:
        """
        Compute simple monotonicity diagnostics by binning real_vals into
        quantiles and computing mean simulated scores per bin.
        """
        if len(real_vals) == 0 or len(real_vals) != len(sim_vals):
            return {
                "bin_means": [],
                "violations": None,
                "monotonicity_ratio": float("nan"),
            }

        df = pd.DataFrame({"real": real_vals, "sim": sim_vals})
        if df["real"].nunique() < 2:
            return {
                "bin_means": [],
                "violations": None,
                "monotonicity_ratio": float("nan"),
            }
        try:
            df["bin"] = pd.qcut(
                df["real"],
                q=min(n_bins, df["real"].nunique()),
                duplicates="drop",
            )
        except ValueError:
            return {
                "bin_means": [],
                "violations": None,
                "monotonicity_ratio": float("nan"),
            }
        grouped = df.groupby("bin")["sim"].mean()
        bin_means = grouped.tolist()
        if len(bin_means) < 2:
            return {
                "bin_means": bin_means,
                "violations": 0,
                "monotonicity_ratio": 1.0,
            }
        diffs = np.diff(bin_means)
        violations = int((diffs < 0).sum())
        monotonicity_ratio = float((diffs >= 0).sum() / diffs.size)
        return {
            "bin_means": bin_means,
            "violations": violations,
            "monotonicity_ratio": monotonicity_ratio,
        }

    def compute_metrics(
        self,
        sim_results: List[SimulationResult],
        state: SimulationState,
        val_ids: Iterable[str],
    ) -> Dict[str, Any]:
        df = state.data_bundle.personas_df
        val_ids_list = list(val_ids)
        df_val = df[df["persona_id"].isin(val_ids_list)].copy()
        df_val.set_index("persona_id", inplace=True)

        sim_by_id: Dict[str, SimulationResult] = {
            r.persona_id: r for r in sim_results
        }

        trait_metrics: Dict[str, Any] = {}

        bfi_domains = [
            "extraversion",
            "agreeableness",
            "conscientiousness",
            "neuroticism",
            "openness",
        ]
        bfi_corrs: Dict[str, Any] = {}
        for domain in bfi_domains:
            real_col = f"bfi_{domain}"
            if real_col not in df_val.columns:
                continue
            real_vals: List[float] = []
            sim_vals: List[float] = []
            for pid in val_ids_list:
                if pid not in df_val.index or pid not in sim_by_id:
                    continue
                real_vals.append(float(df_val.loc[pid, real_col]))
                sim_score = sim_by_id[pid].test_scores.get(TEST_BFI, {}).get(
                    domain, float("nan")
                )
                sim_vals.append(float(sim_score))

            pearson, spearman = self._pearson_spearman(real_vals, sim_vals)
            monot = self._monotonicity_summary(real_vals, sim_vals)
            bfi_corrs[domain] = {
                "pearson": pearson,
                "spearman": spearman,
                "monotonicity": monot,
            }

        trait_metrics["bfi_correlations"] = bfi_corrs

        real_nfc: List[float] = []
        sim_nfc: List[float] = []
        for pid in val_ids_list:
            if pid not in df_val.index or pid not in sim_by_id:
                continue
            real_nfc.append(float(df_val.loc[pid, "nfc_score"]))
            sim_nfc.append(
                float(
                    sim_by_id[pid].test_scores.get(TEST_NFC, {}).get(
                        "nfc_total", float("nan")
                    )
                )
            )
        nfc_pearson, nfc_spearman = self._pearson_spearman(real_nfc, sim_nfc)
        nfc_monot = self._monotonicity_summary(real_nfc, sim_nfc)
        trait_metrics["nfc_correlations"] = {
            "pearson": nfc_pearson,
            "spearman": nfc_spearman,
            "monotonicity": nfc_monot,
        }

        real_crt: List[float] = []
        sim_crt: List[float] = []
        for pid in val_ids_list:
            if pid not in df_val.index or pid not in sim_by_id:
                continue
            real_crt.append(float(df_val.loc[pid, "crt2_level"]))
            sim_crt.append(
                float(
                    sim_by_id[pid].test_scores.get(TEST_CRT2, {}).get(
                        "correct_count", float("nan")
                    )
                )
            )
        crt_pearson, crt_spearman = self._pearson_spearman(real_crt, sim_crt)
        crt_mae = self._mae(real_crt, sim_crt)
        crt_monot = self._monotonicity_summary(real_crt, sim_crt)
        trait_metrics["crt2_correlations"] = {
            "pearson": crt_pearson,
            "spearman": crt_spearman,
            "mae": crt_mae,
            "monotonicity": crt_monot,
        }

        if TEST_BCRT in self.tests:
            sim_bcrt: List[float] = []
            for pid in val_ids_list:
                if pid not in df_val.index or pid not in sim_by_id:
                    continue
                sim_bcrt.append(
                    float(
                        sim_by_id[pid].test_scores.get(TEST_BCRT, {}).get(
                            "correct_count", float("nan")
                        )
                    )
                )
            bcrt_pearson, bcrt_spearman = self._pearson_spearman(
                real_crt, sim_bcrt
            )
            bcrt_mae = self._mae(real_crt, sim_bcrt)
            bcrt_monot = self._monotonicity_summary(real_crt, sim_bcrt)
            trait_metrics["bcrt_correlations"] = {
                "pearson": bcrt_pearson,
                "spearman": bcrt_spearman,
                "mae": bcrt_mae,
                "monotonicity": bcrt_monot,
            }

        behav_metrics: Dict[str, Any] = {}
        features = ["analytic", "insight", "affect"]
        for feat in features:
            y: List[float] = []
            x_nfc: List[float] = []
            x_crt: List[float] = []
            for pid in val_ids_list:
                if pid not in df_val.index or pid not in sim_by_id:
                    continue
                feat_val = sim_by_id[pid].writing_features.get(
                    feat, float("nan")
                )
                if math.isnan(float(feat_val)):
                    continue
                y.append(float(feat_val))
                x_nfc.append(float(df_val.loc[pid, "nfc_score"]))
                x_crt.append(float(df_val.loc[pid, "crt2_level"]))
            if len(y) < 3:
                behav_metrics[feat] = {
                    "coeffs": None,
                    "r2": float("nan"),
                }
                continue

            X = np.vstack(
                [np.ones(len(y)), np.asarray(x_nfc), np.asarray(x_crt)]
            ).T
            y_arr = np.asarray(y)
            beta, _, _, _ = np.linalg.lstsq(X, y_arr, rcond=None)
            y_pred = X @ beta
            ss_res = float(((y_arr - y_pred) ** 2).sum())
            ss_tot = float(((y_arr - y_arr.mean()) ** 2).sum())
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

            behav_metrics[feat] = {
                "coeffs": {
                    "intercept": float(beta[0]),
                    "nfc": float(beta[1]),
                    "crt2": float(beta[2]),
                },
                "r2": float(r2),
            }

        lengths = [
            float(
                sim_by_id[pid].writing_features.get(
                    "length_words", float("nan")
                )
            )
            for pid in val_ids_list
            if pid in sim_by_id
        ]
        lengths_arr = np.asarray(lengths, dtype=float)
        behav_metrics["length_summary"] = {
            "mean": float(np.nanmean(lengths_arr))
            if lengths_arr.size > 0
            else float("nan"),
            "std": float(np.nanstd(lengths_arr, ddof=1))
            if lengths_arr.size > 1
            else float("nan"),
        }
        behav_metrics["prompt_mode"] = state.config.bfi_prompt_mode

        resp_metrics: Dict[str, Any] = {}

        bfi_items_all: List[List[float]] = []
        nfc_items_all: List[List[float]] = []

        bfi_test = self.tests.get(TEST_BFI)
        nfc_test = self.tests.get(TEST_NFC)
        bfi_item_ids: List[str] = (
            [it.item_id for it in bfi_test.items] if bfi_test else []
        )
        nfc_item_ids: List[str] = (
            [it.item_id for it in nfc_test.items] if nfc_test else []
        )

        bfi_pairs: List[Tuple[float, float]] = []
        nfc_pairs: List[Tuple[float, float]] = []

        for pid in val_ids_list:
            if pid not in sim_by_id:
                continue
            res = sim_by_id[pid]

            if TEST_BFI in res.test_item_responses and bfi_item_ids:
                row_vals = [
                    float(res.test_item_responses[TEST_BFI].get(iid, np.nan))
                    for iid in bfi_item_ids
                ]
                bfi_items_all.append(row_vals)

            if TEST_NFC in res.test_item_responses and nfc_item_ids:
                row_vals = [
                    float(res.test_item_responses[TEST_NFC].get(iid, np.nan))
                    for iid in nfc_item_ids
                ]
                nfc_items_all.append(row_vals)

            if res.retest_scores is not None:
                bfi_t1 = float(
                    res.test_scores.get(TEST_BFI, {}).get(
                        "extraversion", float("nan")
                    )
                )
                bfi_t2 = float(
                    res.retest_scores.get(TEST_BFI, {}).get(
                        "extraversion", float("nan")
                    )
                )
                if not (math.isnan(bfi_t1) or math.isnan(bfi_t2)):
                    bfi_pairs.append((bfi_t1, bfi_t2))

                nfc_t1 = float(
                    res.test_scores.get(TEST_NFC, {}).get(
                        "nfc_total", float("nan")
                    )
                )
                nfc_t2 = float(
                    res.retest_scores.get(TEST_NFC, {}).get(
                        "nfc_total", float("nan")
                    )
                )
                if not (math.isnan(nfc_t1) or math.isnan(nfc_t2)):
                    nfc_pairs.append((nfc_t1, nfc_t2))

        if bfi_items_all:
            bfi_matrix = np.asarray(bfi_items_all, dtype=float)
            resp_metrics["cronbach_alpha_bfi"] = self._cronbach_alpha(
                bfi_matrix
            )
            resp_metrics["cronbach_alpha_bfi_n"] = int(bfi_matrix.shape[0])
        else:
            resp_metrics["cronbach_alpha_bfi"] = float("nan")
            resp_metrics["cronbach_alpha_bfi_n"] = 0

        if nfc_items_all:
            nfc_matrix = np.asarray(nfc_items_all, dtype=float)
            resp_metrics["cronbach_alpha_nfc"] = self._cronbach_alpha(
                nfc_matrix
            )
            resp_metrics["cronbach_alpha_nfc_n"] = int(nfc_matrix.shape[0])
        else:
            resp_metrics["cronbach_alpha_nfc"] = float("nan")
            resp_metrics["cronbach_alpha_nfc_n"] = 0

        if len(bfi_pairs) >= 3:
            bfi_t1 = [p[0] for p in bfi_pairs]
            bfi_t2 = [p[1] for p in bfi_pairs]
            pearson, spearman = self._pearson_spearman(bfi_t1, bfi_t2)
            resp_metrics["test_retest_bfi"] = {
                "pearson": pearson,
                "spearman": spearman,
                "n": len(bfi_pairs),
            }
        else:
            resp_metrics["test_retest_bfi"] = None

        if len(nfc_pairs) >= 3:
            nfc_t1 = [p[0] for p in nfc_pairs]
            nfc_t2 = [p[1] for p in nfc_pairs]
            pearson, spearman = self._pearson_spearman(nfc_t1, nfc_t2)
            resp_metrics["test_retest_nfc"] = {
                "pearson": pearson,
                "spearman": spearman,
                "n": len(nfc_pairs),
            }
        else:
            resp_metrics["test_retest_nfc"] = None

        n_shuffled_tests = 0
        n_shuffled_items = 0
        for pid in val_ids_list:
            res = sim_by_id.get(pid)
            if not res or not res.admin_metadata:
                continue
            test_order = res.admin_metadata.get("test_order", [])
            canonical_order = [
                t
                for t in [TEST_BFI, TEST_NFC, TEST_CRT2, TEST_BCRT]
                if t in self.tests
            ]
            if test_order and test_order != canonical_order:
                n_shuffled_tests += 1
            item_orders = res.admin_metadata.get("item_order", {})
            for tname, order in item_orders.items():
                test_def = self.tests.get(tname)
                if not test_def:
                    continue
                canonical_items = [it.item_id for it in test_def.items]
                if order and order != canonical_items:
                    n_shuffled_items += 1
                    break
        resp_metrics["order_randomisation"] = {
            "n_personas_with_shuffled_tests": n_shuffled_tests,
            "n_personas_with_shuffled_items": n_shuffled_items,
            "total_personas": len(val_ids_list),
        }

        return {
            "trait_fidelity": trait_metrics,
            "behavioural_alignment": behav_metrics,
            "response_consistency": resp_metrics,
        }


# ---------------------------------------------------------------------------
# Data loading and preparation
# ---------------------------------------------------------------------------


def _ensure_data_dir() -> str:
    if DATA_DIR is None:
        raise ValueError(
            "DATA_DIR is not configured. Ensure that both PROJECT_ROOT and "
            "DATA_PATH environment variables are set. Example:\n\n"
            "    export PROJECT_ROOT=/absolute/path/to/project\n"
            "    export DATA_PATH=data\n"
        )

    if not os.path.isdir(DATA_DIR):
        raise FileNotFoundError(
            f"DATA_DIR '{DATA_DIR}' does not exist. Verify that PROJECT_ROOT "
            "and DATA_PATH environment variables point to the correct "
            "directory containing the required CSV and YAML files."
        )
    return DATA_DIR


def _load_personas(personas_path: str) -> pd.DataFrame:
    if not os.path.isfile(personas_path):
        raise FileNotFoundError(
            f"Persona file '{personas_path}' not found. Ensure that "
            "synthetic_personas.csv is present in the DATA_DIR."
        )
    df = pd.read_csv(personas_path)

    required_cols = [
        "persona_id",
        "age",
        "gender",
        "bfi_extraversion",
        "bfi_agreeableness",
        "bfi_conscientiousness",
        "bfi_neuroticism",
        "bfi_openness",
        "nfc_score",
        "crt2_level",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            "synthetic_personas.csv is missing required columns: "
            f"{missing}. Available columns: {list(df.columns)}.\n"
            "Please ensure the file includes these columns."
        )

    return df


def _load_markers(markers_path: str) -> pd.DataFrame:
    if not os.path.isfile(markers_path):
        raise FileNotFoundError(
            f"Markers file '{markers_path}' not found. Ensure that "
            "serapio_goldberg_markers.csv is present in the DATA_DIR."
        )
    df = pd.read_csv(markers_path)

    required_cols = ["trait", "pole", "adjective"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            "serapio_goldberg_markers.csv is missing required columns: "
            f"{missing}. Available columns: {list(df.columns)}."
        )
    return df


def _safe_load_yaml(path: str) -> Any:
    """
    Load a YAML file. For writing_tasks.yaml, an existing but empty file
    (yaml.safe_load -> None) is treated as an empty list.
    """
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None and os.path.basename(path) == "writing_tasks.yaml":
        return []
    return data


# Hard-coded trait and keying maps for BFI-10 and NFC-18
BFI10_ITEM_TRAITS: Dict[str, str] = {
    "bfi1": "extraversion",
    "bfi2": "extraversion",
    "bfi3": "agreeableness",
    "bfi4": "agreeableness",
    "bfi5": "conscientiousness",
    "bfi6": "conscientiousness",
    "bfi7": "neuroticism",
    "bfi8": "neuroticism",
    "bfi9": "openness",
    "bfi10": "openness",
}
BFI10_REVERSE_IDS = {"bfi2", "bfi4", "bfi6", "bfi8", "bfi10"}

# Reverse-keyed NFC-18 items based on the standard short-form scoring key.
NFC18_ITEM_TRAITS: Dict[str, str] = {f"nfc{i}": "nfc" for i in range(1, 19)}
NFC18_REVERSE_IDS = {
    f"nfc{i}" for i in (3, 4, 6, 7, 9, 10, 12, 13, 15, 18)
}


def _load_likert_test_from_yaml(
    yaml_path: str,
    test_name: str,
    trait_map: Dict[str, str],
    reverse_ids: Iterable[str],
    scale_type: str = "likert_1_5",
) -> PsychometricTest:
    data = _safe_load_yaml(yaml_path)
    if not data:
        raise FileNotFoundError(
            f"Likert test definition YAML '{yaml_path}' is missing or empty."
        )

    reverse_ids_set = set(reverse_ids)
    items: List[PsychometricItem] = []

    for row in data:
        item_id = (
            row.get("id")
            or row.get("item_id")
            or row.get("name")
        )
        text = row.get("text") or row.get("prompt") or row.get("question")
        if not item_id or not text:
            raise ValueError(
                f"Invalid item in {yaml_path}: each entry must have id and text."
            )
        item_id_str = str(item_id)
        trait = trait_map.get(item_id_str)
        if trait is None:
            raise ValueError(
                f"Item '{item_id_str}' in {yaml_path} not found in trait map."
            )
        reverse_scored = item_id_str in reverse_ids_set
        items.append(
            PsychometricItem(
                item_id=item_id_str,
                text=str(text),
                trait=trait,
                reverse_scored=reverse_scored,
            )
        )

    instrument_id = None
    if isinstance(data, list) and data:
        instrument_id = os.path.basename(yaml_path)

    return PsychometricTest(
        name=test_name,
        items=items,
        scale_type=scale_type,
        instrument_id=instrument_id,
    )


def _load_crt_test_from_yaml(
    yaml_path: str,
    test_name: str,
    default_trait: str,
    correct_answers: Optional[Dict[str, str]] = None,
) -> PsychometricTest:
    """
    Load a CRT-style test from YAML.

    Correct answers must be provided either in the YAML (via a 'correct' or
    'answer' field) or via the `correct_answers` mapping. A missing answer
    for any item results in a ValueError.
    """
    data = _safe_load_yaml(yaml_path)
    if not data:
        raise FileNotFoundError(
            f"CRT test definition YAML '{yaml_path}' is missing or empty."
        )
    correct_answers = correct_answers or {}
    items: List[PsychometricItem] = []
    for row in data:
        item_id = (
            row.get("id")
            or row.get("item_id")
            or row.get("name")
        )
        text = row.get("text") or row.get("prompt") or row.get("question")
        if not item_id or not text:
            raise ValueError(
                f"Invalid CRT item in {yaml_path}: each entry must have id and text."
            )
        item_id_str = str(item_id)

        correct = row.get("correct") or row.get("answer")
        if not correct:
            correct = correct_answers.get(item_id_str)
        if not correct:
            raise ValueError(
                f"CRT item '{item_id_str}' in {yaml_path} is missing a correct answer. "
                "Provide it either in the YAML under 'correct'/'answer' or via the "
                "code-side correct_answers mapping."
            )

        items.append(
            PsychometricItem(
                item_id=item_id_str,
                text=str(text),
                trait=default_trait,
                reverse_scored=False,
                correct_answer=str(correct),
            )
        )

    instrument_id = None
    if isinstance(data, list) and data:
        instrument_id = os.path.basename(yaml_path)

    return PsychometricTest(
        name=test_name,
        items=items,
        scale_type="binary",
        instrument_id=instrument_id,
    )


def _load_writing_tasks_from_yaml(yaml_path: str) -> List[Dict[str, Any]]:
    """
    Load writing tasks from YAML.

    - If the file is missing, return [] and log a warning.
    - If the file exists but is empty, return [] and log an informational
      message.
    - If the file contains a non-list, ignore it and return [] with a warning.
    """
    if not os.path.isfile(yaml_path):
        logging.warning(
            "writing_tasks.yaml not found at %s; proceeding with a default writing task.",
            yaml_path,
        )
        return []

    data = _safe_load_yaml(yaml_path)

    # At this point the file exists. An empty file will result in [] from
    # _safe_load_yaml for writing_tasks.yaml.
    if data == [] and os.path.getsize(yaml_path) == 0:
        logging.info(
            "writing_tasks.yaml at %s is present but empty; no explicit writing tasks configured.",
            yaml_path,
        )
        return []

    if not isinstance(data, list):
        logging.warning(
            "writing_tasks.yaml at %s did not contain a list; ignoring content.",
            yaml_path,
        )
        return []

    return data


def _validate_test_bank(tests: Dict[str, PsychometricTest]) -> None:
    """
    Validate basic invariants for the psychometric test bank.

    - All Likert items in BFI must have traits in the Big Five domains.
    - All Likert items in NFC must have trait 'nfc'.
    - All CRT items (CRT2, bCRT) must have a meaningful correct_answer.
    """
    if TEST_BFI in tests:
        allowed = {
            "extraversion",
            "agreeableness",
            "conscientiousness",
            "neuroticism",
            "openness",
        }
        for item in tests[TEST_BFI].items:
            if not item.trait:
                raise ValueError(
                    f"BFI item '{item.item_id}' has an empty trait field."
                )
            if item.trait not in allowed:
                raise ValueError(
                    f"BFI item '{item.item_id}' has invalid trait '{item.trait}'."
                )

    if TEST_NFC in tests:
        for item in tests[TEST_NFC].items:
            if not item.trait:
                raise ValueError(
                    f"NFC item '{item.item_id}' has an empty trait field."
                )
            if item.trait != "nfc":
                raise ValueError(
                    f"NFC item '{item.item_id}' has invalid trait '{item.trait}'. "
                    "Expected 'nfc'."
                )

    for tname in [TEST_CRT2, TEST_BCRT]:
        if tname not in tests:
            continue
        for item in tests[tname].items:
            if not item.correct_answer:
                raise ValueError(
                    f"{tname} item '{item.item_id}' is missing a correct_answer."
                )
            if str(item.correct_answer).strip().upper() == "UNKNOWN":
                raise ValueError(
                    f"{tname} item '{item.item_id}' has placeholder correct_answer "
                    "'UNKNOWN'; provide a real scoring key."
                )


def _build_psychometric_tests(data_dir: str) -> Dict[str, PsychometricTest]:
    """
    Construct definitions of the psychometric tests used in the simulator
    by loading from YAML files and applying hard-coded trait/keying maps.
    """
    bfi_yaml = os.path.join(data_dir, "bfi10_items.yaml")
    nfc_yaml = os.path.join(data_dir, "nfc18_items.yaml")
    crt2_yaml = os.path.join(data_dir, "crt2_items.yaml")
    bcrt_yaml = os.path.join(data_dir, "bcrt_items.yaml")

    tests: Dict[str, PsychometricTest] = {}

    tests[TEST_BFI] = _load_likert_test_from_yaml(
        bfi_yaml,
        test_name=TEST_BFI,
        trait_map=BFI10_ITEM_TRAITS,
        reverse_ids=BFI10_REVERSE_IDS,
        scale_type="likert_1_5",
    )

    tests[TEST_NFC] = _load_likert_test_from_yaml(
        nfc_yaml,
        test_name=TEST_NFC,
        trait_map=NFC18_ITEM_TRAITS,
        reverse_ids=NFC18_REVERSE_IDS,
        scale_type="likert_1_5",
    )

    # Code-side correct answer dictionaries can be provided here if desired.
    crt2_correct: Dict[str, str] = {}
    bcrt_correct: Dict[str, str] = {}

    tests[TEST_CRT2] = _load_crt_test_from_yaml(
        crt2_yaml,
        test_name=TEST_CRT2,
        default_trait="crt2",
        correct_answers=crt2_correct,
    )
    tests[TEST_BCRT] = _load_crt_test_from_yaml(
        bcrt_yaml,
        test_name=TEST_BCRT,
        default_trait="bcrt",
        correct_answers=bcrt_correct,
    )

    _validate_test_bank(tests)
    return tests


def load_data(config: SimulationConfig) -> DataBundle:
    data_dir = _ensure_data_dir()
    personas_path = os.path.join(data_dir, "synthetic_personas.csv")
    markers_path = os.path.join(data_dir, "serapio_goldberg_markers.csv")
    writing_tasks_path = os.path.join(data_dir, "writing_tasks.yaml")

    personas_df = _load_personas(personas_path)
    markers_df = _load_markers(markers_path)
    tests = _build_psychometric_tests(data_dir)
    writing_tasks = _load_writing_tasks_from_yaml(writing_tasks_path)

    logging.info(
        "Loaded %d personas, %d markers, %d tests, and %d writing tasks.",
        len(personas_df),
        len(markers_df),
        len(tests),
        len(writing_tasks),
    )

    return DataBundle(
        personas_df=personas_df,
        markers_df=markers_df,
        tests=tests,
        writing_tasks=writing_tasks,
    )


# ---------------------------------------------------------------------------
# Network and agents construction
# ---------------------------------------------------------------------------


def build_network_and_agents(
    data_bundle: DataBundle, config: SimulationConfig
) -> SimulationState:
    personas: Dict[str, Persona] = {}
    for _, row in data_bundle.personas_df.iterrows():
        p = Persona.from_series(row)
        personas[p.persona_id] = p

    node_ids = list(personas.keys())
    network = SocialNetwork(nodes=node_ids)
    network.add_layer("friendship")
    network.add_layer("information")

    n = len(node_ids)
    for i, nid in enumerate(node_ids):
        if n > 1:
            neighbor = node_ids[(i + 1) % n]
            network.add_edge("friendship", nid, neighbor)

    for src in node_ids:
        for dst in node_ids:
            if src != dst:
                network.add_edge("information", src, dst)

    exogenous_signals = {
        "reflection_salience": ExogenousSignal(
            name="reflection_salience", values_by_time={0: 0.0, 1: 1.0}
        )
    }

    return SimulationState(
        personas=personas,
        network=network,
        exogenous_signals=exogenous_signals,
        data_bundle=data_bundle,
        config=config,
    )


# ---------------------------------------------------------------------------
# Holdout split
# ---------------------------------------------------------------------------


def holdout_split(
    state: SimulationState, config: SimulationConfig
) -> Tuple[List[str], List[str]]:
    df = state.data_bundle.personas_df.copy()
    if "timestamp" in df.columns:
        df.sort_values("timestamp", inplace=True)
    elif "time" in df.columns:
        df.sort_values("time", inplace=True)
    else:
        df.sort_values("persona_id", inplace=True)

    n_total = len(df)
    if n_total < 2:
        raise ValueError(
            "At least two personas are required for a holdout split; "
            f"found {n_total}."
        )

    n_val = max(1, int(round(config.holdout_fraction * n_total)))
    n_val = min(n_val, n_total - 1)

    val_df = df.tail(n_val)
    train_df = df.head(n_total - n_val)

    train_ids = train_df["persona_id"].astype(str).tolist()
    val_ids = val_df["persona_id"].astype(str).tolist()

    logging.info(
        "Holdout split: %d training personas, %d validation personas.",
        len(train_ids),
        len(val_ids),
    )

    return train_ids, val_ids


# ---------------------------------------------------------------------------
# Results saving
# ---------------------------------------------------------------------------


def save_results(
    sim_results: List[SimulationResult],
    metrics: Dict[str, Any],
    state: SimulationState,
    val_ids: Iterable[str],
    config: SimulationConfig,
) -> None:
    data_dir = _ensure_data_dir()
    if config.output_dir is None:
        output_dir = os.path.join(data_dir, "outputs")
    else:
        if os.path.isabs(config.output_dir):
            output_dir = config.output_dir
        else:
            output_dir = os.path.join(data_dir, config.output_dir)

    os.makedirs(output_dir, exist_ok=True)

    df_val = state.data_bundle.personas_df[
        state.data_bundle.personas_df["persona_id"].isin(list(val_ids))
    ].copy()
    df_val.set_index("persona_id", inplace=True)

    rows = []
    for res in sim_results:
        pid = res.persona_id
        if pid not in df_val.index:
            raise KeyError(
                f"Persona ID '{pid}' in simulation results is not present "
                "in validation DataFrame."
            )

        base = df_val.loc[pid].to_dict()
        for test_name, scores in res.test_scores.items():
            for key, val in scores.items():
                base[f"{test_name.lower()}_{key}"] = val
        for key, val in res.writing_features.items():
            base[f"writing_{key}"] = val
        base["writing_text"] = res.writing_text
        base["persona_prompt"] = res.persona_prompt

        full_json = {
            "test_item_responses": res.test_item_responses,
            "test_scores": res.test_scores,
            "writing_features": res.writing_features,
            "writing_text": res.writing_text,
            "retest_scores": res.retest_scores,
            "admin_metadata": res.admin_metadata,
        }
        base["full_response_json"] = json.dumps(full_json, ensure_ascii=False)

        rows.append({"persona_id": pid, **base})

    results_df = pd.DataFrame(rows)
    results_df.set_index("persona_id", inplace=True)

    results_path = os.path.join(output_dir, "simulation_results.csv")
    results_df.to_csv(results_path)

    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    logging.info("Saved simulation results to %s", results_path)
    logging.info("Saved metrics to %s", metrics_path)


# ---------------------------------------------------------------------------
# CLI parsing and main orchestrator
# ---------------------------------------------------------------------------


def parse_cli() -> SimulationConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Simulate LLM-based personas from synthetic_personas.csv, "
            "administer psychometric tests and writing tasks, and evaluate "
            "trait fidelity, behavioural alignment, and response consistency."
        )
    )
    parser.add_argument(
        "--admin-mode",
        choices=["per_item", "per_test", "all_tests"],
        default="per_test",
        help=(
            "Granularity of simulated API calls: per_item, per_test, or all_tests. "
            "Currently affects grouping and recorded metadata only; psychometric "
            "tests are simulated analytically, not via LLM."
        ),
    )
    parser.add_argument(
        "--shuffle-items",
        action="store_true",
        help="If set, shuffle items within tests (ignored if --fixed-order is in effect).",
    )
    parser.add_argument(
        "--shuffle-tests",
        action="store_true",
        help="If set, shuffle order of tests (ignored if --fixed-order is in effect).",
    )
    parser.add_argument(
        "--fixed-order",
        dest="fixed_order",
        action="store_true",
        default=True,
        help=(
            "Enforce a fixed item and test order for full reproducibility. "
            "This overrides shuffling flags."
        ),
    )
    parser.add_argument(
        "--no-fixed-order",
        dest="fixed_order",
        action="store_false",
        help=(
            "Disable fixed order so that --shuffle-items and --shuffle-tests "
            "can randomise administration."
        ),
    )
    parser.add_argument(
        "--holdout-fraction",
        type=float,
        default=0.2,
        help="Fraction of personas reserved for validation (0 < f < 1).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Output directory for results. If relative, interpreted "
            "relative to DATA_DIR. Defaults to DATA_DIR/outputs."
        ),
    )
    parser.add_argument(
        "--bfi-prompt-mode",
        choices=["coarse_numeric", "coarse_descriptive", "granular_serapio"],
        default="granular_serapio",
        help="Prompt mode for BFI-based persona shaping.",
    )
    parser.add_argument(
        "--crt-prompt-mode",
        choices=["numeric", "descriptive"],
        default="descriptive",
        help="Prompt mode for CRT2-based persona shaping.",
    )
    parser.add_argument(
        "--test-retest-fraction",
        type=float,
        default=0.0,
        help=(
            "Fraction of validation personas to receive a second "
            "administration of the tests (0 <= f <= 1)."
        ),
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=GLOBAL_RANDOM_SEED,
        help="Global random seed for reproducibility.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level: DEBUG, INFO, WARNING, ERROR, or CRITICAL.",
    )

    args = parser.parse_args()

    config = SimulationConfig(
        random_seed=args.random_seed,
        admin_mode=args.admin_mode,
        shuffle_items=args.shuffle_items,
        shuffle_tests=args.shuffle_tests,
        fixed_order=args.fixed_order,
        holdout_fraction=args.holdout_fraction,
        output_dir=args.output_dir,
        bfi_prompt_mode=args.bfi_prompt_mode,
        crt_prompt_mode=args.crt_prompt_mode,
        nfc_prompt_mode="granular_9_level",
        test_retest_fraction=args.test_retest_fraction,
        log_level=args.log_level.upper(),
    )
    config.validate()

    logging.basicConfig(
        level=getattr(logging, config.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    random.seed(config.random_seed)
    np.random.seed(config.random_seed)

    logging.info("SimulationConfig: %s", config)

    return config


def main() -> None:
    """
    Main orchestrator for the simulation run.
    """
    config = parse_cli()
    data_bundle = load_data(config)
    state = build_network_and_agents(data_bundle, config)
    train_ids, val_ids = holdout_split(state, config)

    calibrator = Calibrator(config)
    calibrator.fit(state, train_ids)

    simulator = Simulator(state, calibrator)
    sim_results = simulator.rollout(val_ids)

    evaluator = Evaluator(config, tests=state.data_bundle.tests)
    metrics = evaluator.compute_metrics(sim_results, state, val_ids)

    save_results(sim_results, metrics, state, val_ids, config)

    logging.info("Simulation pipeline completed successfully.")


main()