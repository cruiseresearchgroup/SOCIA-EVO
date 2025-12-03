#!/usr/bin/env python3
"""
simulate.py

End-to-end multi-agent simulator for LLM-based personas parameterised by
human trait scores (BFI, Need for Cognition, CRT2).

The script:

1. Parses CLI arguments.
2. Loads synthetic personas and (optionally) lexical marker metadata.
3. Builds persona prompts and a simple interaction network.
4. Performs a temporal-style holdout split into train and validation sets.
5. Calibrates simulator parameters on the training set.
6. Runs a forward simulation (rollout) on the validation set.
7. Evaluates trait fidelity, behavioural alignment, and response consistency.
8. Saves responses, metrics, and configuration to disk.

The implementation is fully deterministic (given a random seed) and is
designed to be modular, allowing replacement of the calibration algorithm
or simulation components without changing the orchestrator.
"""

import argparse
import itertools
import json
import math
import os
import random
import re
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from openai import OpenAI


# ---------------------------------------------------------------------
# Global configuration and reproducibility
# ---------------------------------------------------------------------

PROJECT_ROOT = os.environ.get("PROJECT_ROOT")
if PROJECT_ROOT is None:
    PROJECT_ROOT = os.getcwd()

DATA_PATH = os.environ.get("DATA_PATH")
if DATA_PATH is None:
    # Default to the persona_trait_experiment folder as per spec
    DATA_DIR = os.path.join(PROJECT_ROOT, "data_fitting", "persona_trait_experiment")
else:
    DATA_DIR = os.path.join(PROJECT_ROOT, DATA_PATH)

GLOBAL_RANDOM_SEED = 42
random.seed(GLOBAL_RANDOM_SEED)
np.random.seed(GLOBAL_RANDOM_SEED)


# ---------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------


@dataclass
class SimulationConfig:
    """
    Configuration for a simulation run.

    Attributes
    ----------
    personas_filename : str
        CSV file name for synthetic personas, relative to DATA_DIR.
    markers_filename : str
        CSV file name for adjective markers (Serapio/Goldberg), relative
        to DATA_DIR.
    admin_mode : str
        Administration granularity: 'per_item', 'per_test', or 'all_tests'.
    shuffle_items_within_tests : bool
        Whether to shuffle item order within each test.
    shuffle_test_order : bool
        Whether to shuffle the order of tests.
    holdout_fraction : float
        Fraction of personas used for validation (temporal-style holdout).
    time_column : Optional[str]
        Column name used for temporal ordering, if available.
    random_seed : int
        Random seed for this run.
    output_dir : str
        Directory (relative to PROJECT_ROOT) where results are stored.
    bfi_columns : Dict[str, str]
        Mapping from Big Five domain keys to column names in personas CSV.
    nfc_column : str
        Column name for Need for Cognition score (continuous target).
    crt_column : str
        Column name for CRT2 continuous score (e.g., CRT2_Total_adj).
    crt_level_column : str
        Column name for discrete CRT2 level (0-4) derived from crt_column.
    use_crt_numeric_only : bool
        If True, use numeric-only CRT specification; else include
        descriptive sentence.
    use_openai : bool
        If True, use OpenAI LLM via Responses API instead of PseudoLLM.
    openai_model : str
        OpenAI model name to use (default 'gpt-5').
    openai_max_output_tokens : int
        Maximum output tokens for Responses API calls.
    """

    personas_filename: str = "synthetic_personas.csv"
    markers_filename: str = "serapio_goldberg_markers.csv"
    admin_mode: str = "per_test"
    shuffle_items_within_tests: bool = False
    shuffle_test_order: bool = False
    holdout_fraction: float = 0.2
    time_column: Optional[str] = None
    random_seed: int = GLOBAL_RANDOM_SEED
    output_dir: str = "simulation_outputs"
    # Default to the real synthetic_personas.csv schema
    bfi_columns: Dict[str, str] = field(
        default_factory=lambda: {
            "extraversion": "BFI_Extraversion",
            "agreeableness": "BFI_Agreeableness",
            "conscientiousness": "BFI_Conscientiousness",
            "neuroticism": "BFI_Neuroticism",
            "openness": "BFI_Openness",
        }
    )
    nfc_column: str = "NFC_Total"
    crt_column: str = "CRT2_Total_adj"
    crt_level_column: str = "crt2_level"
    use_crt_numeric_only: bool = False
    use_openai: bool = False
    openai_model: str = "gpt-5"
    openai_max_output_tokens: int = 4000

    def validate(self) -> None:
        """
        Validate configuration values.

        Raises
        ------
        ValueError
            If configuration values are inconsistent or invalid.
        """
        if self.admin_mode not in {"per_item", "per_test", "all_tests"}:
            raise ValueError(
                f"Invalid admin_mode '{self.admin_mode}'. "
                "Must be one of: 'per_item', 'per_test', 'all_tests'."
            )
        if not (0.0 < self.holdout_fraction < 1.0):
            raise ValueError(
                f"holdout_fraction must be between 0 and 1, got "
                f"{self.holdout_fraction}."
            )
        if self.random_seed < 0:
            raise ValueError("random_seed must be non-negative.")
        # Column mapping sanity check
        if not self.bfi_columns:
            raise ValueError("bfi_columns mapping must not be empty.")


@dataclass
class TestItem:
    """
    Single test item (question or writing task).

    Attributes
    ----------
    item_id : str
        Unique identifier of the item within its test.
    test_name : str
        Name of the test (e.g., 'BFI', 'NFC', 'CRT2', 'Writing').
    scale_name : Optional[str]
        Scale or subscale name associated with the item (e.g.,
        'extraversion'). None for items that do not belong to a classic
        scale (e.g., some writing prompts).
    item_text : str
        Text presented to the LLM persona.
    trait : Optional[str]
        Primary trait the item is intended to measure or load on
        (e.g., 'extraversion', 'nfc', 'crt2').
    keyed : int
        Direction of scoring: +1 for positively keyed (higher trait
        implies higher response), -1 for reverse-keyed items, 0 for
        non-Likert/writing items.
    item_type : str
        One of 'likert', 'crt', or 'open'.
    correct_answer : Optional[str]
        Canonical correct answer for CRT items; None otherwise.
    lure_answers : Optional[List[str]]
        List of typical incorrect answers for CRT items.
    """

    item_id: str
    test_name: str
    scale_name: Optional[str]
    item_text: str
    trait: Optional[str]
    keyed: int
    item_type: str
    correct_answer: Optional[str] = None
    lure_answers: Optional[List[str]] = None


@dataclass
class TestBank:
    """
    Collection of all psychometric tests and writing tasks.

    Attributes
    ----------
    items : List[TestItem]
        Flat list of all items across all tests.
    """

    items: List[TestItem]

    def items_by_test(self) -> Dict[str, List[TestItem]]:
        """
        Group items by test name.

        Returns
        -------
        Dict[str, List[TestItem]]
            Mapping from test name to list of items.
        """
        grouped: Dict[str, List[TestItem]] = {}
        for item in self.items:
            grouped.setdefault(item.test_name, []).append(item)
        return grouped


@dataclass
class SimulationResult:
    """
    Container for simulation outputs.

    Attributes
    ----------
    per_persona : pd.DataFrame
        Persona-level aggregated scores and writing metrics.
    per_item : pd.DataFrame
        Item-level responses (including numeric scoring where relevant).
    config : SimulationConfig
        Configuration used to produce these results.
    api_calls : Optional[pd.DataFrame]
        Metadata about logical API calls/blocks (one row per call).
    """

    per_persona: pd.DataFrame
    per_item: pd.DataFrame
    config: SimulationConfig
    api_calls: Optional[pd.DataFrame] = None


# ---------------------------------------------------------------------
# OpenAI integration
# ---------------------------------------------------------------------


def get_openai_api_key() -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        return api_key
    raise ValueError("OpenAI API key not found in environment")


def call_gpt5_with_responses_api(
    prompt: str, model: str = "gpt-5", max_output_tokens: int = 4000
) -> str:
    api_key = get_openai_api_key()
    client = OpenAI(api_key=api_key)

    responses_kwargs = {
        "model": model,
        "input": [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": prompt}],
            }
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
                content = (
                    output[0].get("content") if isinstance(output[0], dict) else None
                )
                if content and isinstance(content, list) and len(content) > 0:
                    text = content[0].get("text")
                    if isinstance(text, str):
                        return text
        except Exception:
            pass
        return str(resp_obj)

    return extract_response(resp)


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------


def ensure_dir(path: str) -> None:
    """
    Ensure that a directory exists, creating it if necessary.

    Parameters
    ----------
    path : str
        Path to the directory.
    """
    os.makedirs(path, exist_ok=True)


def pearson_corr(x: pd.Series, y: pd.Series) -> float:
    """
    Compute Pearson correlation between two pandas Series.

    Parameters
    ----------
    x : pd.Series
        First variable.
    y : pd.Series
        Second variable.

    Returns
    -------
    float
        Pearson correlation coefficient. NaN if undefined.
    """
    if x.size == 0 or y.size == 0:
        return float("nan")
    if x.nunique() <= 1 or y.nunique() <= 1:
        return float("nan")
    return float(x.corr(y))


def spearman_corr(x: pd.Series, y: pd.Series) -> float:
    """
    Compute Spearman rank correlation between two pandas Series.

    Parameters
    ----------
    x : pd.Series
        First variable.
    y : pd.Series
        Second variable.

    Returns
    -------
    float
        Spearman rank correlation coefficient. NaN if undefined.
    """
    if x.size == 0 or y.size == 0:
        return float("nan")
    return pearson_corr(x.rank(method="average"), y.rank(method="average"))


def cronbach_alpha(df: pd.DataFrame) -> float:
    """
    Compute Cronbach's alpha for a set of items.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame of shape (n_observations, n_items) containing item
        scores. Rows correspond to subjects; columns correspond to items.

    Returns
    -------
    float
        Cronbach's alpha. NaN if cannot be computed (e.g., <2 items or
        zero variance).
    """
    k = df.shape[1]
    if k < 2 or df.shape[0] < 2:
        return float("nan")

    item_vars = df.var(axis=0, ddof=1)
    total_score = df.sum(axis=1)
    total_var = total_score.var(ddof=1)
    if total_var == 0:
        return float("nan")

    alpha = (k / (k - 1.0)) * (1.0 - item_vars.sum() / total_var)
    return float(alpha)


def map_crt_total_to_level(scores: pd.Series) -> pd.Series:
    """
    Map continuous CRT total scores to discrete levels 0–4 using quantiles.

    Parameters
    ----------
    scores : pd.Series
        Continuous CRT scores.

    Returns
    -------
    pd.Series
        Discrete levels in {0,1,2,3,4}.
    """
    scores = scores.astype(float)
    if scores.dropna().empty:
        return pd.Series([2] * len(scores), index=scores.index, dtype=int)

    qs = scores.quantile([0.2, 0.4, 0.6, 0.8])
    q1, q2, q3, q4 = qs.loc[0.2], qs.loc[0.4], qs.loc[0.6], qs.loc[0.8]

    def to_level(val: float) -> int:
        if pd.isna(val):
            return 2
        if val <= q1:
            return 0
        if val <= q2:
            return 1
        if val <= q3:
            return 2
        if val <= q4:
            return 3
        return 4

    return scores.apply(to_level).astype(int)


# ---------------------------------------------------------------------
# Persona prompt construction
# ---------------------------------------------------------------------


class PersonaPromptBuilder:
    """
    Build persona prompts based on trait scores and blueprint specifications.

    This class encapsulates the transformation from numeric trait scores
    (BFI, NFC, CRT2) into rich, natural language persona descriptions
    using qualifier scales and descriptor markers.
    """

    QUALIFIER_SCALE_1_TO_9: Dict[int, str] = {
        1: "extremely {low_adjective}",
        2: "very {low_adjective}",
        3: "{low_adjective}",
        4: "a bit {low_adjective}",
        5: "neither {low_adjective} nor {high_adjective}",
        6: "a bit {high_adjective}",
        7: "{high_adjective}",
        8: "very {high_adjective}",
        9: "extremely {high_adjective}",
    }

    def __init__(
        self,
        markers_df: Optional[pd.DataFrame] = None,
        rng: Optional[np.random.Generator] = None,
    ):
        """
        Initialize the prompt builder.

        Parameters
        ----------
        markers_df : Optional[pd.DataFrame]
            Optional DataFrame containing Serapio/Goldberg markers. If
            None or malformed, simple default descriptors are used.
            Expected columns (if provided): 'domain', 'pole', 'marker'.
        rng : Optional[np.random.Generator]
            Random number generator for deterministic descriptor sampling.
        """
        self.markers_df = markers_df
        self.rng = rng if rng is not None else np.random.default_rng(GLOBAL_RANDOM_SEED)
        self._validate_markers_df()

    def _validate_markers_df(self) -> None:
        """Validate markers DataFrame structure; fall back if invalid."""
        if self.markers_df is None:
            return
        required = {"domain", "pole", "marker"}
        if not required.issubset(self.markers_df.columns):
            warnings.warn(
                "Markers DataFrame is missing required columns "
                f"{required}; falling back to default descriptors.",
                UserWarning,
            )
            self.markers_df = None

    @staticmethod
    def _score_to_9_level(score: float) -> int:
        """
        Map synthetic scores on 1–5 scale (with 0.5 steps) to 1–9 levels.

        Parameters
        ----------
        score : float
            Score on a 1–5 scale.

        Returns
        -------
        int
            Level on a 1–9 scale.

        Raises
        ------
        ValueError
            If score is outside [1, 5].
        """
        if score < 1.0 or score > 5.0:
            raise ValueError(
                f"Score for 1–5 mapping must be in [1, 5], got {score}."
            )
        level = int(round(2 * (score - 1.0) + 1.0))
        return max(1, min(9, level))

    def _sample_markers(
        self, domain: str, pole: str, n: int = 3
    ) -> List[str]:
        """
        Sample adjective markers for a given domain and pole.

        Parameters
        ----------
        domain : str
            Big Five domain (e.g., 'extraversion').
        pole : str
            'low' or 'high'.
        n : int
            Number of markers to sample (with replacement if necessary).

        Returns
        -------
        List[str]
            List of adjectives or short phrases.
        """
        default_markers: Dict[str, Dict[str, List[str]]] = {
            "extraversion": {
                "low": ["quiet", "reserved", "shy"],
                "high": ["outgoing", "talkative", "energetic"],
            },
            "agreeableness": {
                "low": ["critical", "stubborn", "cold"],
                "high": ["kind", "cooperative", "warm"],
            },
            "conscientiousness": {
                "low": ["careless", "disorganized", "impulsive"],
                "high": ["organized", "reliable", "hard-working"],
            },
            "neuroticism": {
                "low": ["calm", "emotionally stable"],
                "high": ["anxious", "moody", "easily upset"],
            },
            "openness": {
                "low": ["conventional", "unimaginative"],
                "high": ["curious", "imaginative", "open to new ideas"],
            },
        }

        if self.markers_df is not None:
            subset = self.markers_df[
                (self.markers_df["domain"].str.lower() == domain.lower())
                & (self.markers_df["pole"].str.lower() == pole.lower())
            ]
            if not subset.empty:
                markers = subset["marker"].dropna().astype(str).tolist()
                if not markers:
                    return default_markers.get(domain, {}).get(pole, [])
                return list(
                    self.rng.choice(
                        markers, size=min(n, len(markers)), replace=False
                    )
                )

        return default_markers.get(domain, {}).get(pole, [])

    def build_bfi_granular_prompt(
        self, traits: Dict[str, float]
    ) -> str:
        """
        Build a granular BFI-based persona description.

        Parameters
        ----------
        traits : Dict[str, float]
            Mapping from Big Five domain names to scores (1–5).

        Returns
        -------
        str
            Natural language persona description for BFI traits.
        """
        segments: List[str] = []
        for domain, score in traits.items():
            level = self._score_to_9_level(score)
            template = self.QUALIFIER_SCALE_1_TO_9[level]
            if level <= 4:
                low = ", ".join(self._sample_markers(domain, "low", n=2))
                high = ", ".join(self._sample_markers(domain, "high", n=2))
            elif level >= 6:
                low = ", ".join(self._sample_markers(domain, "low", n=2))
                high = ", ".join(self._sample_markers(domain, "high", n=2))
            else:  # level == 5
                low = ", ".join(self._sample_markers(domain, "low", n=1))
                high = ", ".join(self._sample_markers(domain, "high", n=1))
            descriptor = template.format(
                low_adjective=low if low else "low on this trait",
                high_adjective=high if high else "high on this trait",
            )
            domain_label = domain.capitalize()
            segments.append(
                f"In terms of {domain_label}, I am {descriptor}."
            )

        joined = " ".join(segments)
        return (
            "For the following tasks, respond as a person described as: "
            f"\"{joined}\""
        )

    def build_nfc_prompt(self, nfc_score: float) -> str:
        """
        Build a Need for Cognition persona description.

        Parameters
        ----------
        nfc_score : float
            NFC score on 1–5 scale (will be mapped to 1–9 levels).

        Returns
        -------
        str
            Natural language persona description for NFC.
        """
        level = self._score_to_9_level(nfc_score)

        high_descriptors = [
            "enjoys complex rather than simple problems",
            "likes responsibility for situations that require a lot of thinking",
            "finds satisfaction in deliberating hard and for long periods",
            "enjoys tasks that involve generating new solutions to problems",
            "prefers life to be filled with puzzles and challenging questions",
            "finds abstract thinking appealing",
            "prefers intellectual, difficult, and important tasks",
            "often deliberates about issues even when they do not affect them "
            "personally",
        ]
        low_descriptors = [
            "thinks only as hard as necessary",
            "prefers tasks that require little thought once learned",
            "would rather do something that requires little thought",
            "tries to avoid situations that demand deep thinking",
            "finds thinking effortful rather than fun",
            "feels relief rather than satisfaction after heavy mental effort",
            "is content when something works without understanding how or why",
        ]

        if level <= 4:
            chosen_low = self.rng.choice(
                low_descriptors, size=3, replace=False
            ).tolist()
            qualifier = self.QUALIFIER_SCALE_1_TO_9[level]
            clause = ", ".join(chosen_low)
            sentence = qualifier.format(
                low_adjective=clause, high_adjective=""
            )
            sentence = sentence.replace("  ", " ").strip(", ")
            return (
                "For the following tasks, respond as a person described as: "
                f"\"I {sentence}.\""
            )
        elif level >= 6:
            chosen_high = self.rng.choice(
                high_descriptors, size=3, replace=False
            ).tolist()
            qualifier = self.QUALIFIER_SCALE_1_TO_9[level]
            clause = ", ".join(chosen_high)
            sentence = qualifier.format(
                low_adjective="", high_adjective=clause
            )
            sentence = sentence.replace("  ", " ").strip(", ")
            return (
                "For the following tasks, respond as a person described as: "
                f"\"I {sentence}.\""
            )
        else:  # level == 5
            chosen_low = self.rng.choice(
                low_descriptors, size=2, replace=False
            ).tolist()
            chosen_high = self.rng.choice(
                high_descriptors, size=2, replace=False
            ).tolist()
            low_clause = ", ".join(chosen_low)
            high_clause = ", ".join(chosen_high)
            sentence = (
                f"neither strongly avoids thinking ({low_clause}) "
                f"nor strongly seeks out thinking ({high_clause})"
            )
            return (
                "For the following tasks, respond as a person described as: "
                f"\"I am {sentence}.\""
            )

    def build_crt_prompt(
        self, crt_level: int, numeric_only: bool = False
    ) -> str:
        """
        Build a CRT-style reflective thinking persona description.

        Parameters
        ----------
        crt_level : int
            Discrete CRT2 ability level (0–4).
        numeric_only : bool
            If True, use only numeric tag specification. Otherwise include
            the descriptive sentence associated with the level.

        Returns
        -------
        str
            Natural language persona description for CRT-style thinking.

        Raises
        ------
        ValueError
            If crt_level is outside [0, 4].
        """
        if crt_level < 0 or crt_level > 4:
            raise ValueError(
                f"crt_level must be between 0 and 4 (inclusive), "
                f"got {crt_level}."
            )

        level_descriptions = {
            0: "I almost always trust my first impression, answer quickly "
            "without re-checking, and rarely notice when a question might be "
            "tricky.",
            1: "I often go with my first impression and only occasionally "
            "stop to reconsider whether it might be misleading.",
            2: "I sometimes pause to reconsider my first impression before "
            "answering, but I am inconsistent and often stick with the "
            "obvious answer.",
            3: "I usually pause to check whether an obvious answer could be "
            "a trap, and I am willing to change my mind after thinking "
            "things through.",
            4: "I almost always look for hidden assumptions, carefully check "
            "for tricks, and verify my answers with calculations before "
            "responding.",
        }

        numeric_tag = (
            f"This persona has a CRT2 ability level of {crt_level} on a 0–4 "
            "scale."
        )

        if numeric_only:
            return (
                "For the following CRT-style questions, respond as a person "
                f"with the following thinking style: \"{numeric_tag}\""
            )

        sentence = level_descriptions[crt_level]
        return (
            "For the following CRT-style questions, respond as a person "
            f"described as: \"{sentence}\""
        )

    def build_full_persona_prompt(
        self,
        bfi_traits: Dict[str, float],
        nfc_score: float,
        crt_level: int,
        numeric_crt_only: bool = False,
    ) -> str:
        """
        Build a combined persona prompt including BFI, NFC, and CRT specs.

        Parameters
        ----------
        bfi_traits : Dict[str, float]
            Big Five trait scores.
        nfc_score : float
            Need for Cognition score (1–5).
        crt_level : int
            CRT2 ability level (0–4).
        numeric_crt_only : bool
            If True, use numeric-only CRT specification.

        Returns
        -------
        str
            Combined persona prompt for configuring the pseudo-LLM.
        """
        bfi_prompt = self.build_bfi_granular_prompt(bfi_traits)
        nfc_prompt = self.build_nfc_prompt(nfc_score)
        crt_prompt = self.build_crt_prompt(crt_level, numeric_only=numeric_crt_only)

        # Concatenate with spacing
        return " ".join([bfi_prompt, nfc_prompt, crt_prompt])


# ---------------------------------------------------------------------
# Test bank construction (currently hardcoded; YAML integration TBD)
# ---------------------------------------------------------------------


def build_test_bank() -> TestBank:
    """
    Construct the psychometric test bank and writing tasks.

    Returns
    -------
    TestBank
        TestBank instance containing all items.
    """
    items: List[TestItem] = []

    # Simplified BFI-brief: 2 items per domain (10 total)
    bfi_definitions = {
        "extraversion": [
            ("BFI_E1", "I am talkative.", 1),
            ("BFI_E2", "I am reserved.", -1),
        ],
        "agreeableness": [
            ("BFI_A1", "I am considerate and kind to almost everyone.", 1),
            ("BFI_A2", "I tend to find fault with others.", -1),
        ],
        "conscientiousness": [
            ("BFI_C1", "I do a thorough job.", 1),
            ("BFI_C2", "I tend to be careless.", -1),
        ],
        "neuroticism": [
            ("BFI_N1", "I worry a lot.", 1),
            ("BFI_N2", "I am relaxed, handle stress well.", -1),
        ],
        "openness": [
            ("BFI_O1", "I am original, come up with new ideas.", 1),
            ("BFI_O2", "I have little interest in abstract ideas.", -1),
        ],
    }

    for domain, defs in bfi_definitions.items():
        for item_id, text, keyed in defs:
            items.append(
                TestItem(
                    item_id=item_id,
                    test_name="BFI",
                    scale_name=domain,
                    item_text=text,
                    trait=domain,
                    keyed=keyed,
                    item_type="likert",
                )
            )

    # NFC-18: simplified subset of 6 items
    nfc_defs = [
        ("NFC1", "I would prefer complex to simple problems.", 1),
        ("NFC2", "I like to have the responsibility of handling a situation "
                 "that requires a lot of thinking.", 1),
        ("NFC3", "Thinking is not my idea of fun.", -1),
        ("NFC4", "The idea of relying on thought to make my way to the top "
                 "appeals to me.", 1),
        ("NFC5", "I try to avoid situations that require thinking in depth "
                 "about something.", -1),
        ("NFC6", "I prefer to think about small, daily projects to long-term "
                 "ones.", -1),
    ]

    for item_id, text, keyed in nfc_defs:
        items.append(
            TestItem(
                item_id=item_id,
                test_name="NFC",
                scale_name="nfc",
                item_text=text,
                trait="nfc",
                keyed=keyed,
                item_type="likert",
            )
        )

    # CRT2-like items (4 items)
    crt_items = [
        (
            "CRT1",
            "A bat and a ball cost $1.10 in total. The bat costs $1.00 more "
            "than the ball. How much does the ball cost?",
            "0.05",
            ["0.10", "0.1", "10 cents"],
        ),
        (
            "CRT2",
            "If it takes 5 machines 5 minutes to make 5 widgets, how long "
            "would it take 100 machines to make 100 widgets?",
            "5",
            ["100", "100 minutes"],
        ),
        (
            "CRT3",
            "In a lake, there is a patch of lily pads. Every day, the patch "
            "doubles in size. If it takes 48 days for the patch to cover the "
            "entire lake, how long would it take for the patch to cover half "
            "of the lake?",
            "47",
            ["24", "24 days"],
        ),
        (
            "CRT4",
            "A farmer had 15 sheep and all but 8 died. How many are left?",
            "8",
            ["7", "0"],
        ),
    ]
    for item_id, text, correct, lures in crt_items:
        items.append(
            TestItem(
                item_id=item_id,
                test_name="CRT2",
                scale_name="crt2",
                item_text=text,
                trait="crt2",
                keyed=0,
                item_type="crt",
                correct_answer=correct,
                lure_answers=lures,
            )
        )

    # Simple writing tasks
    writing_tasks = [
        (
            "WT1",
            "Writing",
            None,
            "Please write a short reflection (150-200 words) about a time "
            "you solved a difficult problem. Describe what you thought and "
            "felt during the process.",
        ),
        (
            "WT2",
            "Writing",
            None,
            "Please write a short reflection (150-200 words) about a time "
            "you noticed that your first impression was misleading and you "
            "changed your mind.",
        ),
    ]
    for item_id, test_name, scale_name, text in writing_tasks:
        items.append(
            TestItem(
                item_id=item_id,
                test_name=test_name,
                scale_name=scale_name,
                item_text=text,
                trait=None,
                keyed=0,
                item_type="open",
            )
        )

    return TestBank(items=items)


# ---------------------------------------------------------------------
# Pseudo LLM simulator
# ---------------------------------------------------------------------


class PseudoLLM:
    """
    Simple, fully deterministic pseudo-LLM that simulates responses based
    on persona traits and random perturbations.

    This is a stand-in for actual LLM API calls and is calibrated to
    approximate desired psychometric properties rather than natural
    language generation fidelity.
    """

    def __init__(
        self,
        likert_noise: float = 0.5,
        crt_slip: float = 0.2,
        random_seed: int = GLOBAL_RANDOM_SEED,
    ):
        """
        Initialize the pseudo-LLM.

        Parameters
        ----------
        likert_noise : float
            Standard deviation of Gaussian noise added to latent trait
            before mapping to Likert responses.
        crt_slip : float
            Base error probability for CRT items even at highest CRT level.
        random_seed : int
            Random seed for deterministic behaviour.
        """
        self.likert_noise = likert_noise
        self.crt_slip = crt_slip
        self.rng = np.random.default_rng(random_seed)

    def _latent_from_trait(
        self, trait_value: float, keyed: int
    ) -> float:
        """
        Compute latent variable from a trait value and item keying.

        Parameters
        ----------
        trait_value : float
            Trait value on 1–5 scale.
        keyed : int
            +1 for positively keyed, -1 for reverse-keyed.

        Returns
        -------
        float
            Latent value (higher means more endorsement).
        """
        centered = trait_value - 3.0  # center at neutral
        direction = keyed if keyed != 0 else 1
        return direction * centered

    def answer_likert(
        self, trait_value: float, keyed: int
    ) -> int:
        """
        Simulate a Likert response (1–5) for a given trait and item.

        Parameters
        ----------
        trait_value : float
            Trait value on 1–5 scale.
        keyed : int
            +1 for positively keyed, -1 for reverse-keyed.

        Returns
        -------
        int
            Simulated Likert response (1–5).
        """
        latent = self._latent_from_trait(trait_value, keyed)
        noisy = latent + self.rng.normal(0.0, self.likert_noise)
        # Map latent roughly [-2,2] to [1,5]
        mapped = 3.0 + noisy
        response = int(round(mapped))
        return max(1, min(5, response))

    def answer_crt(self, crt_level: int, item: TestItem) -> Tuple[str, int]:
        """
        Simulate a CRT-style numeric answer.

        Parameters
        ----------
        crt_level : int
            CRT ability level (0–4).
        item : TestItem
            CRT item metadata.

        Returns
        -------
        Tuple[str, int]
            (textual answer, correctness flag 0/1).
        """
        if item.correct_answer is None:
            raise ValueError("CRT item must have a correct_answer defined.")

        # Probability of a correct answer increases with crt_level
        p_correct_base = 0.1
        p_correct = min(
            1.0 - self.crt_slip,
            p_correct_base + (crt_level / 4.0) * (0.9 - self.crt_slip),
        )
        if self.rng.random() < p_correct:
            return item.correct_answer, 1

        # Incorrect answer: choose from lure answers or random alteration
        if item.lure_answers:
            answer = str(
                self.rng.choice(
                    item.lure_answers, size=1, replace=True
                )[0]
            )
        else:
            # crude fallback: perturb correct answer slightly
            try:
                val = float(item.correct_answer)
                val += self.rng.integers(-3, 4)
                answer = str(val)
            except ValueError:
                answer = "I am not sure."
        return answer, 0

    def generate_writing(
        self,
        persona_prompt: str,
        traits: Dict[str, float],
        nfc_score: float,
        crt_level: int,
        item: TestItem,
    ) -> str:
        """
        Generate a pseudo writing sample conditioned on traits.

        Parameters
        ----------
        persona_prompt : str
            Full persona prompt (unused in detail but available).
        traits : Dict[str, float]
            Big Five trait scores (1–5).
        nfc_score : float
            Need for Cognition score.
        crt_level : int
            CRT level (0–4).
        item : TestItem
            Writing task metadata.

        Returns
        -------
        str
            Generated writing text.
        """
        # word pools reflecting traits
        ext = traits.get("extraversion", 3.0)
        nfc = nfc_score
        crt = float(crt_level)

        social_words = [
            "friends",
            "conversation",
            "team",
            "together",
            "crowd",
            "talkative",
        ]
        analytic_words = [
            "analyze",
            "because",
            "therefore",
            "logic",
            "structure",
            "methodical",
        ]
        insight_words = [
            "realized",
            "understood",
            "noticed",
            "reflected",
            "insight",
        ]
        affect_words = [
            "happy",
            "anxious",
            "relieved",
            "frustrated",
            "excited",
        ]

        base_sentences = [
            "I remember this situation very clearly.",
            "At first, I reacted in a very intuitive way.",
            "As I thought more carefully, my perspective began to change.",
            "Step by step, I tried to understand what was really going on.",
            "Eventually I decided on a course of action.",
            "Looking back, the experience taught me a lot about myself.",
        ]

        num_sentences = 6 + int(round((nfc - 3.0) * 2)) + int(
            round((crt - 2.0))
        )
        num_sentences = max(4, min(10, num_sentences))

        sentences: List[str] = []
        for i in range(num_sentences):
            core = base_sentences[i % len(base_sentences)]

            extra_parts: List[str] = []

            # Extraversion: more social context
            if ext > 3.0 and self.rng.random() < 0.7:
                word = self.rng.choice(social_words)
                extra_parts.append(
                    f"I discussed it with my {word} to understand it better."
                )

            # NFC: more analytic language
            if nfc > 3.0 and self.rng.random() < 0.8:
                word = self.rng.choice(analytic_words)
                extra_parts.append(
                    f"I tried to {word} the problem before acting."
                )

            # CRT: more explicit reconsideration
            if crt >= 3 and self.rng.random() < 0.8:
                word = self.rng.choice(insight_words)
                extra_parts.append(
                    f"After some time, I {word} that my first impression "
                    "might be misleading."
                )

            # Affect: everyone has some affect words
            if self.rng.random() < 0.6:
                word = self.rng.choice(affect_words)
                extra_parts.append(f"I felt quite {word}.")

            sentence = " ".join([core] + extra_parts)
            sentences.append(sentence)

        return " ".join(sentences)


# ---------------------------------------------------------------------
# Memory and Planning Agents for prompt construction
# ---------------------------------------------------------------------


class MemoryAgent:
    """
    Minimal Memory Agent abstraction to provide user and item context
    for LLM prompts.
    """

    def __init__(self, config: SimulationConfig):
        self.config = config

    def get_user_context(self, persona_row: pd.Series) -> str:
        """
        Build a brief textual description of the user/persona based on
        available demographics and trait scores.
        """
        parts: List[str] = []
        persona_id = persona_row.get("persona_id", None)
        if persona_id is not None:
            parts.append(f"Persona ID: {persona_id}")

        # Traits
        trait_desc: List[str] = []
        for domain, col in self.config.bfi_columns.items():
            if col in persona_row:
                trait_desc.append(
                    f"{domain}={persona_row[col]}"
                )
        if self.config.nfc_column in persona_row:
            trait_desc.append(f"NFC={persona_row[self.config.nfc_column]}")
        if self.config.crt_column in persona_row:
            trait_desc.append(f"CRT2={persona_row[self.config.crt_column]}")
        if trait_desc:
            parts.append("Trait scores: " + ", ".join(trait_desc))

        return "\n".join(parts)

    @staticmethod
    def get_item_context(items: List[TestItem]) -> str:
        """
        Describe the items/tests being administered in this block.
        """
        lines: List[str] = []
        for i, item in enumerate(items, start=1):
            lines.append(
                f"{i}. [{item.item_id}] (test={item.test_name}, type={item.item_type}): {item.item_text}"
            )
        return "\n".join(lines)


class PlanningAgent:
    """
    Minimal Planning Agent abstraction that decomposes the task into
    steps for the LLM.
    """

    def make_plan(self, admin_mode: str, items: List[TestItem]) -> str:
        """
        Construct a simple task plan for answering the given items.
        """
        mode_desc = {
            "per_item": "You will answer one item at a time.",
            "per_test": "You will answer all items from a single test.",
            "all_tests": "You will answer all items from multiple tests in one go.",
        }.get(admin_mode, "You will answer the items provided.")

        steps = [
            mode_desc,
            "1. Read the persona description and user context carefully.",
            "2. Read each item and imagine how this persona would respond.",
            "3. For Likert items, answer with an integer from 1 to 5.",
            "4. For CRT items, provide a short numeric answer.",
            "5. For open-ended writing tasks, write a detailed paragraph response.",
            "6. Return all answers in JSON format as a list of objects: "
            '{"item_id": "...", "response": "..."}',
        ]
        return "\n".join(steps)


# ---------------------------------------------------------------------
# Real LLM client using OpenAI Responses API
# ---------------------------------------------------------------------


class RealLLMClient:
    """
    Wrapper around OpenAI Responses API to answer psychometric items
    and writing tasks given constructed prompts.
    """

    def __init__(self, model: str = "gpt-5", max_output_tokens: int = 4000):
        self.model = model
        self.max_output_tokens = max_output_tokens

    def build_prompt(
        self,
        persona_prompt: str,
        user_context: str,
        item_context: str,
        plan: str,
    ) -> str:
        """
        Build the full LLM prompt from persona, memory, and plan components.
        """
        prompt_parts = [
            "You are simulating a specific persona.",
            "Persona description:",
            persona_prompt,
            "",
            "User context from memory:",
            user_context,
            "",
            "Item/test context:",
            item_context,
            "",
            "Plan/steps for this task:",
            plan,
            "",
            "Now, produce answers following the plan. "
            "IMPORTANT: Return your answers strictly as JSON in the format:",
            '[{"item_id": "<ID1>", "response": "<your answer>"}, '
            '{"item_id": "<ID2>", "response": "<your answer>"}, ...]',
        ]
        return "\n".join(prompt_parts)

    def _parse_answers(
        self, raw_text: str, items: List[TestItem]
    ) -> Dict[str, str]:
        """
        Parse JSON answers from the LLM output. Fallback: assign the full
        text to each item if parsing fails.
        """
        raw_text = raw_text.strip()
        # Try to find a JSON substring
        json_text = raw_text
        # Heuristic: if there's a '[' and ']', extract that slice
        if "[" in raw_text and "]" in raw_text:
            start = raw_text.find("[")
            end = raw_text.rfind("]") + 1
            json_text = raw_text[start:end]

        try:
            data = json.loads(json_text)
            mapping: Dict[str, str] = {}
            if isinstance(data, list):
                for entry in data:
                    if not isinstance(entry, dict):
                        continue
                    iid = entry.get("item_id") or entry.get("id")
                    resp = entry.get("response") or entry.get("answer") or ""
                    if iid is not None:
                        mapping[str(iid)] = str(resp)
            if mapping:
                return mapping
        except Exception:
            pass

        # Fallback: use the whole text for all items (or the single item)
        if len(items) == 1:
            return {items[0].item_id: raw_text}
        return {item.item_id: raw_text for item in items}

    def answer_block(
        self,
        prompt: str,
        items: List[TestItem],
    ) -> Dict[str, str]:
        """
        Call the OpenAI Responses API for a block of items and return
        a mapping item_id -> response_text.
        """
        raw = call_gpt5_with_responses_api(
            prompt=prompt,
            model=self.model,
            max_output_tokens=self.max_output_tokens,
        )
        return self._parse_answers(raw, items)


# ---------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------


class PersonaSimulator:
    """
    Core simulator that rolls out test administrations for personas using
    either the PseudoLLM or the OpenAI-backed RealLLMClient.
    """

    def __init__(self, test_bank: TestBank, config: SimulationConfig):
        """
        Initialize the simulator.

        Parameters
        ----------
        test_bank : TestBank
            Collection of test and writing items.
        config : SimulationConfig
            Simulation configuration.
        """
        self.test_bank = test_bank
        self.config = config

    @staticmethod
    def _parse_likert_from_text(text: str) -> int:
        """
        Extract a Likert response (1–5) from LLM text; default to 3 if
        parsing fails.
        """
        match = re.search(r"\b([1-5])\b", text)
        if match:
            try:
                val = int(match.group(1))
                if 1 <= val <= 5:
                    return val
            except ValueError:
                pass
        return 3

    @staticmethod
    def _score_crt_from_text(text: str, item: TestItem) -> bool:
        """
        Determine whether a CRT answer is correct based on the LLM output.
        """
        if item.correct_answer is None:
            return False
        correct_str = item.correct_answer.strip()
        if correct_str and correct_str in text:
            return True
        # Try numeric comparison
        try:
            nums = re.findall(r"-?\d+\.?\d*", text)
            if nums:
                ans_val = float(nums[0])
                corr_val = float(correct_str)
                if abs(ans_val - corr_val) < 1e-6:
                    return True
        except Exception:
            pass
        return False

    def rollout(
        self,
        personas_df: pd.DataFrame,
        params: Dict[str, Any],
        persona_prompts: pd.Series,
    ) -> SimulationResult:
        """
        Run a forward simulation (rollout) on a set of personas.

        Parameters
        ----------
        personas_df : pd.DataFrame
            DataFrame containing personas in the validation or training
            window, with trait columns required by config.
        params : Dict[str, Any]
            Simulator parameter dictionary (e.g., likert_noise, crt_slip).
        persona_prompts : pd.Series
            Series mapping persona index to full persona prompt string.

        Returns
        -------
        SimulationResult
            Structured simulation outputs.
        """
        likert_noise = float(params.get("likert_noise", 0.5))
        crt_slip = float(params.get("crt_slip", 0.2))
        random_seed = int(params.get("random_seed", self.config.random_seed))

        api_call_rows: List[Dict[str, Any]] = []

        if self.config.use_openai:
            result = self._rollout_with_openai(
                personas_df=personas_df,
                persona_prompts=persona_prompts,
                random_seed=random_seed,
                api_call_rows=api_call_rows,
            )
            result.api_calls = pd.DataFrame(api_call_rows) if api_call_rows else None
            return result

        # Fallback: use PseudoLLM (no real OpenAI calls)
        llm = PseudoLLM(
            likert_noise=likert_noise,
            crt_slip=crt_slip,
            random_seed=random_seed,
        )
        items_by_test = self.test_bank.items_by_test()

        persona_rows: List[Dict[str, Any]] = []
        item_rows: List[Dict[str, Any]] = []

        for persona_idx, (idx, row) in enumerate(personas_df.iterrows()):
            persona_id = row.get("persona_id", idx)
            bfi_traits = {
                domain: float(row[self.config.bfi_columns[domain]])
                for domain in self.config.bfi_columns
            }
            nfc_score = float(row[self.config.nfc_column])
            crt_level = int(row[self.config.crt_level_column])
            prompt = persona_prompts.loc[idx]

            # Accumulators for scale scores
            bfi_scores: Dict[str, List[int]] = {
                domain: [] for domain in self.config.bfi_columns
            }
            nfc_items: List[int] = []
            crt_corrects: List[int] = []

            writing_texts: Dict[str, str] = {}
            writing_metrics: Dict[str, Dict[str, float]] = {}

            # Persona-specific RNG for shuffling
            persona_rng = np.random.default_rng(random_seed + persona_idx)

            # Determine order of tests and items, then construct blocks
            tests_order = list(items_by_test.keys())
            if self.config.shuffle_test_order:
                persona_rng.shuffle(tests_order)

            blocks: List[List[TestItem]] = []

            if self.config.admin_mode == "per_item":
                for test_name in tests_order:
                    test_items = list(items_by_test[test_name])
                    if self.config.shuffle_items_within_tests:
                        persona_rng.shuffle(test_items)
                    for item in test_items:
                        blocks.append([item])
            elif self.config.admin_mode == "per_test":
                for test_name in tests_order:
                    test_items = list(items_by_test[test_name])
                    if self.config.shuffle_items_within_tests:
                        persona_rng.shuffle(test_items)
                    blocks.append(test_items)
            else:  # "all_tests"
                all_items: List[TestItem] = []
                for test_name in tests_order:
                    test_items = list(items_by_test[test_name])
                    if self.config.shuffle_items_within_tests:
                        persona_rng.shuffle(test_items)
                    all_items.extend(test_items)
                blocks.append(all_items)

            # Process blocks (logical API calls)
            for block_index, block_items in enumerate(blocks):
                api_call_rows.append(
                    {
                        "persona_id": persona_id,
                        "block_index": block_index,
                        "admin_mode": self.config.admin_mode,
                        "engine": "pseudo",
                        "test_names": ",".join(
                            sorted({it.test_name for it in block_items})
                        ),
                        "item_ids": [it.item_id for it in block_items],
                        "prompt": None,
                    }
                )

                for item in block_items:
                    self._simulate_item(
                        llm,
                        persona_id,
                        row,
                        bfi_traits,
                        nfc_score,
                        crt_level,
                        prompt,
                        item,
                        bfi_scores,
                        nfc_items,
                        crt_corrects,
                        item_rows,
                        writing_texts,
                        writing_metrics,
                    )

            # Aggregate scores
            persona_record: Dict[str, Any] = row.to_dict()
            persona_record["persona_id"] = persona_id

            for domain, scores in bfi_scores.items():
                if scores:
                    persona_record[f"sim_bfi_{domain}"] = float(
                        np.mean(scores)
                    )
                else:
                    persona_record[f"sim_bfi_{domain}"] = float("nan")

            persona_record["sim_nfc_score"] = (
                float(np.mean(nfc_items)) if nfc_items else float("nan")
            )
            persona_record["sim_crt2_score"] = (
                float(np.sum(crt_corrects)) if crt_corrects else float("nan")
            )

            # Add writing metrics
            for wt_id, text in writing_texts.items():
                persona_record[f"writing_{wt_id}_text"] = text
                metrics = writing_metrics.get(wt_id, {})
                for m_name, m_val in metrics.items():
                    persona_record[f"writing_{wt_id}_{m_name}"] = m_val

            persona_rows.append(persona_record)

        per_persona_df = pd.DataFrame(persona_rows)
        per_item_df = pd.DataFrame(item_rows)

        api_calls_df = pd.DataFrame(api_call_rows) if api_call_rows else None

        return SimulationResult(
            per_persona=per_persona_df,
            per_item=per_item_df,
            config=self.config,
            api_calls=api_calls_df,
        )

    def _rollout_with_openai(
        self,
        personas_df: pd.DataFrame,
        persona_prompts: pd.Series,
        random_seed: int,
        api_call_rows: List[Dict[str, Any]],
    ) -> SimulationResult:
        """
        Rollout that uses the OpenAI Responses API as the reasoning engine.
        """
        items_by_test = self.test_bank.items_by_test()
        persona_rows: List[Dict[str, Any]] = []
        item_rows: List[Dict[str, Any]] = []

        memory_agent = MemoryAgent(self.config)
        planning_agent = PlanningAgent()
        llm_client = RealLLMClient(
            model=self.config.openai_model,
            max_output_tokens=self.config.openai_max_output_tokens,
        )

        for persona_idx, (idx, row) in enumerate(personas_df.iterrows()):
            persona_id = row.get("persona_id", idx)
            bfi_traits = {
                domain: float(row[self.config.bfi_columns[domain]])
                for domain in self.config.bfi_columns
            }
            nfc_score = float(row[self.config.nfc_column])
            crt_level = int(row[self.config.crt_level_column])
            persona_prompt = persona_prompts.loc[idx]

            bfi_scores: Dict[str, List[int]] = {
                domain: [] for domain in self.config.bfi_columns
            }
            nfc_items: List[int] = []
            crt_corrects: List[int] = []
            writing_texts: Dict[str, str] = {}
            writing_metrics: Dict[str, Dict[str, float]] = {}

            persona_rng = np.random.default_rng(random_seed + persona_idx)

            tests_order = list(items_by_test.keys())
            if self.config.shuffle_test_order:
                persona_rng.shuffle(tests_order)

            blocks: List[List[TestItem]] = []

            if self.config.admin_mode == "per_item":
                for test_name in tests_order:
                    test_items = list(items_by_test[test_name])
                    if self.config.shuffle_items_within_tests:
                        persona_rng.shuffle(test_items)
                    for item in test_items:
                        blocks.append([item])
            elif self.config.admin_mode == "per_test":
                for test_name in tests_order:
                    test_items = list(items_by_test[test_name])
                    if self.config.shuffle_items_within_tests:
                        persona_rng.shuffle(test_items)
                    blocks.append(test_items)
            else:  # "all_tests"
                all_items: List[TestItem] = []
                for test_name in tests_order:
                    test_items = list(items_by_test[test_name])
                    if self.config.shuffle_items_within_tests:
                        persona_rng.shuffle(test_items)
                    all_items.extend(test_items)
                blocks.append(all_items)

            user_context = memory_agent.get_user_context(row)

            for block_index, block_items in enumerate(blocks):
                item_context = MemoryAgent.get_item_context(block_items)
                plan = planning_agent.make_plan(self.config.admin_mode, block_items)
                full_prompt = llm_client.build_prompt(
                    persona_prompt=persona_prompt,
                    user_context=user_context,
                    item_context=item_context,
                    plan=plan,
                )

                api_call_rows.append(
                    {
                        "persona_id": persona_id,
                        "block_index": block_index,
                        "admin_mode": self.config.admin_mode,
                        "engine": "openai",
                        "test_names": ",".join(
                            sorted({it.test_name for it in block_items})
                        ),
                        "item_ids": [it.item_id for it in block_items],
                        "prompt": full_prompt,
                    }
                )

                answers = llm_client.answer_block(full_prompt, block_items)

                for item in block_items:
                    answer_text = answers.get(item.item_id, "").strip()
                    record: Dict[str, Any] = {
                        "persona_id": persona_id,
                        "test_name": item.test_name,
                        "item_id": item.item_id,
                        "response_text": answer_text,
                    }

                    if item.item_type == "likert":
                        response_numeric = self._parse_likert_from_text(answer_text)
                        record["response_numeric"] = response_numeric
                        record["correct"] = None

                        if item.test_name == "BFI" and item.scale_name:
                            bfi_scores[item.scale_name].append(response_numeric)
                        elif item.test_name == "NFC":
                            nfc_items.append(response_numeric)

                    elif item.item_type == "crt":
                        correct_flag = self._score_crt_from_text(answer_text, item)
                        record["response_numeric"] = None
                        record["correct"] = int(correct_flag)
                        crt_corrects.append(int(correct_flag))

                    elif item.item_type == "open":
                        text = answer_text
                        record["response_numeric"] = None
                        record["correct"] = None
                        writing_texts[item.item_id] = text
                        writing_metrics[item.item_id] = compute_writing_metrics(text)
                    else:
                        raise ValueError(f"Unknown item_type '{item.item_type}'.")

                    item_rows.append(record)

            # Aggregate per-persona scores
            persona_record: Dict[str, Any] = row.to_dict()
            persona_record["persona_id"] = persona_id

            for domain, scores in bfi_scores.items():
                persona_record[f"sim_bfi_{domain}"] = (
                    float(np.mean(scores)) if scores else float("nan")
                )

            persona_record["sim_nfc_score"] = (
                float(np.mean(nfc_items)) if nfc_items else float("nan")
            )
            persona_record["sim_crt2_score"] = (
                float(np.sum(crt_corrects)) if crt_corrects else float("nan")
            )

            for wt_id, text in writing_texts.items():
                persona_record[f"writing_{wt_id}_text"] = text
                metrics = writing_metrics.get(wt_id, {})
                for m_name, m_val in metrics.items():
                    persona_record[f"writing_{wt_id}_{m_name}"] = m_val

            persona_rows.append(persona_record)

        per_persona_df = pd.DataFrame(persona_rows)
        per_item_df = pd.DataFrame(item_rows)

        return SimulationResult(
            per_persona=per_persona_df,
            per_item=per_item_df,
            config=self.config,
        )

    def _simulate_item(
        self,
        llm: PseudoLLM,
        persona_id: Any,
        row: pd.Series,
        bfi_traits: Dict[str, float],
        nfc_score: float,
        crt_level: int,
        persona_prompt: str,
        item: TestItem,
        bfi_scores: Dict[str, List[int]],
        nfc_items: List[int],
        crt_corrects: List[int],
        item_rows: List[Dict[str, Any]],
        writing_texts: Dict[str, str],
        writing_metrics: Dict[str, Dict[str, float]],
    ) -> None:
        """
        Simulate response to a single item and update accumulators.

        Parameters
        ----------
        llm : PseudoLLM
            Pseudo LLM instance.
        persona_id : Any
            Unique persona identifier.
        row : pd.Series
            Original persona row.
        bfi_traits : Dict[str, float]
            Big Five traits.
        nfc_score : float
            NFC score.
        crt_level : int
            CRT level.
        persona_prompt : str
            Full persona prompt text.
        item : TestItem
            Item metadata.
        bfi_scores, nfc_items, crt_corrects, item_rows, writing_texts,
        writing_metrics :
            Accumulators updated in place.
        """
        record: Dict[str, Any] = {
            "persona_id": persona_id,
            "test_name": item.test_name,
            "item_id": item.item_id,
        }

        if item.item_type == "likert":
            if item.trait == "nfc":
                trait_val = nfc_score
            else:
                # Assume BFI domain
                trait_val = float(
                    row[self.config.bfi_columns[item.trait]]  # type: ignore
                )
            response = llm.answer_likert(trait_val, item.keyed)
            record["response_text"] = str(response)
            record["response_numeric"] = response
            record["correct"] = None

            if item.test_name == "BFI" and item.scale_name:
                bfi_scores[item.scale_name].append(response)
            elif item.test_name == "NFC":
                nfc_items.append(response)

        elif item.item_type == "crt":
            ans_text, correct = llm.answer_crt(crt_level, item)
            record["response_text"] = ans_text
            record["response_numeric"] = None
            record["correct"] = int(correct)
            crt_corrects.append(int(correct))

        elif item.item_type == "open":
            text = llm.generate_writing(
                persona_prompt, bfi_traits, nfc_score, crt_level, item
            )
            record["response_text"] = text
            record["response_numeric"] = None
            record["correct"] = None
            writing_texts[item.item_id] = text
            writing_metrics[item.item_id] = compute_writing_metrics(text)
        else:
            raise ValueError(f"Unknown item_type '{item.item_type}'.")

        item_rows.append(record)


# ---------------------------------------------------------------------
# Writing metrics (behavioural alignment proxies)
# ---------------------------------------------------------------------


def compute_writing_metrics(text: str) -> Dict[str, float]:
    """
    Compute simple LIWC-inspired metrics for a writing sample.

    Parameters
    ----------
    text : str
        Writing sample text.

    Returns
    -------
    Dict[str, float]
        Dictionary of metrics: length, analytic, insight, affect.
    """
    tokens = [t.lower() for t in text.split()]
    n_tokens = len(tokens) if tokens else 1

    analytic_vocab = {
        "because",
        "therefore",
        "logic",
        "analyze",
        "structure",
        "methodical",
        "reason",
        "reasoning",
        "evidence",
    }
    insight_vocab = {
        "realized",
        "understood",
        "noticed",
        "insight",
        "reflection",
        "reflected",
        "think",
        "thought",
        "considered",
        "aware",
    }
    affect_vocab = {
        "happy",
        "glad",
        "relieved",
        "sad",
        "anxious",
        "afraid",
        "frustrated",
        "angry",
        "excited",
        "upset",
    }

    analytic_count = sum(t in analytic_vocab for t in tokens)
    insight_count = sum(t in insight_vocab for t in tokens)
    affect_count = sum(t in affect_vocab for t in tokens)

    return {
        "length": float(n_tokens),
        "analytic": analytic_count / n_tokens,
        "insight": insight_count / n_tokens,
        "affect": affect_count / n_tokens,
    }


# ---------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------


class GridSearchCalibrator:
    """
    Simple grid-search calibrator for simulator parameters.

    This class is intentionally modular so that alternative calibration
    strategies (e.g., Bayesian optimisation) can be substituted without
    changing the orchestrator.
    """

    def __init__(
        self,
        param_grid: Dict[str, List[Any]],
        random_seed: int = GLOBAL_RANDOM_SEED,
    ):
        """
        Initialize the calibrator.

        Parameters
        ----------
        param_grid : Dict[str, List[Any]]
            Mapping from parameter names to lists of candidate values.
        random_seed : int
            Random seed for reproducibility.
        """
        self.param_grid = param_grid
        self.random_seed = random_seed
        self.best_params_: Optional[Dict[str, Any]] = None
        self.best_score_: Optional[float] = None

    def fit(
        self,
        train_df: pd.DataFrame,
        simulator: PersonaSimulator,
        persona_prompts: pd.Series,
        config: SimulationConfig,
    ) -> None:
        """
        Fit (calibrate) simulator parameters on training personas.

        Parameters
        ----------
        train_df : pd.DataFrame
            Training personas with ground-truth trait scores.
        simulator : PersonaSimulator
            Simulator to evaluate.
        persona_prompts : pd.Series
            Persona prompts for each row in train_df.
        config : SimulationConfig
            Configuration specifying trait column names.

        Notes
        -----
        For each parameter combination, this method:
        - runs a rollout on the training personas,
        - computes average Pearson correlation between simulated and
          target trait scores (BFI domains, NFC, CRT2),
        - selects the parameter combination with the highest score.
        """
        # Build all combinations of parameters (lazy product, no materialisation)
        keys = sorted(self.param_grid.keys())

        best_score = -float("inf")
        best_params: Optional[Dict[str, Any]] = None

        for combo_index, combo in enumerate(
            itertools.product(*(self.param_grid[k] for k in keys))
        ):
            params = {k: v for k, v in zip(keys, combo)}
            # Deterministic but distinct seed per combo
            params["random_seed"] = self.random_seed + combo_index

            result = simulator.rollout(train_df, params, persona_prompts)
            score = self._trait_fidelity_score(
                train_df, result.per_persona, config
            )

            if score > best_score:
                best_score = score
                best_params = params

        self.best_score_ = best_score
        self.best_params_ = best_params if best_params is not None else {}

    @staticmethod
    def _trait_fidelity_score(
        true_df: pd.DataFrame,
        sim_df: pd.DataFrame,
        config: SimulationConfig,
    ) -> float:
        """
        Compute an aggregate trait fidelity score between true and simulated.

        Parameters
        ----------
        true_df : pd.DataFrame
            DataFrame with ground-truth trait scores.
        sim_df : pd.DataFrame
            DataFrame with simulated trait scores.
        config : SimulationConfig
            Configuration with trait column names.

        Returns
        -------
        float
            Average Pearson correlation across traits; NaN-safe (ignores
            NaN correlations by averaging over defined values).
        """
        metrics: List[float] = []

        # BFI domains
        for domain, col_name in config.bfi_columns.items():
            sim_col = f"sim_bfi_{domain}"
            if col_name in true_df.columns and sim_col in sim_df.columns:
                corr = pearson_corr(true_df[col_name], sim_df[sim_col])
                if not math.isnan(corr):
                    metrics.append(corr)

        # NFC
        if (
            config.nfc_column in true_df.columns
            and "sim_nfc_score" in sim_df.columns
        ):
            corr = pearson_corr(
                true_df[config.nfc_column], sim_df["sim_nfc_score"]
            )
            if not math.isnan(corr):
                metrics.append(corr)

        # CRT2 (true continuous vs simulated discrete correct count)
        if (
            config.crt_column in true_df.columns
            and "sim_crt2_score" in sim_df.columns
        ):
            corr = pearson_corr(
                true_df[config.crt_column], sim_df["sim_crt2_score"]
            )
            if not math.isnan(corr):
                metrics.append(corr)

        if not metrics:
            return -float("inf")

        return float(np.mean(metrics))


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------


class Evaluator:
    """
    Evaluate simulation outputs according to blueprint metrics:
    trait fidelity, behavioural alignment, and response consistency.
    """

    def __init__(self, config: SimulationConfig):
        """
        Initialize the evaluator.

        Parameters
        ----------
        config : SimulationConfig
            Simulation configuration for trait column names.
        """
        self.config = config

    def compute_metrics(
        self,
        sim_result: SimulationResult,
        val_df: pd.DataFrame,
        train_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        Compute evaluation metrics on validation results.

        Parameters
        ----------
        sim_result : SimulationResult
            Simulation outputs for validation personas.
        val_df : pd.DataFrame
            Validation personas (ground-truth traits).
        train_df : Optional[pd.DataFrame]
            Optional training personas (unused but kept for extensibility).

        Returns
        -------
        Dict[str, Any]
            Nested dictionary of metrics.
        """
        per_persona = sim_result.per_persona
        per_item = sim_result.per_item

        metrics: Dict[str, Any] = {
            "trait_fidelity": {},
            "behavioural_alignment": {},
            "response_consistency": {},
        }

        # ----------------- Trait fidelity -----------------
        tf = metrics["trait_fidelity"]

        # BFI domains
        for domain, col_name in self.config.bfi_columns.items():
            sim_col = f"sim_bfi_{domain}"
            if col_name in val_df.columns and sim_col in per_persona.columns:
                true_vals = val_df[col_name].astype(float)
                sim_vals = per_persona[sim_col].astype(float)
                tf[domain] = {
                    "pearson": pearson_corr(true_vals, sim_vals),
                    "spearman": spearman_corr(true_vals, sim_vals),
                }

        # NFC
        if (
            self.config.nfc_column in val_df.columns
            and "sim_nfc_score" in per_persona.columns
        ):
            true_vals = val_df[self.config.nfc_column].astype(float)
            sim_vals = per_persona["sim_nfc_score"].astype(float)
            tf["nfc"] = {
                "pearson": pearson_corr(true_vals, sim_vals),
                "spearman": spearman_corr(true_vals, sim_vals),
            }

        # CRT2: also MAE (true continuous vs simulated correct count)
        if (
            self.config.crt_column in val_df.columns
            and "sim_crt2_score" in per_persona.columns
        ):
            true_vals = val_df[self.config.crt_column].astype(float)
            sim_vals = per_persona["sim_crt2_score"].astype(float)
            mae = float(np.mean(np.abs(true_vals - sim_vals)))
            tf["crt2"] = {
                "pearson": pearson_corr(true_vals, sim_vals),
                "spearman": spearman_corr(true_vals, sim_vals),
                "mae": mae,
            }

        # Monotonicity checks (rough): correlation sign expectations
        tf["monotonicity_notes"] = (
            "Higher target traits should correspond to higher simulated "
            "scale scores; see trait_fidelity correlations."
        )

        # ----------------- Behavioural alignment -----------------
        ba = metrics["behavioural_alignment"]

        # Writing-task metrics vs traits: compute correlations
        writing_cols = [
            col for col in per_persona.columns if col.startswith("writing_")
        ]
        trait_pairs: List[Tuple[str, str]] = []

        # Pairs: (trait, writing metric col)
        for col in writing_cols:
            if col.endswith("_length") or col.endswith(
                ("_analytic", "_insight", "_affect")
            ):
                trait_pairs.append(("nfc", col))
                trait_pairs.append(("crt2", col))

        corrs: Dict[str, Any] = {}
        for trait, w_col in trait_pairs:
            if w_col not in per_persona.columns:
                continue
            if trait == "nfc":
                t_col = self.config.nfc_column
            else:
                t_col = self.config.crt_column

            if t_col not in val_df.columns:
                continue

            true_vals = val_df[t_col].astype(float)
            sim_vals = per_persona[w_col].astype(float)
            key = f"{trait}_vs_{w_col}"
            corrs[key] = {
                "pearson": pearson_corr(true_vals, sim_vals),
                "spearman": spearman_corr(true_vals, sim_vals),
            }

        ba["trait_writing_correlations"] = corrs

        # ----------------- Response consistency -----------------
        rc = metrics["response_consistency"]

        # Cronbach's alpha for BFI scales (simulated item responses)
        alpha_results: Dict[str, float] = {}
        for domain in self.config.bfi_columns.keys():
            item_ids = [
                f"BFI_{domain[0].upper()}1",
                f"BFI_{domain[0].upper()}2",
            ]
            subset = per_item[
                (per_item["test_name"] == "BFI")
                & (per_item["item_id"].isin(item_ids))
            ]
            if subset.empty:
                alpha_results[domain] = float("nan")
                continue
            pivot = subset.pivot(
                index="persona_id",
                columns="item_id",
                values="response_numeric",
            )
            pivot = pivot.dropna(axis=0, how="any")
            alpha_results[domain] = cronbach_alpha(pivot)

        rc["cronbach_alpha_bfi"] = alpha_results

        # NFC internal consistency (all NFC items together)
        subset_nfc = per_item[per_item["test_name"] == "NFC"]
        if not subset_nfc.empty:
            pivot_nfc = subset_nfc.pivot(
                index="persona_id",
                columns="item_id",
                values="response_numeric",
            )
            pivot_nfc = pivot_nfc.dropna(axis=0, how="any")
            rc["cronbach_alpha_nfc"] = cronbach_alpha(pivot_nfc)
        else:
            rc["cronbach_alpha_nfc"] = float("nan")

        # Placeholder notes for test–retest and order effects
        rc["notes"] = (
            "Test–retest reliability and order effects are not explicitly "
            "simulated here but could be assessed by repeating rollouts "
            "with the same personas under different shuffling settings."
        )

        return metrics


# ---------------------------------------------------------------------
# Data loading, network building, and holdout split
# ---------------------------------------------------------------------


def load_data(config: SimulationConfig) -> Dict[str, pd.DataFrame]:
    """
    Load required data files: synthetic personas and optional markers.

    Parameters
    ----------
    config : SimulationConfig
        Simulation configuration.

    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary with keys 'personas' and 'markers'.

    Raises
    ------
    FileNotFoundError
        If the required personas file does not exist.
    ValueError
        If the personas file is empty.
    """
    personas_path = os.path.join(DATA_DIR, config.personas_filename)
    markers_path = os.path.join(DATA_DIR, config.markers_filename)

    if not os.path.isabs(personas_path):
        personas_path = os.path.abspath(personas_path)
    if not os.path.exists(personas_path):
        raise FileNotFoundError(
            f"Personas file not found at '{personas_path}'. "
            f"Ensure PROJECT_ROOT ('{PROJECT_ROOT}') and DATA_DIR "
            f"('{DATA_DIR}') are set correctly and that the file exists."
        )

    personas_df = pd.read_csv(personas_path)
    if personas_df.empty:
        raise ValueError(f"Personas file '{personas_path}' is empty.")

    # Ensure persona_id column
    if "persona_id" not in personas_df.columns:
        personas_df["persona_id"] = np.arange(len(personas_df))

    if os.path.exists(markers_path):
        markers_df = pd.read_csv(markers_path)
    else:
        markers_df = pd.DataFrame()

    return {"personas": personas_df, "markers": markers_df}


def build_network_and_agents(
    data: Dict[str, pd.DataFrame], config: SimulationConfig
) -> Tuple[pd.DataFrame, pd.Series, TestBank]:
    """
    Build persona prompts and a trivial interaction network.

    Parameters
    ----------
    data : Dict[str, pd.DataFrame]
        Loaded data, including 'personas' and 'markers'.
    config : SimulationConfig
        Simulation configuration.

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series, TestBank]
        - Personas DataFrame (possibly augmented),
        - Series of persona prompts indexed by personas_df index,
        - TestBank instance.

    Raises
    ------
    ValueError
        If required trait columns are missing.
    """
    personas_df = data["personas"]
    markers_df = data.get("markers")

    # Validate required columns
    missing_cols = [
        col
        for col in config.bfi_columns.values()
        if col not in personas_df.columns
    ]
    if config.nfc_column not in personas_df.columns:
        missing_cols.append(config.nfc_column)
    if config.crt_column not in personas_df.columns:
        missing_cols.append(config.crt_column)

    if missing_cols:
        raise ValueError(
            "The following required trait columns are missing from "
            f"personas file: {missing_cols}. Please ensure the CSV "
            "contains these columns or adjust the configuration."
        )

    # Map continuous CRT scores to discrete levels if needed
    if config.crt_level_column not in personas_df.columns:
        personas_df[config.crt_level_column] = map_crt_total_to_level(
            personas_df[config.crt_column]
        )

    prompt_builder = PersonaPromptBuilder(
        markers_df if markers_df is not None and not markers_df.empty else None,
        rng=np.random.default_rng(config.random_seed),
    )

    persona_prompts: Dict[Any, str] = {}

    for idx, row in personas_df.iterrows():
        bfi_traits = {
            domain: float(row[col])
            for domain, col in config.bfi_columns.items()
        }
        nfc_score = float(row[config.nfc_column])
        crt_level = int(row[config.crt_level_column])

        prompt = prompt_builder.build_full_persona_prompt(
            bfi_traits,
            nfc_score,
            crt_level,
            numeric_crt_only=config.use_crt_numeric_only,
        )
        persona_prompts[idx] = prompt

    persona_prompts_series = pd.Series(persona_prompts)

    # Build test bank (currently hardcoded; can be replaced by YAML-driven)
    test_bank = build_test_bank()

    # A full interaction network is out of scope; for completeness, we
    # could construct a similarity graph here. For now, we rely only on
    # per-persona prompts but keep the design extensible.

    return personas_df, persona_prompts_series, test_bank


def holdout_split(
    personas_df: pd.DataFrame, config: SimulationConfig
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform a temporal-style holdout split of personas into train/validation.

    Parameters
    ----------
    personas_df : pd.DataFrame
        Full personas DataFrame.
    config : SimulationConfig
        Simulation configuration specifying holdout_fraction and
        optional time_column.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (train_df, val_df), with non-overlapping sets of personas.
    """
    if config.time_column and config.time_column in personas_df.columns:
        sorted_df = personas_df.sort_values(config.time_column)
    else:
        # Fallback: sort by persona_id or index as pseudo-time
        sort_col = "persona_id" if "persona_id" in personas_df.columns else None
        if sort_col:
            sorted_df = personas_df.sort_values(sort_col)
        else:
            sorted_df = personas_df.sort_index()

    n_total = len(sorted_df)
    n_val = max(1, int(round(config.holdout_fraction * n_total)))
    n_train = max(1, n_total - n_val)

    train_df = sorted_df.iloc[:n_train].reset_index(drop=True)
    val_df = sorted_df.iloc[n_train:].reset_index(drop=True)

    return train_df, val_df


# ---------------------------------------------------------------------
# Saving results
# ---------------------------------------------------------------------


def save_results(
    metrics: Dict[str, Any],
    sim_result: SimulationResult,
    config: SimulationConfig,
    calibrator: GridSearchCalibrator,
) -> None:
    """
    Save simulation results and metrics to disk.

    Parameters
    ----------
    metrics : Dict[str, Any]
        Evaluation metrics computed on validation data.
    sim_result : SimulationResult
        Simulation outputs for validation personas.
    config : SimulationConfig
        Simulation configuration.
    calibrator : GridSearchCalibrator
        Calibrator containing best parameters and scores.
    """
    output_dir = os.path.join(PROJECT_ROOT, config.output_dir)
    ensure_dir(output_dir)

    # Per-persona and per-item results
    per_persona_path = os.path.join(output_dir, "per_persona_results.csv")
    per_item_path = os.path.join(output_dir, "per_item_results.csv")
    sim_result.per_persona.to_csv(per_persona_path, index=False)
    sim_result.per_item.to_csv(per_item_path, index=False)

    # API call metadata (if available)
    if sim_result.api_calls is not None:
        api_calls_csv_path = os.path.join(output_dir, "api_calls.csv")
        sim_result.api_calls.to_csv(api_calls_csv_path, index=False)
        api_calls_jsonl_path = os.path.join(output_dir, "api_calls.jsonl")
        with open(api_calls_jsonl_path, "w", encoding="utf-8") as f_api:
            for _, row in sim_result.api_calls.iterrows():
                rec = row.to_dict()
                json.dump(rec, f_api)
                f_api.write("\n")

    # Metrics and configuration
    report = {
        "metrics": metrics,
        "best_params": calibrator.best_params_,
        "best_calibration_score": calibrator.best_score_,
        "config": {
            "admin_mode": config.admin_mode,
            "shuffle_items_within_tests": config.shuffle_items_within_tests,
            "shuffle_test_order": config.shuffle_test_order,
            "holdout_fraction": config.holdout_fraction,
            "time_column": config.time_column,
            "bfi_columns": config.bfi_columns,
            "nfc_column": config.nfc_column,
            "crt_column": config.crt_column,
            "crt_level_column": config.crt_level_column,
            "use_crt_numeric_only": config.use_crt_numeric_only,
            "use_openai": config.use_openai,
            "openai_model": config.openai_model,
            "openai_max_output_tokens": config.openai_max_output_tokens,
        },
    }

    metrics_path = os.path.join(output_dir, "metrics_and_config.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # JSONL with combined persona data and simulated outputs
    jsonl_path = os.path.join(output_dir, "per_persona_results.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for _, row in sim_result.per_persona.iterrows():
            rec = row.to_dict()
            json.dump(rec, f)
            f.write("\n")


# ---------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------


def parse_cli() -> SimulationConfig:
    """
    Parse command-line arguments into a SimulationConfig.

    Returns
    -------
    SimulationConfig
        Validated simulation configuration.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Simulate LLM-based personas, administer psychometric tests "
            "and writing tasks, calibrate simulator parameters, and "
            "evaluate fidelity and reliability."
        )
    )

    parser.add_argument(
        "--personas-filename",
        type=str,
        default="synthetic_personas.csv",
        help="File name of synthetic personas CSV (relative to DATA_DIR).",
    )
    parser.add_argument(
        "--markers-filename",
        type=str,
        default="serapio_goldberg_markers.csv",
        help="File name of Serapio/Goldberg markers CSV (relative to "
        "DATA_DIR).",
    )
    parser.add_argument(
        "--admin-mode",
        type=str,
        default="per_test",
        choices=["per_item", "per_test", "all_tests"],
        help="Administration granularity for API calls.",
    )
    parser.add_argument(
        "--shuffle-items-within-tests",
        action="store_true",
        help="Shuffle item order within each test.",
    )
    parser.add_argument(
        "--shuffle-test-order",
        action="store_true",
        help="Shuffle the order of tests.",
    )
    parser.add_argument(
        "--holdout-fraction",
        type=float,
        default=0.2,
        help="Fraction of personas used for validation.",
    )
    parser.add_argument(
        "--time-column",
        type=str,
        default=None,
        help="Optional column for temporal ordering (if present in "
        "personas CSV).",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=GLOBAL_RANDOM_SEED,
        help="Random seed for deterministic simulations.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="simulation_outputs",
        help="Output directory relative to PROJECT_ROOT.",
    )
    parser.add_argument(
        "--use-crt-numeric-only",
        action="store_true",
        help="Use numeric-only CRT specification rather than descriptive "
        "sentence.",
    )
    parser.add_argument(
        "--use-openai",
        action="store_true",
        help="Use OpenAI LLM via Responses API instead of PseudoLLM.",
    )
    parser.add_argument(
        "--openai-model",
        type=str,
        default="gpt-5",
        help="OpenAI model to use with the Responses API (default: gpt-5).",
    )
    parser.add_argument(
        "--openai-max-output-tokens",
        type=int,
        default=4000,
        help="Maximum number of output tokens for Responses API calls.",
    )

    args = parser.parse_args()

    config = SimulationConfig(
        personas_filename=args.personas_filename,
        markers_filename=args.markers_filename,
        admin_mode=args.admin_mode,
        shuffle_items_within_tests=args.shuffle_items_within_tests,
        shuffle_test_order=args.shuffle_test_order,
        holdout_fraction=args.holdout_fraction,
        time_column=args.time_column,
        random_seed=args.random_seed,
        output_dir=args.output_dir,
        use_crt_numeric_only=args.use_crt_numeric_only,
        use_openai=args.use_openai,
        openai_model=args.openai_model,
        openai_max_output_tokens=args.openai_max_output_tokens,
    )
    config.validate()
    return config


# ---------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------


def main() -> None:
    """
    Orchestrate the full simulation pipeline.

    Steps
    -----
    1. parse_cli()
    2. load_data()
    3. build_network_and_agents()
    4. holdout_split()
    5. calibrator.fit()
    6. simulator.rollout() on validation set
    7. evaluator.compute_metrics()
    8. save_results()
    """
    # 1. CLI and configuration
    config = parse_cli()

    # Override global random seeds for this run
    random.seed(config.random_seed)
    np.random.seed(config.random_seed)

    # 2. Load data
    data = load_data(config)

    # 3. Build persona prompts and test bank
    personas_df, persona_prompts, test_bank = build_network_and_agents(
        data, config
    )

    # 4. Holdout split
    train_df, val_df = holdout_split(personas_df, config)
    train_prompts = persona_prompts.loc[train_df.index]
    val_prompts = persona_prompts.loc[val_df.index]

    # 5. Calibration (only meaningful for PseudoLLM; for OpenAI we still
    # run with default params but skip grid search adjustments to API.)
    simulator = PersonaSimulator(test_bank=test_bank, config=config)

    param_grid = {
        "likert_noise": [0.3, 0.5, 0.7],
        "crt_slip": [0.1, 0.2, 0.3],
    }
    calibrator = GridSearchCalibrator(
        param_grid=param_grid, random_seed=config.random_seed
    )

    if not config.use_openai:
        calibrator.fit(train_df, simulator, train_prompts, config)
        if calibrator.best_params_ is None:
            raise RuntimeError(
                "Calibration failed to identify any valid parameter set."
            )
        best_params = calibrator.best_params_
    else:
        # When using OpenAI, we do not tune likert_noise/crt_slip; keep defaults.
        best_params = {
            "likert_noise": 0.5,
            "crt_slip": 0.2,
            "random_seed": config.random_seed,
        }
        calibrator.best_params_ = best_params
        calibrator.best_score_ = None

    # 6. Forward simulation on validation window
    val_result = simulator.rollout(val_df, best_params, val_prompts)

    # 7. Evaluation
    evaluator = Evaluator(config)
    metrics = evaluator.compute_metrics(val_result, val_df, train_df)

    # 8. Save results
    save_results(metrics, val_result, config, calibrator)


# Execute main for both direct execution and sandbox wrapper invocation
main()