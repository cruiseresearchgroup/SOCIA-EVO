#!/usr/bin/env python3
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
import yaml
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
    # Persona prompt style for BFI portion: granular_serapio, coarse_numeric, coarse_descriptive
    persona_prompt_style: str = "granular_serapio"

    def validate(self) -> None:
        """
        Validate configuration values.
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
        if not self.bfi_columns:
            raise ValueError("bfi_columns mapping must not be empty.")
        if self.persona_prompt_style not in {
            "granular_serapio",
            "coarse_numeric",
            "coarse_descriptive",
        }:
            raise ValueError(
                "persona_prompt_style must be one of "
                "'granular_serapio', 'coarse_numeric', 'coarse_descriptive'."
            )


@dataclass
class Persona:
    """
    Structured representation of a single persona, derived from a row of
    synthetic_personas.csv.
    """

    persona_id: Any
    bfi_traits: Dict[str, float]
    nfc_score: float
    crt2_score: float
    crt2_level: int
    demographics: Dict[str, Any]

    @classmethod
    def from_series(cls, row: pd.Series, config: SimulationConfig) -> "Persona":
        """
        Build a Persona instance from a pandas Series using the schema in
        SimulationConfig.
        """
        persona_id = row.get("persona_id", None)
        bfi_traits = {
            domain: float(row[col])
            for domain, col in config.bfi_columns.items()
        }
        nfc_score = float(row[config.nfc_column])
        crt2_score = float(row[config.crt_column])
        crt2_level = int(row[config.crt_level_column])

        demo_cols = [
            "Country",
            "Employment Status",
            "Occupation_cat",
            "sex",
            "Highest education level completed",
            "Ethnicity Simplified",
            "Age",
        ]
        demographics: Dict[str, Any] = {}
        for col in demo_cols:
            if col in row:
                demographics[col] = row[col]

        return cls(
            persona_id=persona_id,
            bfi_traits=bfi_traits,
            nfc_score=nfc_score,
            crt2_score=crt2_score,
            crt2_level=crt2_level,
            demographics=demographics,
        )


@dataclass
class TestItem:
    """
    Single test item (question or writing task).
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
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TestBank:
    """
    Collection of all psychometric tests and writing tasks.
    """

    items: List[TestItem]

    def items_by_test(self) -> Dict[str, List[TestItem]]:
        grouped: Dict[str, List[TestItem]] = {}
        for item in self.items:
            grouped.setdefault(item.test_name, []).append(item)
        return grouped


@dataclass
class SimulationResult:
    """
    Container for simulation outputs.
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

    responses_kwargs: Dict[str, Any] = {
        "model": model,
        "input": [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": prompt}],
            }
        ],
        "max_output_tokens": max_output_tokens,
    }

    try:
        resp = client.responses.create(**responses_kwargs)
    except Exception as exc:
        warnings.warn(
            f"OpenAI Responses API call failed: {exc}. Returning empty string.",
            RuntimeWarning,
        )
        return ""

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
    os.makedirs(path, exist_ok=True)


def pearson_corr(x: pd.Series, y: pd.Series) -> float:
    if x.size == 0 or y.size == 0:
        return float("nan")
    if x.nunique() <= 1 or y.nunique() <= 1:
        return float("nan")
    return float(x.corr(y))


def spearman_corr(x: pd.Series, y: pd.Series) -> float:
    if x.size == 0 or y.size == 0:
        return float("nan")
    return pearson_corr(x.rank(method="average"), y.rank(method="average"))


def cronbach_alpha(df: pd.DataFrame) -> float:
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
    Map continuous CRT total scores into discrete 0–4 levels using quintiles.
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
        self.markers_df = markers_df
        self.rng = rng if rng is not None else np.random.default_rng(GLOBAL_RANDOM_SEED)
        self._validate_markers_df()

    def _validate_markers_df(self) -> None:
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
        Map a 1–5 scale score to a 1–9 discrete level using
        level = 2 * (score - 1) + 1.
        """
        if score < 1.0 or score > 5.0:
            raise ValueError(
                f"Score for 1–5 mapping must be in [1, 5], got {score}."
            )
        level = int(round(2 * (score - 1.0) + 1.0))
        return max(1, min(9, level))

    def _sample_markers(self, domain: str, pole: str, n: int = 3) -> List[str]:
        """
        Sample up to n markers for a given Big Five domain and pole
        ('low' or 'high') from the Serapio/Goldberg markers file if
        available, otherwise use a small default set.
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

    def build_bfi_granular_prompt(self, traits: Dict[str, float]) -> str:
        """
        Granular 1–9 level BFI prompt using Serapio/Goldberg markers.
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
            else:
                low = ", ".join(self._sample_markers(domain, "low", n=1))
                high = ", ".join(self._sample_markers(domain, "high", n=1))
            descriptor = template.format(
                low_adjective=low if low else "low on this trait",
                high_adjective=high if high else "high on this trait",
            )
            domain_label = domain.capitalize()
            segments.append(f"In terms of {domain_label}, I am {descriptor}.")

        joined = " ".join(segments)
        return (
            "For the following tasks, respond as a person described as: "
            f"\"{joined}\""
        )

    def build_bfi_coarse_numeric_prompt(self, traits: Dict[str, float]) -> str:
        """
        Coarse three-bin numeric BFI prompt.

        Mapping:
            low: score <= 2.5
            neutral: 2.5 < score < 3.5
            high: score >= 3.5
        """
        segments: List[str] = []
        for domain, score in traits.items():
            if score <= 2.5:
                polarity = "low"
            elif score >= 3.5:
                polarity = "high"
            else:
                polarity = "neutral"
            trait_name = domain.lower()
            segments.append(
                f"You are a person with {polarity} {trait_name}."
            )
        return " ".join(segments)

    def build_bfi_coarse_descriptive_prompt(self, traits: Dict[str, float]) -> str:
        """
        Coarse three-bin descriptive BFI prompt using marker adjectives.
        """
        descriptors: List[str] = []

        for domain, score in traits.items():
            if score <= 2.5:
                # Emphasize low pole
                descriptors.extend(self._sample_markers(domain, "low", n=3))
            elif score >= 3.5:
                # Emphasize high pole
                descriptors.extend(self._sample_markers(domain, "high", n=3))
            else:
                # Neutral: mix of both poles
                descriptors.extend(self._sample_markers(domain, "low", n=1))
                descriptors.extend(self._sample_markers(domain, "high", n=1))

        # Deduplicate while preserving order
        seen: Dict[str, None] = {}
        unique_descriptors = [
            d for d in descriptors if not (d in seen or seen.setdefault(d, None))
        ]

        if not unique_descriptors:
            return (
                "You are a person whose personality traits are broadly average "
                "across the Big Five dimensions."
            )

        descriptor_list = ", ".join(unique_descriptors)
        return f"You are a person who is {descriptor_list}."

    def build_nfc_prompt(self, nfc_score: float) -> str:
        """
        Granular 9-level Need for Cognition specification based on
        high/low descriptors and the 1–9 qualifier scale.
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
        else:
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
        Construct CRT2-style thinking-style prompt, either numeric-only
        or descriptive.
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
        persona: Persona,
        style: str = "granular_serapio",
        numeric_crt_only: bool = False,
    ) -> str:
        """
        Build the full persona prompt (BFI + NFC + CRT) for a given
        Persona instance.
        """
        if style == "granular_serapio":
            bfi_prompt = self.build_bfi_granular_prompt(persona.bfi_traits)
        elif style == "coarse_numeric":
            bfi_prompt = self.build_bfi_coarse_numeric_prompt(persona.bfi_traits)
        elif style == "coarse_descriptive":
            bfi_prompt = self.build_bfi_coarse_descriptive_prompt(persona.bfi_traits)
        else:
            raise ValueError(f"Unknown persona prompt style: {style}")

        nfc_prompt = self.build_nfc_prompt(persona.nfc_score)
        crt_prompt = self.build_crt_prompt(
            persona.crt2_level, numeric_only=numeric_crt_only
        )
        return " ".join([bfi_prompt, nfc_prompt, crt_prompt])


# ---------------------------------------------------------------------
# YAML-based Test bank construction
# ---------------------------------------------------------------------


def _load_yaml_items_file(rel_filename: str) -> Any:
    """
    Load a YAML file from DATA_DIR and return its raw contents.
    """
    path = os.path.join(DATA_DIR, rel_filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Required YAML file '{rel_filename}' not found in DATA_DIR '{DATA_DIR}'."
        )
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data


def _coerce_items_list(yaml_data: Any, filename: str) -> List[Dict[str, Any]]:
    """
    Coerce YAML contents into a list of item dictionaries.
    """
    if isinstance(yaml_data, dict):
        if "items" in yaml_data and isinstance(yaml_data["items"], list):
            return yaml_data["items"]
        if "questions" in yaml_data and isinstance(yaml_data["questions"], list):
            return yaml_data["questions"]
        if "tasks" in yaml_data and isinstance(yaml_data["tasks"], list):
            return yaml_data["tasks"]
        raise ValueError(
            f"YAML file '{filename}' must contain an 'items', 'questions', "
            "or 'tasks' list."
        )
    if isinstance(yaml_data, list):
        return yaml_data
    raise ValueError(
        f"YAML file '{filename}' must be a list or contain an 'items' list."
    )


def load_likert_test_from_yaml(
    filename: str,
    default_test_name: str,
    default_trait: Optional[str] = None,
) -> List[TestItem]:
    """
    Load a Likert-scale questionnaire from a YAML file into TestItem
    objects. Validates keyed directions and, for BFI, enforces that
    items map to known Big Five domains.
    """
    yaml_data = _load_yaml_items_file(filename)
    items_raw = _coerce_items_list(yaml_data, filename)
    items: List[TestItem] = []

    for entry in items_raw:
        item_id = str(
            entry.get("id")
            or entry.get("item_id")
            or entry.get("name")
            or entry.get("code")
        )
        if not item_id:
            raise ValueError(
                f"Missing 'id'/'item_id' in entry of '{filename}': {entry}"
            )
        text = entry.get("text") or entry.get("prompt") or entry.get("question")
        if not text:
            raise ValueError(
                f"Missing 'text'/'prompt' in entry for item '{item_id}' "
                f"in '{filename}'."
            )

        test_name = entry.get("test_name") or default_test_name
        scale_name = (
            entry.get("scale_name")
            or entry.get("domain")
            or entry.get("scale")
            or default_trait
        )

        trait = entry.get("trait") or scale_name or default_trait

        if "keyed" in entry:
            keyed = int(entry["keyed"])
        else:
            reverse_flag = bool(
                entry.get("reverse_scored")
                or entry.get("reverse_keyed")
                or entry.get("reversed")
            )
            keyed = -1 if reverse_flag else 1

        if default_test_name.lower() in {"bfi", "nfc"}:
            if keyed not in {-1, 1}:
                raise ValueError(
                    f"Likert item '{item_id}' in '{filename}' must have "
                    f"keyed in {{-1, 1}}, got {keyed}."
                )

        # Enforce Big Five domain mapping for BFI items
        if default_test_name.lower() == "bfi":
            valid_domains = {
                "extraversion",
                "agreeableness",
                "conscientiousness",
                "neuroticism",
                "openness",
            }
            if trait is None:
                raise ValueError(
                    f"BFI item '{item_id}' in '{filename}' is missing a "
                    "'trait'/'domain' field."
                )
            trait_norm = str(trait).strip().lower()
            if trait_norm not in valid_domains:
                raise ValueError(
                    f"BFI item '{item_id}' in '{filename}' has trait/domain "
                    f"'{trait}', expected one of {sorted(valid_domains)}."
                )
            trait = trait_norm
            scale_name = trait_norm

        # For NFC-18 we force a single 'nfc' trait label
        if default_test_name.lower() == "nfc":
            trait = "nfc"
            scale_name = "nfc"

        metadata = {
            k: v
            for k, v in entry.items()
            if k
            not in {
                "id",
                "item_id",
                "text",
                "prompt",
                "question",
                "test_name",
                "scale_name",
                "domain",
                "scale",
                "trait",
                "keyed",
                "reverse_scored",
                "reverse_keyed",
                "reversed",
            }
        }

        items.append(
            TestItem(
                item_id=item_id,
                test_name=str(test_name),
                scale_name=str(scale_name) if scale_name is not None else None,
                item_text=str(text),
                trait=str(trait) if trait is not None else None,
                keyed=keyed,
                item_type="likert",
                metadata=metadata,
            )
        )

    return items


def load_crt_test_from_yaml(
    filename: str,
    default_test_name: str,
    default_trait: Optional[str] = None,
) -> List[TestItem]:
    """
    Load CRT-style items (numeric response, correct answer) from YAML
    into TestItem objects.
    """
    yaml_data = _load_yaml_items_file(filename)
    items_raw = _coerce_items_list(yaml_data, filename)
    items: List[TestItem] = []

    for entry in items_raw:
        item_id = str(
            entry.get("id")
            or entry.get("item_id")
            or entry.get("name")
            or entry.get("code")
        )
        if not item_id:
            raise ValueError(
                f"Missing 'id'/'item_id' in entry of '{filename}': {entry}"
            )
        text = entry.get("text") or entry.get("prompt") or entry.get("question")
        if not text:
            raise ValueError(
                f"Missing 'text'/'prompt' in entry for item '{item_id}' "
                f"in '{filename}'."
            )

        test_name = entry.get("test_name") or default_test_name
        scale_name = (
            entry.get("scale_name")
            or entry.get("domain")
            or entry.get("scale")
            or default_trait
        )
        trait = entry.get("trait") or scale_name or default_trait

        correct_answer = (
            entry.get("correct_answer")
            or entry.get("answer")
            or entry.get("solution")
        )
        if correct_answer is None or str(correct_answer).strip() == "":
            raise ValueError(
                f"Missing 'correct_answer'/'answer' for CRT item '{item_id}' "
                f"in '{filename}'."
            )

        lure_answers = (
            entry.get("lure_answers")
            or entry.get("lures")
            or entry.get("incorrect_answers")
            or entry.get("incorrect_options")
        )
        if lure_answers is not None and not isinstance(lure_answers, list):
            lure_answers = [lure_answers]

        metadata = {
            k: v
            for k, v in entry.items()
            if k
            not in {
                "id",
                "item_id",
                "text",
                "prompt",
                "question",
                "test_name",
                "scale_name",
                "domain",
                "scale",
                "trait",
                "correct_answer",
                "answer",
                "solution",
                "lure_answers",
                "lures",
                "incorrect_answers",
                "incorrect_options",
            }
        }

        items.append(
            TestItem(
                item_id=item_id,
                test_name=str(test_name),
                scale_name=str(scale_name) if scale_name is not None else None,
                item_text=str(text),
                trait=str(trait) if trait is not None else None,
                keyed=0,
                item_type="crt",
                correct_answer=str(correct_answer),
                lure_answers=[str(x) for x in lure_answers] if lure_answers else None,
                metadata=metadata,
            )
        )

    return items


def load_writing_tasks_from_yaml(
    filename: str,
    default_test_name: str = "Writing",
) -> List[TestItem]:
    """
    Load open-ended writing tasks from YAML into TestItem objects.
    """
    yaml_data = _load_yaml_items_file(filename)
    items_raw = _coerce_items_list(yaml_data, filename)
    items: List[TestItem] = []

    for entry in items_raw:
        item_id = str(
            entry.get("id")
            or entry.get("item_id")
            or entry.get("name")
            or entry.get("code")
        )
        if not item_id:
            raise ValueError(
                f"Missing 'id'/'item_id' in entry of '{filename}': {entry}"
            )
        text = entry.get("text") or entry.get("prompt") or entry.get("instruction")
        if not text:
            raise ValueError(
                f"Missing 'text'/'prompt'/'instruction' for writing task "
                f"'{item_id}' in '{filename}'."
            )

        test_name = entry.get("test_name") or default_test_name
        scale_name = entry.get("scale_name") or entry.get("domain") or None
        trait = entry.get("trait")

        metadata = {
            k: v
            for k, v in entry.items()
            if k
            not in {
                "id",
                "item_id",
                "text",
                "prompt",
                "instruction",
                "test_name",
                "scale_name",
                "domain",
                "trait",
            }
        }

        items.append(
            TestItem(
                item_id=item_id,
                test_name=str(test_name),
                scale_name=str(scale_name) if scale_name is not None else None,
                item_text=str(text),
                trait=str(trait) if trait is not None else None,
                keyed=0,
                item_type="open",
                metadata=metadata,
            )
        )

    return items


def build_test_bank() -> TestBank:
    """
    Construct the psychometric test bank and writing tasks from YAML
    specifications in DATA_DIR.
    """
    required_yaml_files = [
        "bfi10_items.yaml",
        "nfc18_items.yaml",
        "crt2_items.yaml",
        "bcrt_items.yaml",
        "writing_tasks.yaml",
    ]
    missing_files = [
        fname
        for fname in required_yaml_files
        if not os.path.exists(os.path.join(DATA_DIR, fname))
    ]
    if missing_files:
        raise FileNotFoundError(
            "The following required YAML files are missing in "
            f"DATA_DIR '{DATA_DIR}': {missing_files}. "
            "Please ensure all psychometric and writing task specifications "
            "are present."
        )

    items: List[TestItem] = []
    # BFI-10 from YAML
    items.extend(load_likert_test_from_yaml("bfi10_items.yaml", default_test_name="BFI"))
    # NFC-18 from YAML
    items.extend(
        load_likert_test_from_yaml(
            "nfc18_items.yaml", default_test_name="NFC", default_trait="nfc"
        )
    )
    # CRT2 items
    items.extend(
        load_crt_test_from_yaml(
            "crt2_items.yaml", default_test_name="CRT2", default_trait="crt2"
        )
    )
    # bCRT items
    items.extend(
        load_crt_test_from_yaml(
            "bcrt_items.yaml", default_test_name="bCRT", default_trait="crt2"
        )
    )
    # Writing tasks
    items.extend(load_writing_tasks_from_yaml("writing_tasks.yaml", "Writing"))

    return TestBank(items=items)


# ---------------------------------------------------------------------
# Pseudo LLM simulator
# ---------------------------------------------------------------------


class PseudoLLM:
    """
    Simple, fully deterministic pseudo-LLM.
    """

    def __init__(
        self,
        likert_noise: float = 0.5,
        crt_slip: float = 0.2,
        random_seed: int = GLOBAL_RANDOM_SEED,
    ):
        self.likert_noise = likert_noise
        self.crt_slip = crt_slip
        self.rng = np.random.default_rng(random_seed)

    @staticmethod
    def _latent_from_trait(trait_value: float) -> float:
        """
        Latent score centered around the 1–5 Likert midpoint (3).
        """
        return trait_value - 3.0

    def answer_likert(self, trait_value: float, keyed: int) -> int:
        """
        Generate a 1–5 Likert response aligned with the instrument's
        direction. Reverse-scoring is applied explicitly using
        response = 6 - response when keyed == -1.
        """
        latent = self._latent_from_trait(trait_value)
        noisy = latent + self.rng.normal(0.0, self.likert_noise)
        mapped = 3.0 + noisy
        response = int(round(mapped))
        response = max(1, min(5, response))

        # Explicit reverse scoring when keyed == -1
        if keyed == -1:
            response = 6 - response

        return response

    def answer_crt(self, crt_level: int, item: TestItem) -> Tuple[str, int]:
        if item.correct_answer is None:
            raise ValueError("CRT item must have a correct_answer defined.")

        p_correct_base = 0.1
        p_correct = min(
            1.0 - self.crt_slip,
            p_correct_base + (crt_level / 4.0) * (0.9 - self.crt_slip),
        )
        if self.rng.random() < p_correct:
            return item.correct_answer, 1

        if item.lure_answers:
            answer = str(
                self.rng.choice(
                    item.lure_answers, size=1, replace=True
                )[0]
            )
        else:
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

            if ext > 3.0 and self.rng.random() < 0.7:
                word = self.rng.choice(social_words)
                extra_parts.append(
                    f"I discussed it with my {word} to understand it better."
                )

            if nfc > 3.0 and self.rng.random() < 0.8:
                word = self.rng.choice(analytic_words)
                extra_parts.append(
                    f"I tried to {word} the problem before acting."
                )

            if crt >= 3 and self.rng.random() < 0.8:
                word = self.rng.choice(insight_words)
                extra_parts.append(
                    f"After some time, I {word} that my first impression "
                    "might be misleading."
                )

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
        parts: List[str] = []
        persona_id = persona_row.get("persona_id", None)
        if persona_id is not None:
            parts.append(f"Persona ID: {persona_id}")

        # Demographics
        demo_cols = [
            "Country",
            "Employment Status",
            "Occupation_cat",
            "sex",
            "Highest education level completed",
            "Ethnicity Simplified",
            "Age",
        ]
        demo_desc: List[str] = []
        for col in demo_cols:
            if col in persona_row and pd.notna(persona_row[col]):
                demo_desc.append(f"{col}={persona_row[col]}")
        if demo_desc:
            parts.append("Demographics: " + ", ".join(demo_desc))

        # Traits
        trait_desc: List[str] = []
        for domain, col in self.config.bfi_columns.items():
            if col in persona_row:
                trait_desc.append(f"{domain}={persona_row[col]}")
        if self.config.nfc_column in persona_row:
            trait_desc.append(f"NFC={persona_row[self.config.nfc_column]}")
        if self.config.crt_column in persona_row:
            trait_desc.append(f"CRT2={persona_row[self.config.crt_column]}")
        if trait_desc:
            parts.append("Trait scores: " + ", ".join(trait_desc))

        return "\n".join(parts)

    @staticmethod
    def get_item_context(items: List[TestItem]) -> str:
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
        raw_text = raw_text.strip()
        json_text = raw_text
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

        if len(items) == 1:
            return {items[0].item_id: raw_text}
        return {item.item_id: raw_text for item in items}

    def answer_block(
        self,
        prompt: str,
        items: List[TestItem],
    ) -> Dict[str, str]:
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
        self.test_bank = test_bank
        self.config = config

    @staticmethod
    def _parse_likert_from_text(text: str) -> int:
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
        if item.correct_answer is None:
            return False
        correct_str = item.correct_answer.strip()
        if correct_str and correct_str in text:
            return True
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
        Roll out a full set of test administrations for the given personas
        using either a pseudo-LLM or the OpenAI-backed LLM.
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
            return result

        llm = PseudoLLM(
            likert_noise=likert_noise,
            crt_slip=crt_slip,
            random_seed=random_seed,
        )
        items_by_test = self.test_bank.items_by_test()

        memory_agent = MemoryAgent(self.config)
        planning_agent = PlanningAgent()
        prompt_builder_client = RealLLMClient(
            model=self.config.openai_model,
            max_output_tokens=self.config.openai_max_output_tokens,
        )

        persona_rows: List[Dict[str, Any]] = []
        item_rows: List[Dict[str, Any]] = []

        for persona_idx, (idx, row) in enumerate(personas_df.iterrows()):
            persona_obj = Persona.from_series(row, self.config)
            persona_id = persona_obj.persona_id if persona_obj.persona_id is not None else idx
            bfi_traits = persona_obj.bfi_traits
            nfc_score = persona_obj.nfc_score
            crt_level = persona_obj.crt2_level
            prompt = persona_prompts.loc[idx]
            user_context = memory_agent.get_user_context(row)

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
            else:
                all_items: List[TestItem] = []
                for test_name in tests_order:
                    test_items = list(items_by_test[test_name])
                    if self.config.shuffle_items_within_tests:
                        persona_rng.shuffle(test_items)
                    all_items.extend(test_items)
                blocks.append(all_items)

            for block_index, block_items in enumerate(blocks):
                item_context = MemoryAgent.get_item_context(block_items)
                plan = planning_agent.make_plan(self.config.admin_mode, block_items)
                full_prompt = prompt_builder_client.build_prompt(
                    persona_prompt=prompt,
                    user_context=user_context,
                    item_context=item_context,
                    plan=plan,
                )

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
                        "prompt": full_prompt,
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

            for wt_id, text in writing_texts.items():
                persona_record[f"writing_{wt_id}_text"] = text
                metrics_ = writing_metrics.get(wt_id, {})
                for m_name, m_val in metrics_.items():
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
        Rollout using the real OpenAI LLM through the Responses API.
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
            persona_obj = Persona.from_series(row, self.config)
            persona_id = persona_obj.persona_id if persona_obj.persona_id is not None else idx
            bfi_traits = persona_obj.bfi_traits
            nfc_score = persona_obj.nfc_score
            crt_level = persona_obj.crt2_level
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
            else:
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
                        "scale_name": item.scale_name,
                        "trait": item.trait,
                        "item_type": item.item_type,
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
                        if item.metadata:
                            record["writing_condition"] = item.metadata.get("condition")
                            record["writing_tags"] = item.metadata.get("tags")
                        writing_texts[item.item_id] = text
                        metrics_ = compute_writing_metrics(text)
                        writing_metrics[item.item_id] = metrics_
                        # Store metrics on per-item record as well
                        for m_name, m_val in metrics_.items():
                            record[f"writing_{m_name}"] = m_val
                    else:
                        raise ValueError(f"Unknown item_type '{item.item_type}'.")

                    item_rows.append(record)

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
                metrics_ = writing_metrics.get(wt_id, {})
                for m_name, m_val in metrics_.items():
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
        record: Dict[str, Any] = {
            "persona_id": persona_id,
            "test_name": item.test_name,
            "item_id": item.item_id,
            "scale_name": item.scale_name,
            "trait": item.trait,
            "item_type": item.item_type,
        }

        if item.item_type == "likert":
            if item.trait == "nfc":
                trait_val = nfc_score
            else:
                trait_key = item.trait
                if trait_key is None:
                    raise ValueError(
                        f"Likert item '{item.item_id}' missing trait mapping."
                    )
                if trait_key not in self.config.bfi_columns:
                    raise ValueError(
                        f"Likert item '{item.item_id}' references unknown "
                        f"trait '{trait_key}'. Expected one of "
                        f"{sorted(self.config.bfi_columns.keys())} or 'nfc'."
                    )
                trait_val = float(row[self.config.bfi_columns[trait_key]])
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
            if item.metadata:
                record["writing_condition"] = item.metadata.get("condition")
                record["writing_tags"] = item.metadata.get("tags")
            writing_texts[item.item_id] = text
            metrics_ = compute_writing_metrics(text)
            writing_metrics[item.item_id] = metrics_
            for m_name, m_val in metrics_.items():
                record[f"writing_{m_name}"] = m_val
        else:
            raise ValueError(f"Unknown item_type '{item.item_type}'.")

        item_rows.append(record)


# ---------------------------------------------------------------------
# Writing metrics (behavioural alignment proxies)
# ---------------------------------------------------------------------


def compute_writing_metrics(text: str) -> Dict[str, float]:
    """
    Compute simple proxy metrics for writing behaviour, intended as
    stand-ins for LIWC-like analytic, insight, and affect scores.
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
    """

    def __init__(
        self,
        param_grid: Dict[str, List[Any]],
        random_seed: int = GLOBAL_RANDOM_SEED,
    ):
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
        Perform grid search over simulator parameters, selecting the set
        that maximises trait fidelity on the training personas.
        """
        keys = sorted(self.param_grid.keys())

        best_score = -float("inf")
        best_params: Optional[Dict[str, Any]] = None

        for combo_index, combo in enumerate(
            itertools.product(*(self.param_grid[k] for k in keys))
        ):
            params = {k: v for k, v in zip(keys, combo)}
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
        Compute a scalar trait fidelity score by averaging Pearson
        correlations between target trait scores and simulated scores,
        aligning rows by persona_id.
        """
        if "persona_id" not in true_df.columns or "persona_id" not in sim_df.columns:
            raise ValueError(
                "Both true_df and sim_df must contain a 'persona_id' column "
                "for trait fidelity scoring."
            )

        merged = true_df.merge(sim_df, on="persona_id", how="inner")
        if merged.empty:
            return -float("inf")

        metrics: List[float] = []

        for domain, col_name in config.bfi_columns.items():
            sim_col = f"sim_bfi_{domain}"
            if col_name in merged.columns and sim_col in merged.columns:
                corr = pearson_corr(merged[col_name].astype(float), merged[sim_col].astype(float))
                if not math.isnan(corr):
                    metrics.append(corr)

        if (
            config.nfc_column in merged.columns
            and "sim_nfc_score" in merged.columns
        ):
            corr = pearson_corr(
                merged[config.nfc_column].astype(float),
                merged["sim_nfc_score"].astype(float),
            )
            if not math.isnan(corr):
                metrics.append(corr)

        if (
            config.crt_column in merged.columns
            and "sim_crt2_score" in merged.columns
        ):
            corr = pearson_corr(
                merged[config.crt_column].astype(float),
                merged["sim_crt2_score"].astype(float),
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
    Evaluate simulation outputs according to blueprint metrics.
    """

    def __init__(self, config: SimulationConfig):
        self.config = config

    def compute_metrics(
        self,
        sim_result: SimulationResult,
        val_df: pd.DataFrame,
        train_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        Compute trait fidelity, behavioural alignment, and response
        consistency metrics for the validation personas.
        """
        per_persona = sim_result.per_persona
        per_item = sim_result.per_item

        if "persona_id" not in val_df.columns or "persona_id" not in per_persona.columns:
            raise ValueError(
                "Both validation DataFrame and per_persona results must "
                "contain a 'persona_id' column."
            )

        # Align validation data and simulated per-persona data by persona_id
        merged_pp = val_df.merge(per_persona, on="persona_id", how="inner")

        metrics: Dict[str, Any] = {
            "trait_fidelity": {},
            "behavioural_alignment": {},
            "response_consistency": {},
        }

        # ----------------- Trait fidelity -----------------
        tf = metrics["trait_fidelity"]

        for domain, col_name in self.config.bfi_columns.items():
            sim_col = f"sim_bfi_{domain}"
            if col_name in merged_pp.columns and sim_col in merged_pp.columns:
                true_vals = merged_pp[col_name].astype(float)
                sim_vals = merged_pp[sim_col].astype(float)
                tf[domain] = {
                    "pearson": pearson_corr(true_vals, sim_vals),
                    "spearman": spearman_corr(true_vals, sim_vals),
                }

        if (
            self.config.nfc_column in merged_pp.columns
            and "sim_nfc_score" in merged_pp.columns
        ):
            true_vals = merged_pp[self.config.nfc_column].astype(float)
            sim_vals = merged_pp["sim_nfc_score"].astype(float)
            tf["nfc"] = {
                "pearson": pearson_corr(true_vals, sim_vals),
                "spearman": spearman_corr(true_vals, sim_vals),
            }

        if (
            self.config.crt_column in merged_pp.columns
            and "sim_crt2_score" in merged_pp.columns
        ):
            true_vals = merged_pp[self.config.crt_column].astype(float)
            sim_vals = merged_pp["sim_crt2_score"].astype(float)
            mae = float(np.mean(np.abs(true_vals - sim_vals)))
            tf["crt2"] = {
                "pearson": pearson_corr(true_vals, sim_vals),
                "spearman": spearman_corr(true_vals, sim_vals),
                "mae": mae,
            }

        tf["monotonicity_notes"] = (
            "Higher target traits should correspond to higher simulated "
            "scale scores; see trait_fidelity correlations."
        )

        # ----------------- Behavioural alignment -----------------
        ba = metrics["behavioural_alignment"]

        writing_cols = [
            col for col in merged_pp.columns if col.startswith("writing_")
        ]
        trait_pairs: List[Tuple[str, str]] = []
        for col in writing_cols:
            if col.endswith("_length") or col.endswith(
                ("_analytic", "_insight", "_affect")
            ):
                trait_pairs.append(("nfc", col))
                trait_pairs.append(("crt2", col))

        corrs: Dict[str, Any] = {}
        regressions: Dict[str, Any] = {}
        for trait, w_col in trait_pairs:
            if w_col not in merged_pp.columns:
                continue
            if trait == "nfc":
                t_col = self.config.nfc_column
            else:
                t_col = self.config.crt_column

            if t_col not in merged_pp.columns:
                continue

            true_vals = merged_pp[t_col].astype(float)
            sim_vals = merged_pp[w_col].astype(float)
            mask = true_vals.notna() & sim_vals.notna()
            if mask.sum() < 3:
                continue

            t = true_vals[mask]
            s = sim_vals[mask]
            key = f"{trait}_vs_{w_col}"

            corrs[key] = {
                "pearson": pearson_corr(t, s),
                "spearman": spearman_corr(t, s),
            }

            # Simple linear regression: s = a * t + b
            try:
                coef = np.polyfit(t.to_numpy(), s.to_numpy(), 1)
                regressions[key] = {
                    "slope": float(coef[0]),
                    "intercept": float(coef[1]),
                }
            except Exception:
                regressions[key] = {"slope": float("nan"), "intercept": float("nan")}

        ba["trait_writing_correlations"] = corrs
        ba["trait_writing_regressions"] = regressions

        # Stratified analyses by writing condition and tags (per writing item)
        writing_items = per_item[per_item["item_type"] == "open"].copy()
        by_condition: Dict[str, Any] = {}
        by_tag: Dict[str, Any] = {}

        if not writing_items.empty:
            # Merge trait scores from validation df
            trait_subset_cols = ["persona_id"]
            for col in [self.config.nfc_column, self.config.crt_column]:
                if col in val_df.columns:
                    trait_subset_cols.append(col)
            trait_df = val_df[trait_subset_cols].drop_duplicates("persona_id")
            writing_items = writing_items.merge(trait_df, on="persona_id", how="left")

            metric_cols = [
                c
                for c in writing_items.columns
                if c
                in {
                    "writing_length",
                    "writing_analytic",
                    "writing_insight",
                    "writing_affect",
                }
            ]

            # By condition
            if "writing_condition" in writing_items.columns:
                for cond in sorted(
                    set(
                        v
                        for v in writing_items["writing_condition"].dropna().unique()
                    )
                ):
                    cond_df = writing_items[
                        writing_items["writing_condition"] == cond
                    ]
                    group_metrics: Dict[str, Any] = {}
                    for m_col in metric_cols:
                        for trait, t_col in [
                            ("nfc", self.config.nfc_column),
                            ("crt2", self.config.crt_column),
                        ]:
                            if t_col not in cond_df.columns:
                                continue
                            t_vals = cond_df[t_col].astype(float)
                            s_vals = cond_df[m_col].astype(float)
                            mask = t_vals.notna() & s_vals.notna()
                            if mask.sum() < 3:
                                continue
                            t = t_vals[mask]
                            s = s_vals[mask]
                            key = f"{trait}_vs_{m_col}"
                            corr_info = {
                                "pearson": pearson_corr(t, s),
                                "spearman": spearman_corr(t, s),
                            }
                            try:
                                coef = np.polyfit(t.to_numpy(), s.to_numpy(), 1)
                                reg_info = {
                                    "slope": float(coef[0]),
                                    "intercept": float(coef[1]),
                                }
                            except Exception:
                                reg_info = {
                                    "slope": float("nan"),
                                    "intercept": float("nan"),
                                }
                            group_metrics[key] = {
                                "correlations": corr_info,
                                "regression": reg_info,
                            }
                    if group_metrics:
                        by_condition[str(cond)] = group_metrics

            # By tag (explode tags)
            if "writing_tags" in writing_items.columns:
                def split_tags(val: Any) -> List[str]:
                    if isinstance(val, list):
                        return [str(v).strip() for v in val if str(v).strip()]
                    if isinstance(val, str):
                        return [
                            t.strip()
                            for t in re.split(r"[;,]", val)
                            if t.strip()
                        ]
                    return []

                writing_items = writing_items.assign(
                    _tag_list=writing_items["writing_tags"].apply(split_tags)
                ).explode("_tag_list")

                tag_vals = writing_items["_tag_list"].dropna().unique()
                for tag in sorted(set(tag_vals)):
                    tag_df = writing_items[writing_items["_tag_list"] == tag]
                    group_metrics: Dict[str, Any] = {}
                    for m_col in metric_cols:
                        for trait, t_col in [
                            ("nfc", self.config.nfc_column),
                            ("crt2", self.config.crt_column),
                        ]:
                            if t_col not in tag_df.columns:
                                continue
                            t_vals = tag_df[t_col].astype(float)
                            s_vals = tag_df[m_col].astype(float)
                            mask = t_vals.notna() & s_vals.notna()
                            if mask.sum() < 3:
                                continue
                            t = t_vals[mask]
                            s = s_vals[mask]
                            key = f"{trait}_vs_{m_col}"
                            corr_info = {
                                "pearson": pearson_corr(t, s),
                                "spearman": spearman_corr(t, s),
                            }
                            try:
                                coef = np.polyfit(t.to_numpy(), s.to_numpy(), 1)
                                reg_info = {
                                    "slope": float(coef[0]),
                                    "intercept": float(coef[1]),
                                }
                            except Exception:
                                reg_info = {
                                    "slope": float("nan"),
                                    "intercept": float("nan"),
                                }
                            group_metrics[key] = {
                                "correlations": corr_info,
                                "regression": reg_info,
                            }
                    if group_metrics:
                        by_tag[str(tag)] = group_metrics

        ba["by_condition"] = by_condition
        ba["by_tag"] = by_tag

        # ----------------- Response consistency -----------------
        rc = metrics["response_consistency"]

        alpha_results: Dict[str, float] = {}
        for domain in self.config.bfi_columns.keys():
            subset = per_item[
                (per_item["test_name"] == "BFI")
                & (per_item["scale_name"] == domain)
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
    Load personas and markers from disk according to the provided
    SimulationConfig.
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
    Validate the personas schema, compute CRT levels, construct
    Persona-based prompts, and build the TestBank from YAML specs.
    """
    personas_df = data["personas"]
    markers_df = data.get("markers")

    # Schema validation
    expected_columns = [
        "Country",
        "Employment Status",
        "Occupation_cat",
        "sex",
        "Highest education level completed",
        "Ethnicity Simplified",
        "Age",
        config.crt_column,
        config.nfc_column,
    ] + list(config.bfi_columns.values())
    missing_schema = [c for c in expected_columns if c not in personas_df.columns]
    if missing_schema:
        raise ValueError(
            "synthetic_personas.csv is missing required columns: "
            f"{missing_schema}. Expected schema includes demographic "
            "variables and trait scores as per the experimental spec."
        )

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

    # Range checks for BFI and NFC
    for col in config.bfi_columns.values():
        vals = personas_df[col].astype(float)
        if ((vals < 0.5) | (vals > 5.5)).any():
            raise ValueError(
                f"Column '{col}' appears to be out of expected [1,5] range. "
                "Please ensure BFI scores are on a 1–5 scale."
            )
    nfc_vals = personas_df[config.nfc_column].astype(float)
    if ((nfc_vals < 0.5) | (nfc_vals > 5.5)).any():
        raise ValueError(
            f"Column '{config.nfc_column}' appears to be out of expected [1,5] range. "
            "Please ensure NFC_Total is on a 1–5 scale."
        )

    crt_vals = pd.to_numeric(personas_df[config.crt_column], errors="coerce")
    if crt_vals.isna().any():
        raise ValueError(
            f"Column '{config.crt_column}' must be numeric and non-missing "
            "for all personas."
        )

    personas_df[config.crt_column] = crt_vals

    # Always compute/discretize crt_level_column from the continuous CRT trait
    personas_df[config.crt_level_column] = map_crt_total_to_level(
        personas_df[config.crt_column]
    )

    prompt_builder = PersonaPromptBuilder(
        markers_df if markers_df is not None and not markers_df.empty else None,
        rng=np.random.default_rng(config.random_seed),
    )

    persona_prompts: Dict[Any, str] = {}

    for idx, row in personas_df.iterrows():
        persona = Persona.from_series(row, config)
        prompt = prompt_builder.build_full_persona_prompt(
            persona,
            style=config.persona_prompt_style,
            numeric_crt_only=config.use_crt_numeric_only,
        )
        persona_prompts[idx] = prompt

    persona_prompts_series = pd.Series(persona_prompts)

    test_bank = build_test_bank()

    return personas_df, persona_prompts_series, test_bank


def holdout_split(
    personas_df: pd.DataFrame, config: SimulationConfig
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split personas into training and validation sets according to
    config.holdout_fraction, optionally respecting a temporal ordering
    column if provided.
    """
    if config.time_column and config.time_column in personas_df.columns:
        sorted_df = personas_df.sort_values(config.time_column)
    else:
        sort_col = "persona_id" if "persona_id" in personas_df.columns else None
        if sort_col:
            sorted_df = personas_df.sort_values(sort_col)
        else:
            sorted_df = personas_df.sort_index()

    n_total = len(sorted_df)
    n_val = max(1, int(round(config.holdout_fraction * n_total)))
    n_train = max(1, n_total - n_val)

    # Preserve original indices to keep alignment with persona_prompts
    train_df = sorted_df.iloc[:n_train]
    val_df = sorted_df.iloc[n_train:]

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
    Save simulation outputs (per-persona and per-item), API call logs,
    and a metrics/configuration report to disk.
    """
    output_dir = os.path.join(PROJECT_ROOT, config.output_dir)
    ensure_dir(output_dir)

    per_persona_path = os.path.join(output_dir, "per_persona_results.csv")
    per_item_path = os.path.join(output_dir, "per_item_results.csv")
    sim_result.per_persona.to_csv(per_persona_path, index=False)
    sim_result.per_item.to_csv(per_item_path, index=False)

    if sim_result.api_calls is not None:
        api_calls_csv_path = os.path.join(output_dir, "api_calls.csv")
        sim_result.api_calls.to_csv(api_calls_csv_path, index=False)
        api_calls_jsonl_path = os.path.join(output_dir, "api_calls.jsonl")
        with open(api_calls_jsonl_path, "w", encoding="utf-8") as f_api:
            for _, row in sim_result.api_calls.iterrows():
                rec = row.to_dict()
                json.dump(rec, f_api)
                f_api.write("\n")

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
            "persona_prompt_style": config.persona_prompt_style,
        },
    }

    metrics_path = os.path.join(output_dir, "metrics_and_config.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

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
    parser.add_argument(
        "--persona-prompt-style",
        type=str,
        default="granular_serapio",
        choices=["granular_serapio", "coarse_numeric", "coarse_descriptive"],
        help="Style for BFI persona prompts.",
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
        persona_prompt_style=args.persona_prompt_style,
    )
    config.validate()
    return config


# ---------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------


def main() -> None:
    config = parse_cli()

    random.seed(config.random_seed)
    np.random.seed(config.random_seed)

    data = load_data(config)

    personas_df, persona_prompts, test_bank = build_network_and_agents(
        data, config
    )

    train_df, val_df = holdout_split(personas_df, config)
    train_prompts = persona_prompts.loc[train_df.index]
    val_prompts = persona_prompts.loc[val_df.index]

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
        best_params = {
            "likert_noise": 0.5,
            "crt_slip": 0.2,
            "random_seed": config.random_seed,
        }
        calibrator.best_params_ = best_params
        calibrator.best_score_ = None

    val_result = simulator.rollout(val_df, best_params, val_prompts)

    evaluator = Evaluator(config)
    metrics = evaluator.compute_metrics(val_result, val_df, train_df)

    save_results(metrics, val_result, config, calibrator)


main()