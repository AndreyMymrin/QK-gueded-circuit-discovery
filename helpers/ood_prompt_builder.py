from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

PROMPT_PREFIX = "Use provided context and answer whether the statement is true or false."

DEFAULT_DATASET_BY_HOPS = {
    2: "2hop_ProofsOnly_4shot_random_noadj.json",
    3: "3hop_ProofsOnly_random_noadj.json",
    4: "4hop_ProofsOnly_random_noadj.json",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_dataset_path(length_of_chain: int, dataset_path: Optional[str]) -> Path:
    if dataset_path is not None:
        path = Path(dataset_path)
        if not path.is_absolute():
            path = _repo_root() / path
        return path

    if length_of_chain not in DEFAULT_DATASET_BY_HOPS:
        valid = ", ".join(str(x) for x in sorted(DEFAULT_DATASET_BY_HOPS))
        raise ValueError(f"Unsupported length_of_chain={length_of_chain}. Expected one of: {valid}")

    return _repo_root() / "generated_ood_data" / DEFAULT_DATASET_BY_HOPS[length_of_chain]


def _example_sort_key(name: str) -> Tuple[int, str]:
    match = re.fullmatch(r"example(\d+)", name)
    if match:
        return (int(match.group(1)), name)
    return (10**9, name)


def _load_dataset_examples(dataset_file: Path) -> Tuple[Dict[str, Any], list[str]]:
    if not dataset_file.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_file}")

    with dataset_file.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)

    if not isinstance(raw, dict):
        raise ValueError(f"Expected JSON object at top-level, got {type(raw)}")

    example_ids = sorted(raw.keys(), key=_example_sort_key)
    if not example_ids:
        raise ValueError(f"No examples found in dataset file: {dataset_file}")

    return raw, example_ids


def _strip_prove_prefix(query: str) -> str:
    return re.sub(r"^\s*Prove:\s*", "", query).strip()


def _build_prompt(context: str, statement: str) -> str:
    return (
        f"{PROMPT_PREFIX}\n"
        f"CONTEXT: {context}\n"
        f"STATEMENT: {statement}\n"
        "ANSWER: "
    )


def _extract_rule_fact(chain_of_thought: Any) -> Tuple[str, str]:
    if not isinstance(chain_of_thought, list) or len(chain_of_thought) == 0:
        return "", ""

    correct_fact = chain_of_thought[0] if isinstance(chain_of_thought[0], str) else ""
    queried_rule = chain_of_thought[-2] if len(chain_of_thought) >= 2 and isinstance(chain_of_thought[-2], str) else ""
    return queried_rule, correct_fact


def generate_cot_question_query_based(
    length_of_chain: int = 2,
    num_cot_samples: int = 6,
    dataset_path: Optional[str] = None,
    seed: Optional[int] = None,
):
    """
    Legacy-compatible 6-element return API backed by generated_ood_data JSON files.

    Returns:
        question_string_LO,
        gt_string_LO,
        question_string_lin,
        gt_string_lin,
        problem_info_dict_LO,
        problem_info_dict_lin,
    """
    _ = num_cot_samples  # kept for compatibility with legacy signature

    dataset_file = _resolve_dataset_path(length_of_chain, dataset_path)
    dataset, example_ids = _load_dataset_examples(dataset_file)

    rng = random.Random(seed) if seed is not None else random

    example_i_id = rng.choice(example_ids)
    if len(example_ids) < 2:
        raise ValueError("Dataset must contain at least two examples to build LO/lin contrast pair.")

    other_ids = [ex_id for ex_id in example_ids if ex_id != example_i_id]
    example_j_id = rng.choice(other_ids)

    test_i = dataset[example_i_id]["test_example"]
    test_j = dataset[example_j_id]["test_example"]

    statement_i = _strip_prove_prefix(test_i["query"])

    question_string_LO = _build_prompt(test_i["question"], statement_i)
    question_string_lin = _build_prompt(test_j["question"], statement_i)

    gt_string_LO = "true"
    gt_string_lin = "false"

    queried_rule_lo, correct_fact_lo = _extract_rule_fact(test_i.get("chain_of_thought"))
    queried_rule_lin, correct_fact_lin = _extract_rule_fact(test_j.get("chain_of_thought"))

    problem_info_dict_LO = {
        "source_example_id": example_i_id,
        "context_source_example_id": example_i_id,
        "statement_source_example_id": example_i_id,
        "statement": statement_i,
        "chain_of_thought": test_i.get("chain_of_thought", []),
        "queried_rule": queried_rule_lo,
        "correct_fact": correct_fact_lo,
    }

    problem_info_dict_lin = {
        "source_example_id": example_j_id,
        "context_source_example_id": example_j_id,
        "statement_source_example_id": example_i_id,
        "statement": statement_i,
        "chain_of_thought": test_j.get("chain_of_thought", []),
        "queried_rule": queried_rule_lin,
        "correct_fact": correct_fact_lin,
    }

    return (
        question_string_LO,
        gt_string_LO,
        question_string_lin,
        gt_string_lin,
        problem_info_dict_LO,
        problem_info_dict_lin,
    )
