from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROMPT_PREFIX = "Use provided context and answer whether the statement is true or false."

DEFAULT_DATASET_BY_HOPS = {
    2: "2hop_ProofsOnly_random_noadj.json",
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


def _subexample_sort_key(name: str) -> Tuple[int, str]:
    match = re.fullmatch(r"in_context_example(\d+)", name)
    if match:
        return (int(match.group(1)), name)
    return (10**9, name)


def _load_dataset_examples(dataset_file: Path) -> Tuple[Dict[str, Any], List[str]]:
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


def _extract_statement(query: str) -> str:
    if not isinstance(query, str):
        return ""

    if ": " in query:
        return query.split(": ", 1)[1].strip()
    if ":" in query:
        return query.split(":", 1)[1].strip()

    match = re.match(r"^\s*(Prove|True or false)\s*:?\s*(.*)$", query, flags=re.IGNORECASE)
    if match:
        return match.group(2).strip()
    return query.strip()


def _get_negation(sss: str) -> str:
    # Ported from ProntoQA_experiments/prontoqa-to spread.ipynb
    separator = ""
    if len(sss) == 0:
        return ""

    if " and " in sss:
        pos = sss.find(" and ")
        separator = " or "
        parts = [_get_negation(sss[:pos]), _get_negation(sss[pos + 5 :])]
    elif " or " in sss:
        pos = sss.find(" or ")
        separator = " and "
        parts = [_get_negation(sss[:pos]), _get_negation(sss[pos + 4 :])]
    elif "," in sss:
        pos = sss.find(",")
        separator = ", "
        right_start = pos + 2 if (pos + 1 < len(sss) and sss[pos + 1] == " ") else pos + 1
        parts = [_get_negation(sss[:pos]), _get_negation(sss[right_start:])]
        if parts[1] == "":
            separator = ","
    else:
        if " not " in sss:
            parts = sss.split(" not ", 1)
            return parts[0] + " " + parts[1]
        if sss.startswith("not "):
            return sss[4:]
        if " is " in sss:
            parts = sss.split(" is ", 1)
            return parts[0] + " is not " + parts[1]
        if " a " in sss:
            parts = sss.split(" a ", 1)
            return parts[0] + " not a " + parts[1]
        return "not " + sss

    result = parts[0] + separator + parts[1]

    if " or " in result:
        result = result.replace(", ", " or ")
        result = result.replace(" or or ", " or ")

    if " and " in result:
        result = result.replace(" and ", ", ")
        result = result.replace(",,", ",")
        for i in range(len(result) - 1, 0, -1):
            if result[i] == ",":
                if "," in result[:i]:
                    result = result[:i] + ", and " + result[i + 2 :]
                else:
                    result = result[:i] + " and " + result[i + 2 :]
                break

    return result.strip()


def _build_prompt(context: str, statement: str) -> str:
    return (
        f"{PROMPT_PREFIX}\n"
        f"Context: {context}\n"
        f"Statement: {statement}\n"
        "Answer: "
    )


def _has_double_newline(items: List[str]) -> bool:
    for s in items:
        if isinstance(s, str) and "\n\n" in s:
            return True
    return False


def _build_prontoqa_few_shot_prompt(
    in_context_items: List[Dict[str, Any]],
    test_question: str,
    test_statement: str,
) -> str:
    """
    Build a few-shot prompt in the style of prontoqa/data/prompt.py:
      Q: <question> <query>
      A: <chain_of_thought> <answer>
    """
    all_texts: List[str] = [test_question, test_statement]
    for item in in_context_items:
        all_texts.append(str(item.get("question", "")))
        all_texts.append(str(item.get("query", "")))
        cot = item.get("chain_of_thought", [])
        if isinstance(cot, list):
            all_texts.extend([str(x) for x in cot if isinstance(x, str)])
    newline = "\n\n" if _has_double_newline(all_texts) else "\n"

    prompt = ""
    for item in in_context_items:
        context = str(item.get("question", "")).strip()
        statement = _extract_statement(str(item.get("query", "")))
        query = f"True or false: {statement}"

        cot = item.get("chain_of_thought", [])
        cot_text = " ".join(cot).strip() if isinstance(cot, list) else ""
        answer = str(item.get("answer", "true")).strip().lower()
        if answer not in {"true", "false"}:
            answer = "true"

        prompt += f"Q: {context} {query}{newline}A: "
        if cot_text:
            prompt += f"{cot_text} "
        prompt += f"{answer}{newline}\n"

    test_query = f"True or false: {test_statement}"
    prompt += f"Q: {test_question} {test_query}{newline}A:"
    return prompt


def _extract_rule_fact(chain_of_thought: Any) -> Tuple[str, str]:
    if not isinstance(chain_of_thought, list) or len(chain_of_thought) == 0:
        return "", ""

    correct_fact = chain_of_thought[0] if isinstance(chain_of_thought[0], str) else ""
    queried_rule = chain_of_thought[-2] if len(chain_of_thought) >= 2 and isinstance(chain_of_thought[-2], str) else ""
    return queried_rule, correct_fact


def _collect_in_context_pool(dataset: Dict[str, Any], example_ids: List[str]) -> List[Dict[str, Any]]:
    pool: List[Dict[str, Any]] = []
    for example_id in example_ids:
        example = dataset[example_id]
        if not isinstance(example, dict):
            continue
        sub_keys = sorted(
            [k for k in example.keys() if k.startswith("in_context_example")],
            key=_subexample_sort_key,
        )
        for sub_key in sub_keys:
            item = example[sub_key]
            if not isinstance(item, dict):
                continue
            if "question" not in item or "query" not in item:
                continue
            pool.append(
                {
                    "source_example_id": example_id,
                    "source_subexample_id": sub_key,
                    "question": item["question"],
                    "query": item["query"],
                    "chain_of_thought": item.get("chain_of_thought", []),
                }
            )

    if len(pool) == 0:
        raise ValueError("No in_context_example* entries found in dataset.")
    return pool


def generate_cot_question_query_based(
    length_of_chain: int = 2,
    num_cot_samples: int = 6,
    dataset_path: Optional[str] = None,
    seed: Optional[int] = None,
):
    """
    Legacy-compatible 6-element return API backed by generated_ood_data JSON files.

    Modes:
    - num_cot_samples <= 0:
        Zero-shot context/statement prompt pair:
        same Context, Statement vs negated Statement.
    - num_cot_samples > 0:
        Few-shot prompt in ProntoQA Q:/A: style with CoT demonstrations.
        Uses `test_example` as target and up to `num_cot_samples` in-context demos.
    """
    dataset_file = _resolve_dataset_path(length_of_chain, dataset_path)
    dataset, example_ids = _load_dataset_examples(dataset_file)

    rng = random.Random(seed) if seed is not None else random

    if num_cot_samples <= 0:
        # Zero-shot mode over in_context_example* pool.
        pool = _collect_in_context_pool(dataset, example_ids)
        selected = rng.choice(pool)

        context = str(selected["question"]).strip()
        statement_true = _extract_statement(str(selected["query"]))
        statement_false = _get_negation(statement_true)

        question_string_LO = _build_prompt(context, statement_true)
        question_string_lin = _build_prompt(context, statement_false)

        queried_rule, correct_fact = _extract_rule_fact(selected.get("chain_of_thought", []))
        common_info = {
            "prompt_mode": "zero_shot_context_statement",
            "source_example_id": selected["source_example_id"],
            "source_subexample_id": selected["source_subexample_id"],
            "context": context,
            "chain_of_thought": selected.get("chain_of_thought", []),
            "queried_rule": queried_rule,
            "correct_fact": correct_fact,
        }
    else:
        # Few-shot mode in prontoqa Q/A style, using a full top-level example.
        selected_example_id = rng.choice(example_ids)
        selected_example = dataset[selected_example_id]
        if not isinstance(selected_example, dict):
            raise ValueError(f"Expected dict at dataset[{selected_example_id!r}]")

        in_context_keys = sorted(
            [k for k in selected_example.keys() if k.startswith("in_context_example")],
            key=_subexample_sort_key,
        )
        if len(in_context_keys) == 0:
            raise ValueError(
                f"No in_context_example* entries in {selected_example_id}; "
                "few-shot mode requires in-context demonstrations."
            )
        if "test_example" not in selected_example or not isinstance(selected_example["test_example"], dict):
            raise ValueError(f"Missing valid test_example for {selected_example_id}")

        shots_k = min(num_cot_samples, len(in_context_keys))
        selected_shot_keys = in_context_keys[:shots_k]
        in_context_items = [selected_example[k] for k in selected_shot_keys]

        test_item = selected_example["test_example"]
        test_question = str(test_item.get("question", "")).strip()
        statement_true = _extract_statement(str(test_item.get("query", "")))
        statement_false = _get_negation(statement_true)

        question_string_LO = _build_prontoqa_few_shot_prompt(in_context_items, test_question, statement_true)
        question_string_lin = _build_prontoqa_few_shot_prompt(in_context_items, test_question, statement_false)

        queried_rule, correct_fact = _extract_rule_fact(test_item.get("chain_of_thought", []))
        common_info = {
            "prompt_mode": "few_shot_prontoqa_qa",
            "source_example_id": selected_example_id,
            "source_subexample_id": "test_example",
            "context": test_question,
            "chain_of_thought": test_item.get("chain_of_thought", []),
            "queried_rule": queried_rule,
            "correct_fact": correct_fact,
            "few_shot_source_ids": selected_shot_keys,
            "few_shot_count": shots_k,
        }

    gt_string_LO = "true"
    gt_string_lin = "false"

    problem_info_dict_LO = {**common_info, "statement": statement_true, "expected_label": "true"}
    problem_info_dict_lin = {**common_info, "statement": statement_false, "expected_label": "false"}

    return (
        question_string_LO,
        gt_string_LO,
        question_string_lin,
        gt_string_lin,
        problem_info_dict_LO,
        problem_info_dict_lin,
    )
