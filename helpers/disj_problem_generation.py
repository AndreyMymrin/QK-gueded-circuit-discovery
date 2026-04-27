"""
Legacy compatibility wrapper.

The canonical OOD prompt generation now lives in helpers.ood_prompt_builder.
"""

from __future__ import annotations

from typing import Optional

from helpers.ood_prompt_builder import (
    generate_cot_question_query_based as _generate_cot_question_query_based,
)


def generate_cot_question_query_based(
    length_of_chain: int = 2,
    num_cot_samples: int = 6,
    dataset_path: Optional[str] = None,
    seed: Optional[int] = None,
):
    return _generate_cot_question_query_based(
        length_of_chain=length_of_chain,
        num_cot_samples=num_cot_samples,
        dataset_path=dataset_path,
        seed=seed,
    )

