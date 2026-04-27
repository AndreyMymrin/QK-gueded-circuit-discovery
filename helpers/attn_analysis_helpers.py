# utils_logic_attn.py
from typing import List, Dict, Tuple, Optional, Any
import torch

Span = Tuple[int, int]  # [start, end) token indices (0-based, end exclusive)

# -------------------------------
# Token-ID region identification helpers
# -------------------------------

# Tokenize text to a flat python list of token IDs, no BOS since not relevant
def _to_ids(model, text: str) -> List[int]:
    return model.to_tokens(text, prepend_bos=False)[0].tolist()

# Find token IDs for searching a clause (handles spaces/newlines & trailing punctuation).
# Returns a list of token-id lists.
def _make_variants(model, text: str) -> List[List[int]]:
    leads = ["", " ", "\n"]
    trails = ["", ".", ";", " .", " ;"]
    bag = []
    for L in leads:
        for T in trails:
            bag.append(L + text + T)
    # Also allow these common endings
    bag += [text + ". ", text + "; "]

    # De-duplicate by token sequence
    seen = set()
    out: List[List[int]] = []
    for s in bag:
        ids = _to_ids(model, s)
        key = tuple(ids)
        if len(ids) > 0 and key not in seen:
            seen.add(key)
            out.append(ids)
    return out

# Find a subsequence 'needle' of token IDs in the full tokenized prompt 'hay'
def _find_all_subseq(sntc: List[int], subseq: List[int]) -> List[int]:
    out = []
    if len(subseq) == 0 or len(subseq) > len(sntc):
        return out
    n = len(subseq)
    for i in range(len(sntc) - n + 1):
        if sntc[i:i+n] == subseq:
            out.append(i)
    return out

def _find_marker(tokens_row: List[int], 
                 model, marker: str,
                 start_at: int = 0, 
                 end_at: Optional[int] = None,
                 want_last: bool = False) -> Optional[Span]:

    """ 
    Find markers for the logic problem, including 'Rules:',  'Facts:', 'Question:', 'Answer:'.
    Returns (start, end) token indices or None.
    """
    if end_at is None:
        end_at = len(tokens_row)
    hay = tokens_row[start_at:end_at]

    variants = []
    for s in [marker, " " + marker, "\n" + marker]:
        variants.append(_to_ids(model, s))

    best: Optional[Tuple[int,int]] = None
    for v in variants:
        starts = _find_all_subseq(hay, v)
        if not starts:
            continue
        idx = (starts[-1] if want_last else starts[0]) + start_at
        cand = (idx, idx + len(v))
        if best is None:
            best = cand
        else:
            if want_last and cand[0] > best[0]:
                best = cand
            if (not want_last) and cand[0] < best[0]:
                best = cand
    return best


def _find_marker_any(
    tokens_row: List[int],
    model,
    markers: List[str],
    start_at: int = 0,
    end_at: Optional[int] = None,
    want_last: bool = False,
) -> Optional[Span]:
    """
    Find marker using multiple textual variants (e.g., 'Context:' and 'CONTEXT:').
    Returns the earliest or latest span by start index depending on want_last.
    """
    best: Optional[Span] = None
    for marker in markers:
        span = _find_marker(tokens_row, model, marker, start_at=start_at, end_at=end_at, want_last=want_last)
        if span is None:
            continue
        if best is None:
            best = span
            continue
        if want_last and span[0] > best[0]:
            best = span
        if (not want_last) and span[0] < best[0]:
            best = span
    return best


def _find_in_range(tokens_row: List[int], model, clause_text: str,
                   start_: int, end_: int) -> Optional[Span]:
    """
    Find clause_text (as token-ID sequence) within [start_, end_) using region markers.
    For our purposes, the clause_text would fall into the categories of queried rule and correct fact,
        since this is primarily used for finding the attention mass of a selected attention head on such clauses.
    
    Returns (start, end) or None.
    """
    region = tokens_row[start_:end_]
    variants = _make_variants(model, clause_text)

    for v in variants:
        starts = _find_all_subseq(region, v)
        if starts:
            s = starts[0] + start_
            return (s, s + len(v))

    # If direct match fails, try seeking a trailing punctuation token in the region.
    base_variants = _make_variants(model, clause_text.rstrip(".; "))
    for v in base_variants:
        for punct in [".", ";"]:
            v2 = v + _to_ids(model, punct)
            starts = _find_all_subseq(region, v2)
            if starts:
                s = starts[0] + start_
                return (s, s + len(v2))
    return None

# Finding different regions of the last problem (after the in-context examples).
# Useful for cleanly locating the queried rule and correct fact for writing the proof.
def locate_final_problem_regions(tokens_row: torch.Tensor, model) -> Dict[str, Span]:
    """
    Returns token-ID level bounds (start, end) for the last problem in the prompt:
    - Old format ('Rules/Facts/Question/Answer'):
      - rules_region: from end of 'Rules:' to start of 'Facts:'
      - facts_region: from end of 'Facts:' to start of 'Question:'
      - problem_region: from 'Rules:' to 'Answer:' for the last problem (inclusive on both ends)
    - New format ('CONTEXT/STATEMENT/ANSWER'):
      - rules_region: the CONTENT span inside 'CONTEXT'
      - facts_region: same as rules_region (kept for compatibility)
      - problem_region: from 'CONTEXT:' marker to 'ANSWER:' marker (inclusive on marker end)
    - answer_marker: the 'Answer:' marker span
    """
    row = tokens_row.tolist()

    # Old format path: Rules/Facts/Question/Answer
    rules_m = _find_marker_any(row, model, ["Rules:", "RULES:"], want_last=True)
    if rules_m is not None:
        rules_start, rules_end = rules_m

        facts_m = _find_marker_any(row, model, ["Facts:", "FACTS:"], start_at=rules_end, want_last=False)
        if facts_m is None:
            raise ValueError("Could not find 'Facts:' after final 'Rules:'.")
        facts_start, facts_end = facts_m

        question_m = _find_marker_any(row, model, ["Question:", "QUESTION:"], start_at=facts_end, want_last=False)
        answer_m = _find_marker_any(row, model, ["Answer:", "ANSWER:"], start_at=facts_end, want_last=False)
        if question_m is None and answer_m is None:
            raise ValueError("Could not find either 'Question:' or 'Answer:' after final 'Facts:'.")
        q_start, q_end = (answer_m if question_m is None else question_m)

        final_answer_m = _find_marker_any(row, model, ["Answer:", "ANSWER:"], start_at=q_end, want_last=False)
        if final_answer_m is None:
            if answer_m is not None and answer_m[0] >= facts_end:
                final_answer_m = answer_m
            else:
                raise ValueError("Could not find final 'Answer:' for the last problem.")
        ans_start, ans_end = final_answer_m

        return {
            "rules_region": (rules_end, facts_start),
            "facts_region": (facts_end, q_start),
            "problem_region": (rules_start, ans_end),
            "answer_marker": (ans_start, ans_end),
        }

    # New format path: CONTEXT/STATEMENT/ANSWER (case-robust)
    context_m = _find_marker_any(row, model, ["CONTEXT:", "Context:"], want_last=True)
    if context_m is None:
        raise ValueError("Could not find either legacy ('Rules:') or OOD ('Context:'/'CONTEXT:') prompt markers.")
    context_start, context_end = context_m

    statement_m = _find_marker_any(row, model, ["STATEMENT:", "Statement:"], start_at=context_end, want_last=False)
    if statement_m is None:
        raise ValueError("Could not find 'Statement:'/'STATEMENT:' after final 'Context:' marker.")
    statement_start, statement_end = statement_m

    answer_m = _find_marker_any(row, model, ["ANSWER:", "Answer:"], start_at=statement_end, want_last=False)
    if answer_m is None:
        raise ValueError("Could not find final 'Answer:'/'ANSWER:' for OOD-format prompt.")
    ans_start, ans_end = answer_m

    context_content_region = (context_end, statement_start)
    return {
        "rules_region": context_content_region,
        "facts_region": context_content_region,
        "problem_region": (context_start, ans_end),
        "answer_marker": (ans_start, ans_end),
    }

# Identifying the position of a clause (e.g. a rule, a fact) in the
# final problem (after the in-context examples), for a single sample.
def clause_token_spans_for_row(tokens_row: torch.Tensor,
                               model,
                               problem_info: Dict[str, str]) -> Dict[str, Optional[Span]]:
    """
    Given a single prompt (token ID sequence) and a dict like
      {'queried_rule': 'B implies E', 'correct_fact': 'B is true'}
    return token spans (start, end) for each clause inside the final problem.
    """
    regions = locate_final_problem_regions(tokens_row, model)
    row = tokens_row.tolist()

    out: Dict[str, Optional[Span]] = {"queried_rule": None, "correct_fact": None}

    # Queried rule lives in the Rules region
    if problem_info.get("queried_rule"):
        out["queried_rule"] = _find_in_range(row, model, problem_info["queried_rule"],
                                             *regions["rules_region"])

    # Correct fact lives in the Facts region
    if problem_info.get("correct_fact"):
        out["correct_fact"] = _find_in_range(row, model, problem_info["correct_fact"],
                                             *regions["facts_region"])

    return out

# Invoking clause_token_spans_for_row for a batch of samples.
def clause_token_spans_for_batch(clean_tokens: torch.Tensor,
                                 model,
                                 problem_infos: List[Dict[str, str]]) -> List[Dict[str, Optional[Span]]]:
    """
    Batch version of clause_token_spans_for_row.
    clean_tokens is [B, S]; problem_infos length must be B.
    Returns a list of dicts with 'queried_rule' and 'correct_fact' spans per row.
    """
    B = clean_tokens.shape[0]
    assert len(problem_infos) == B, "problem_infos must match batch size"

    spans = []
    for b in range(B):
        spans.append(clause_token_spans_for_row(clean_tokens[b], model, problem_infos[b]))
    return spans




# -----------------
# Attention stats
# -----------------

# --- Attention stats at chosen destination position -----------

@torch.no_grad()
def head_attention_mass_at_pos(
    model,
    clean_tokens: torch.Tensor,                 # [B, S]
    problem_infos: List[Dict[str, str]],        # len B; used if spans_by_label is None
    layer_idx: int,
    head_idx: int,
    dest_pos: int,                              # single integer destination index for all rows
    span_finder=None,                           # kept for compatibility (unused here)
    spans_by_label: Optional[Dict[str, List[Optional[Span]]]] = None,
    normalize: bool = False,
    attention_patterns: Optional[torch.Tensor] = None,
    problem_regions: Optional[List[Optional[Span]]] = None,
) -> Dict[str, Any]:
    """
    If spans_by_label is None:
      - Behaves like the original: computes spans for 'queried_rule'/'correct_fact' from problem_infos
        and returns 'rule_mass'/'fact_mass' (lists of floats or None).
      - Also returns per-example 'rule_max_other_in_problem' and 'fact_max_other_in_problem', and
        'problem_total_mass' (same total for both).

    If spans_by_label is provided:
      - Use your supplied spans (per label -> per-example Optional[Span]).
      - Returns results under 'by_span' = { label: { 'mass': [...], 'max_other_in_problem': [...] } }
      - Also returns 'problem_total_mass'.
      - For convenience, if labels include 'queried_rule'/'correct_fact', the legacy keys
        'rule_mass'/'fact_mass' and their max-other counterparts are also populated.
    """
    B, S = clean_tokens.shape
    assert isinstance(dest_pos, int), "dest_pos must be a single integer"

    def _slice_pattern(patterns: torch.Tensor) -> torch.Tensor:
        # Supported shapes:
        # - [B, L, H, S, S]
        # - [B, H, S, S]
        # - [B, S, S]
        if patterns.ndim == 5:
            return patterns[:, layer_idx, head_idx, :, :]
        if patterns.ndim == 4:
            return patterns[:, head_idx, :, :]
        if patterns.ndim == 3:
            return patterns
        raise ValueError(f"Unsupported attention_patterns shape: {tuple(patterns.shape)}")

    if attention_patterns is None:
        if model is None:
            raise ValueError("Either `model` or `attention_patterns` must be provided.")
        _, cache = model.run_with_cache(clean_tokens)
        patt = cache["pattern", layer_idx][:, head_idx, :, :]  # [B, dest, src]
    else:
        patt = _slice_pattern(attention_patterns)
        if patt.shape[0] != B:
            raise ValueError("Batch size mismatch between clean_tokens and attention_patterns.")

    # Build spans if not provided.
    if spans_by_label is None:
        if model is None:
            raise ValueError("`model` is required when spans_by_label is not provided.")
        row_spans = clause_token_spans_for_batch(clean_tokens, model, problem_infos)
        spans_by_label = {
            "queried_rule": [row.get("queried_rule") for row in row_spans],
            "correct_fact": [row.get("correct_fact") for row in row_spans],
        }

    # Build problem regions if not provided.
    if problem_regions is None:
        problem_regions = []
        if model is not None:
            for b in range(B):
                try:
                    regions = locate_final_problem_regions(clean_tokens[b], model)
                    problem_regions.append(regions["problem_region"])
                except Exception:
                    problem_regions.append((0, S))
        else:
            problem_regions = [(0, S) for _ in range(B)]

    if len(problem_regions) != B:
        raise ValueError("problem_regions length must match batch size.")

    by_span: Dict[str, Dict[str, List[Optional[float]]]] = {}
    masses: Dict[str, List[Optional[float]]] = {label: [] for label in spans_by_label.keys()}
    problem_total_mass: List[Optional[float]] = [None] * B
    max_others: List[Optional[float]] = [None] * B

    dest_idx = dest_pos if dest_pos >= 0 else patt.shape[1] + dest_pos

    for b in range(B):
        if not (0 <= dest_idx < patt.shape[1]):
            for label in masses.keys():
                masses[label].append(None)
            continue

        region = problem_regions[b]
        if region is None:
            for label in masses.keys():
                masses[label].append(None)
            continue

        p_s, p_e = region
        p_s = max(0, int(p_s))
        p_e = min(int(p_e), patt.shape[-1])
        if p_s >= p_e:
            for label in masses.keys():
                masses[label].append(None)
            continue

        src_all = torch.arange(p_s, p_e, device=patt.device)
        total = float(patt[b, dest_idx, src_all].sum().item())
        problem_total_mass[b] = total
        union_mask = torch.ones_like(src_all, dtype=torch.bool)

        for label, span_list in spans_by_label.items():
            span = span_list[b] if b < len(span_list) else None
            if span is None:
                masses[label].append(None)
                continue

            # Keep backward compatibility:
            # caller often stores inclusive end token for clause span.
            s0, s1 = span
            s1 += 1
            if s0 < 0 and s1 < 0:
                s0 = p_e + s0
                s1 = p_e + s1

            s0 = max(0, int(s0))
            s1 = min(int(s1), patt.shape[-1])
            if s0 >= s1:
                masses[label].append(None)
                continue

            span_mass = float(patt[b, dest_idx, s0:s1].sum().item())
            if normalize:
                masses[label].append(None if total == 0.0 else span_mass / total)
            else:
                masses[label].append(span_mass)

            # Remove overlap with the union mask to compute max_other_in_problem.
            low = max(s0, p_s) - p_s
            high = min(s1, p_e) - p_s
            if low < high:
                union_mask[low:high] = False

        if union_mask.any():
            other = float(patt[b, dest_idx, src_all[union_mask]].max().item())
            if normalize:
                max_others[b] = None if total == 0.0 else other / total
            else:
                max_others[b] = other
        else:
            max_others[b] = None

    for label in masses.keys():
        by_span[label] = {
            "mass": masses[label],
            "max_other_in_problem": max_others,
        }

    out: Dict[str, Any] = {
        "layer": layer_idx,
        "head": head_idx,
        "dest_pos": dest_pos,
        "problem_total_mass": problem_total_mass,
        "max_other_in_problem": max_others,
        "by_span": by_span,
    }

    # Backward-compatible aliases.
    if "queried_rule" in by_span:
        out["rule_mass"] = by_span["queried_rule"]["mass"]
        out["rule_max_other_in_problem"] = by_span["queried_rule"]["max_other_in_problem"]
    if "correct_fact" in by_span:
        out["fact_mass"] = by_span["correct_fact"]["mass"]
        out["fact_max_other_in_problem"] = by_span["correct_fact"]["max_other_in_problem"]

    return out
