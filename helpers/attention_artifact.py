from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import json

import torch


WordSpan = Tuple[str, int, int]  # (word_text, tok_start, tok_end_exclusive)


def _decode_token_piece(tokenizer, token_id: int) -> str:
    try:
        return tokenizer.decode([int(token_id)], clean_up_tokenization_spaces=False)
    except TypeError:
        return tokenizer.decode([int(token_id)])


def _decode_token_pieces(tokenizer, token_ids: List[int]) -> List[str]:
    return [_decode_token_piece(tokenizer, tid) for tid in token_ids]


def _build_word_token_spans(token_pieces: List[str]) -> List[WordSpan]:
    """
    Build word-level spans over token indices.

    Rules:
    - punctuation ':' ',' '.' is a separate "word"
    - '\\n' is a separate "word"
    - contiguous subword pieces are merged into words
    """
    punct_words = {":", ",", "."}

    words: List[WordSpan] = []
    cur_text = ""
    cur_start: Optional[int] = None
    cur_end: Optional[int] = None

    def flush_current() -> None:
        nonlocal cur_text, cur_start, cur_end
        if cur_start is not None and cur_end is not None:
            text = cur_text if len(cur_text) > 0 else " "
            words.append((text, cur_start, cur_end))
        cur_text = ""
        cur_start = None
        cur_end = None

    for idx, piece in enumerate(token_pieces):
        if piece == "":
            continue

        if piece == "\n":
            flush_current()
            words.append(("\n", idx, idx + 1))
            continue

        if piece.strip() == "":
            if cur_start is None:
                words.append((" ", idx, idx + 1))
            else:
                cur_text += piece
                cur_end = idx + 1
            continue

        starts_new = piece.startswith(" ") or piece.startswith("\t")
        piece_core = piece.lstrip(" \t") if starts_new else piece
        piece_core_stripped = piece_core.strip()

        is_punct = piece_core_stripped in punct_words and len(piece_core_stripped) == 1
        if is_punct:
            flush_current()
            words.append((piece_core_stripped, idx, idx + 1))
            continue

        if starts_new:
            flush_current()

        if cur_start is None:
            cur_start = idx
            cur_end = idx + 1
            cur_text = piece_core
        else:
            cur_end = idx + 1
            cur_text += piece_core

    flush_current()
    return words


def _append_suffix_tokens(base_tokens: torch.Tensor, suffix_ids: List[int]) -> torch.Tensor:
    if len(suffix_ids) == 0:
        return base_tokens
    device = base_tokens.device
    suffix = torch.tensor(suffix_ids, dtype=base_tokens.dtype, device=device)
    suffix = suffix.unsqueeze(0).repeat(base_tokens.shape[0], 1)
    return torch.cat([base_tokens, suffix], dim=1)


@torch.no_grad()
def _collect_all_patterns(
    model,
    tokens: torch.Tensor,
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """
    Return attention patterns stacked as [B, L, H, S, S] on CPU.
    """
    pattern_filter = lambda name: name.endswith("pattern")
    _, cache = model.run_with_cache(tokens, names_filter=pattern_filter)

    per_layer: List[torch.Tensor] = []
    for layer_idx in range(model.cfg.n_layers):
        # [B, H, S, S]
        patt = cache["pattern", layer_idx]
        per_layer.append(patt.detach().to(device="cpu", dtype=dtype))

    del cache
    # [L, B, H, S, S] -> [B, L, H, S, S]
    stacked = torch.stack(per_layer, dim=0).permute(1, 0, 2, 3, 4).contiguous()
    return stacked


def _prepare_labels_for_variant(model, tokens: torch.Tensor) -> Dict[str, Any]:
    token_pieces: List[List[str]] = []
    word_spans: List[List[WordSpan]] = []

    for b in range(tokens.shape[0]):
        row_ids = tokens[b].detach().cpu().tolist()
        pieces = _decode_token_pieces(model.tokenizer, row_ids)
        spans = _build_word_token_spans(pieces)
        token_pieces.append(pieces)
        word_spans.append(spans)

    return {"token_pieces": token_pieces, "word_spans": word_spans}


def _resolve_dtype(dtype: Union[torch.dtype, str]) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    if dtype == "float16":
        return torch.float16
    if dtype == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype}")


@torch.no_grad()
def build_and_save_attention_artifact(
    model,
    clean_tokens: torch.Tensor,
    corrupted_tokens: Optional[torch.Tensor],
    answer_tokens: Optional[torch.Tensor],
    artifact_dir: Union[str, Path],
    experiment_id: str,
    model_name: Optional[str] = None,
    dtype: Union[torch.dtype, str] = torch.float16,
) -> Dict[str, str]:
    """
    Build and save an offline attention artifact:
    - attention_payload.pt
    - manifest.json

    Returns paths to saved files.
    """
    save_dtype = _resolve_dtype(dtype)
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    true_ids = model.to_tokens(" true", prepend_bos=False)[0].detach().cpu().tolist()
    false_ids = model.to_tokens(" false", prepend_bos=False)[0].detach().cpu().tolist()

    variant_tokens: Dict[str, torch.Tensor] = {
        "base": clean_tokens,
        "true": _append_suffix_tokens(clean_tokens, true_ids),
        "false": _append_suffix_tokens(clean_tokens, false_ids),
    }

    patterns: Dict[str, torch.Tensor] = {}
    labels: Dict[str, Dict[str, Any]] = {}
    for variant in ("base", "true", "false"):
        tokens = variant_tokens[variant]
        patterns[variant] = _collect_all_patterns(model, tokens, dtype=save_dtype)
        labels[variant] = _prepare_labels_for_variant(model, tokens)

    base_shape = patterns["base"].shape
    if len(base_shape) != 5:
        raise ValueError(f"Unexpected pattern shape for base variant: {base_shape}")

    pad_token_id = getattr(model.tokenizer, "pad_token_id", None)
    eos_token_id = getattr(model.tokenizer, "eos_token_id", None)
    bos_token_id = getattr(model.tokenizer, "bos_token_id", None)

    payload = {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "patterns": patterns,
        "tokens": {
            "clean_tokens": clean_tokens.detach().cpu(),
            "corrupted_tokens": None if corrupted_tokens is None else corrupted_tokens.detach().cpu(),
            "answer_tokens": None if answer_tokens is None else answer_tokens.detach().cpu(),
            "base_tokens": variant_tokens["base"].detach().cpu(),
            "true_tokens": variant_tokens["true"].detach().cpu(),
            "false_tokens": variant_tokens["false"].detach().cpu(),
            "true_suffix_ids": true_ids,
            "false_suffix_ids": false_ids,
            "pad_token_id": pad_token_id,
            "eos_token_id": eos_token_id,
            "bos_token_id": bos_token_id,
        },
        "labels": labels,
    }

    payload_path = artifact_dir / "attention_payload.pt"
    torch.save(payload, payload_path)

    manifest = {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": model_name or getattr(model.cfg, "model_name", None),
        "dtype": str(save_dtype).replace("torch.", ""),
        "shape": {
            "B": int(base_shape[0]),
            "L": int(base_shape[1]),
            "H": int(base_shape[2]),
        },
        "seq_len": {
            "base": int(patterns["base"].shape[-1]),
            "true": int(patterns["true"].shape[-1]),
            "false": int(patterns["false"].shape[-1]),
        },
        "files": {
            "payload": payload_path.name,
            "manifest": "manifest.json",
        },
        "tokens": {
            "true_suffix_ids": true_ids,
            "false_suffix_ids": false_ids,
            "pad_token_id": pad_token_id,
            "eos_token_id": eos_token_id,
            "bos_token_id": bos_token_id,
        },
    }

    manifest_path = artifact_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return {
        "artifact_dir": str(artifact_dir),
        "payload_path": str(payload_path),
        "manifest_path": str(manifest_path),
    }


def load_attention_artifact(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load an offline attention artifact directory or payload path.
    Returns {"payload": ..., "manifest": ..., "artifact_dir": "..."}.
    """
    path = Path(path)
    if path.is_dir():
        artifact_dir = path
        payload_path = artifact_dir / "attention_payload.pt"
        manifest_path = artifact_dir / "manifest.json"
    else:
        payload_path = path
        artifact_dir = payload_path.parent
        manifest_path = artifact_dir / "manifest.json"

    if not payload_path.exists():
        raise FileNotFoundError(f"Missing payload file: {payload_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest file: {manifest_path}")

    payload = torch.load(payload_path, map_location="cpu")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {
        "payload": payload,
        "manifest": manifest,
        "artifact_dir": str(artifact_dir),
    }

