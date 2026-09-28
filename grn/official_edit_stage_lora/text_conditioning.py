"""StageLoRA text conditioning on top of the official packed-T5 contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch


STAGE_LORA_T2V_PREFIX = "<T2V>"
STAGE_LORA_TEXT_SEPARATOR_TOKENS = 1

OfficialTextCondition = Tuple[
    torch.Tensor,
    List[int],
    torch.Tensor,
    int,
    List[int],
]


@dataclass(frozen=True)
class StageLoRATextCondition:
    """Official backbone text plus one edit-only pooled context per stream."""

    backbone: OfficialTextCondition
    edit_context: torch.Tensor


_MISSING_TEXT_VALUES = frozenset({"", "null", "none", "n/a", "na"})


def _clean_required_text(value: object, *, field_name: str) -> str:
    text = str(value or "").strip()
    if text.lower() in _MISSING_TEXT_VALUES:
        raise ValueError(f"StageLoRA requires non-empty {field_name}.")
    return text


def clean_stage_lora_edit_instruction(value: object) -> str:
    """Validate and normalize one edit instruction without changing its wording."""

    return _clean_required_text(value, field_name="edit instruction")


def clean_stage_lora_reprompt(value: object) -> str:
    """Validate one precomputed offline reprompt."""

    return _clean_required_text(value, field_name="offline reprompt")


def add_stage_lora_t2v_prefix(value: object) -> str:
    """Apply the official ``<T2V>`` prefix with no separating whitespace."""

    text = clean_stage_lora_edit_instruction(value)
    if text.startswith(STAGE_LORA_T2V_PREFIX):
        text = text[len(STAGE_LORA_T2V_PREFIX) :].lstrip()
        if not text:
            raise ValueError("StageLoRA requires text after the <T2V> prefix.")
    return f"{STAGE_LORA_T2V_PREFIX}{text}"


def stage_lora_pair_text_lengths(
    edit_length: int,
    reprompt_length: int,
    max_tokens: int,
) -> Tuple[int, int, int]:
    """Return retained edit, reprompt, and combined lengths.

    Edit tokens have priority. Only the required zero separator is reserved;
    the reprompt may be truncated to zero retained tokens when the edit consumes
    the remaining budget. Its source text is still required and encoded.
    """

    edit_length = int(edit_length)
    reprompt_length = int(reprompt_length)
    max_tokens = int(max_tokens)
    if edit_length <= 0:
        raise ValueError(
            f"StageLoRA edit token length must be positive, got {edit_length}."
        )
    if reprompt_length <= 0:
        raise ValueError(
            f"StageLoRA reprompt token length must be positive, got {reprompt_length}."
        )
    if max_tokens < 2:
        raise ValueError(
            "StageLoRA reprompt text needs t5_max_tokens >= 2 for an edit token "
            f"and the required separator; got {max_tokens}."
        )

    retained_edit = min(edit_length, max_tokens - STAGE_LORA_TEXT_SEPARATOR_TOKENS)
    retained_reprompt = min(
        reprompt_length,
        max_tokens - retained_edit - STAGE_LORA_TEXT_SEPARATOR_TOKENS,
    )
    combined = retained_edit + STAGE_LORA_TEXT_SEPARATOR_TOKENS + retained_reprompt
    return retained_edit, retained_reprompt, combined


def stage_lora_combined_text_len(
    edit_length: int,
    reprompt_length: int,
    max_tokens: int,
) -> int:
    """Return the packed length for one edit/reprompt pair."""

    return stage_lora_pair_text_lengths(edit_length, reprompt_length, max_tokens)[2]


def _flatten_text_groups(
    groups: Sequence[Sequence[object]],
    *,
    field_name: str,
    add_t2v_prefix: bool,
    allow_empty: bool = False,
) -> Tuple[List[int], List[str]]:
    if not isinstance(groups, (list, tuple)) or not groups:
        raise ValueError(f"StageLoRA {field_name} must be a non-empty list of groups.")

    group_sizes: List[int] = []
    flattened: List[str] = []
    for sample_index, group in enumerate(groups):
        if not isinstance(group, (list, tuple)) or not group:
            raise ValueError(
                f"StageLoRA {field_name} group {sample_index} must be a non-empty list/tuple."
            )
        if len(group) != 1:
            raise ValueError(
                f"StageLoRA {field_name} group {sample_index} must contain exactly one "
                f"edit instruction, got {len(group)}."
            )
        group_sizes.append(len(group))
        for value in group:
            if allow_empty and isinstance(value, str) and not value.strip():
                flattened.append("")
            elif add_t2v_prefix:
                flattened.append(add_stage_lora_t2v_prefix(value))
            else:
                flattened.append(_clean_required_text(value, field_name=field_name))
    return group_sizes, flattened


def _flatten_reprompts(
    reprompts: Optional[Sequence[Sequence[object]]],
    caption_group_sizes: Sequence[int],
) -> List[str]:
    if reprompts is None:
        raise ValueError(
            "StageLoRA reprompt text is enabled, but the batch has no reprompts."
        )
    if not isinstance(reprompts, (list, tuple)):
        raise ValueError(
            f"StageLoRA reprompts must be a list of groups, got {type(reprompts)!r}."
        )
    if len(reprompts) != len(caption_group_sizes):
        raise ValueError(
            "StageLoRA reprompt/caption sample count mismatch: "
            f"{len(reprompts)} != {len(caption_group_sizes)}."
        )

    flattened: List[str] = []
    for sample_index, (group, expected_size) in enumerate(
        zip(reprompts, caption_group_sizes)
    ):
        if not isinstance(group, (list, tuple)):
            raise ValueError(
                f"StageLoRA reprompt group {sample_index} must be a list/tuple."
            )
        if len(group) != expected_size:
            raise ValueError(
                f"StageLoRA reprompt/caption count mismatch at sample {sample_index}: "
                f"{len(group)} != {expected_size}."
            )
        flattened.extend(clean_stage_lora_reprompt(value) for value in group)
    return flattened


def _encode_t5_once(text_encoder, texts, *, device, max_tokens):
    if not texts or int(max_tokens) <= 0:
        raise ValueError(
            "Text encoding requires a non-empty batch and positive token budget."
        )
    tokenizer = text_encoder.tokenizer
    previous_length = tokenizer.seq_len
    tokenizer.seq_len = int(max_tokens)
    try:
        encoded = text_encoder(list(texts), device)
    finally:
        tokenizer.seq_len = previous_length
    return [feature[:max_tokens].float() for feature in encoded]


def _pack_official_text_condition(
    features: Sequence[torch.Tensor],
    caption_nums: Sequence[int],
) -> OfficialTextCondition:
    if not features:
        raise ValueError("StageLoRA cannot pack an empty T5 feature list.")
    lengths = [int(feature.shape[0]) for feature in features]
    if any(length <= 0 for length in lengths):
        raise ValueError(f"StageLoRA packed text lengths must be positive: {lengths}.")
    compact = torch.cat(list(features), dim=0)
    cu_seqlens = torch.tensor(
        [0, *torch.tensor(lengths, dtype=torch.int64).cumsum(0).tolist()],
        dtype=torch.int32,
    )
    return compact, lengths, cu_seqlens, max(lengths), [int(v) for v in caption_nums]


def _pooled_edit_context(features: Sequence[torch.Tensor]) -> torch.Tensor:
    return torch.stack([feature.mean(dim=0) for feature in features], dim=0).detach()


def _encode_edit_reprompt_features(
    text_encoder,
    edits: Sequence[str],
    reprompts: Sequence[str],
    *,
    device: object,
    max_tokens: int,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Encode conditional edit/reprompt pairs and official empty CFG prompts once."""

    t5_inputs: List[str] = []
    feature_indices: List[Tuple[int, Optional[int]]] = []
    for edit, reprompt in zip(edits, reprompts):
        edit_index = len(t5_inputs)
        t5_inputs.append(edit)
        reprompt_index = None
        if edit:
            reprompt_index = len(t5_inputs)
            t5_inputs.append(reprompt)
        feature_indices.append((edit_index, reprompt_index))

    encoded = _encode_t5_once(
        text_encoder,
        t5_inputs,
        device=device,
        max_tokens=max_tokens,
    )
    combined_features: List[torch.Tensor] = []
    retained_edit_features: List[torch.Tensor] = []
    for edit_index, reprompt_index in feature_indices:
        edit_feature = encoded[edit_index]
        if reprompt_index is None:
            combined_features.append(edit_feature)
            retained_edit_features.append(edit_feature)
            continue

        reprompt_feature = encoded[reprompt_index]
        retained_edit, retained_reprompt, _ = stage_lora_pair_text_lengths(
            edit_feature.shape[0],
            reprompt_feature.shape[0],
            max_tokens,
        )
        separator = edit_feature.new_zeros(
            (STAGE_LORA_TEXT_SEPARATOR_TOKENS, edit_feature.shape[1])
        )
        combined_features.append(
            torch.cat(
                (
                    edit_feature[:retained_edit],
                    separator,
                    reprompt_feature[:retained_reprompt],
                ),
                dim=0,
            )
        )
        retained_edit_features.append(edit_feature[:retained_edit])
    return combined_features, retained_edit_features


def encode_stage_lora_text_with_edit_context(
    text_encoder,
    captions: Sequence[Sequence[object]],
    reprompts: Optional[Sequence[Sequence[object]]] = None,
    *,
    use_reprompt: bool,
    device: object,
    t5_max_tokens: int,
) -> StageLoRATextCondition:
    """Encode official backbone text and the raw edit-only gate context once.

    With reprompt disabled, only edit instructions are encoded. With reprompt
    enabled, conditional samples are packed as ``edit + zero separator +
    reprompt``. An empty edit is the official CFG-drop signal and is encoded by
    itself, so its reprompt cannot leak into the unconditional branch. The edit
    is retained before the reprompt under ``t5_max_tokens``. The independent
    gate context always pools only the effective edit tokens.
    """

    caption_nums, edits = _flatten_text_groups(
        captions,
        field_name="captions",
        add_t2v_prefix=True,
        allow_empty=True,
    )
    max_tokens = int(t5_max_tokens)

    if not use_reprompt:
        edit_features = _encode_t5_once(
            text_encoder,
            edits,
            device=device,
            max_tokens=max_tokens,
        )
        return StageLoRATextCondition(
            backbone=_pack_official_text_condition(edit_features, caption_nums),
            edit_context=_pooled_edit_context(edit_features),
        )

    flattened_reprompts = _flatten_reprompts(reprompts, caption_nums)
    if len(flattened_reprompts) != len(edits):
        raise ValueError(
            "StageLoRA flattened reprompt/edit count mismatch: "
            f"{len(flattened_reprompts)} != {len(edits)}."
        )

    combined_features, retained_edit_features = _encode_edit_reprompt_features(
        text_encoder,
        edits,
        flattened_reprompts,
        device=device,
        max_tokens=max_tokens,
    )
    return StageLoRATextCondition(
        backbone=_pack_official_text_condition(combined_features, caption_nums),
        edit_context=_pooled_edit_context(retained_edit_features),
    )


def encode_stage_lora_negative_text(
    text_encoder,
    negative_prompts: Sequence[Sequence[object]],
    *,
    device: object,
    t5_max_tokens: int,
) -> StageLoRATextCondition:
    """Encode the official unprefixed CFG branch and its source-gate context."""

    caption_nums, texts = _flatten_text_groups(
        negative_prompts,
        field_name="negative prompts",
        add_t2v_prefix=False,
        allow_empty=True,
    )
    features = _encode_t5_once(
        text_encoder,
        texts,
        device=device,
        max_tokens=int(t5_max_tokens),
    )
    return StageLoRATextCondition(
        backbone=_pack_official_text_condition(features, caption_nums),
        edit_context=_pooled_edit_context(features),
    )


__all__ = [
    "STAGE_LORA_T2V_PREFIX",
    "STAGE_LORA_TEXT_SEPARATOR_TOKENS",
    "OfficialTextCondition",
    "StageLoRATextCondition",
    "add_stage_lora_t2v_prefix",
    "clean_stage_lora_edit_instruction",
    "clean_stage_lora_reprompt",
    "encode_stage_lora_negative_text",
    "encode_stage_lora_text_with_edit_context",
    "stage_lora_combined_text_len",
    "stage_lora_pair_text_lengths",
]
