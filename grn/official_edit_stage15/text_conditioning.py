"""Text helpers for Stage1.5 source-conditioned edit training/inference."""

from __future__ import annotations


STAGE15_TEXT_PAIR_MARKER = "stage15_t5_pair_v1"
STAGE15_TEXT_SEPARATOR_TOKENS = 1
STAGE15_TEXT_SCHEMA = "stage15_t5_batched_pair_zero_separator_v1"
STAGE15_T2V_PREFIX = "<T2V>"
STAGE15_T2V_PREFIX_WITH_SPACE = f"{STAGE15_T2V_PREFIX} "


def add_stage15_t2v_prefix(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return text
    if text.startswith(STAGE15_T2V_PREFIX):
        suffix = text[len(STAGE15_T2V_PREFIX):].lstrip()
        return STAGE15_T2V_PREFIX_WITH_SPACE + suffix if suffix else STAGE15_T2V_PREFIX
    return STAGE15_T2V_PREFIX_WITH_SPACE + text


def _clean_required_text(value: object, *, field_name: str) -> str:
    text = str(value or "").strip()
    if not text or text.lower() in {"null", "none", "n/a", "na"}:
        raise ValueError(f"Stage1.5 reprompt text mode requires non-empty {field_name}.")
    return text


def clean_stage15_edited_prompt(value: object) -> str:
    return _clean_required_text(value, field_name="edited prompt")


def clean_stage15_reprompt(value: object) -> str:
    return _clean_required_text(value, field_name="reprompt")


def stage15_combined_text_len(edited_len: int, reprompt_len: int, max_tokens: int) -> int:
    edited_len = max(0, int(edited_len))
    reprompt_len = max(0, int(reprompt_len))
    max_tokens = int(max_tokens)
    if max_tokens <= 0:
        raise ValueError(f"max_tokens must be positive, got {max_tokens}.")
    edited_len = min(edited_len, max_tokens)
    remaining = max_tokens - edited_len
    if reprompt_len <= 0 or remaining <= STAGE15_TEXT_SEPARATOR_TOKENS:
        return edited_len
    return edited_len + STAGE15_TEXT_SEPARATOR_TOKENS + min(reprompt_len, remaining - STAGE15_TEXT_SEPARATOR_TOKENS)
