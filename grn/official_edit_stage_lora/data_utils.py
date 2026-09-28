"""Source-video metadata primitives shared by training and inference."""

import math
from os import path as osp
from typing import Any, Mapping
import numpy as np
from .text_conditioning import clean_stage_lora_edit_instruction


class StageLoRAPairAlignmentError(ValueError):
    """A frame interval that violates the source/target correspondence."""


def extract_single_text(value: object, field_name: str) -> str:
    if isinstance(value, Mapping):
        value = value.get("content")
    elif isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError(
                f"StageLoRA {field_name!r} must contain exactly one instruction, "
                f"got {len(value)}."
            )
        value = value[0]
        if isinstance(value, Mapping):
            value = value.get("content")
    return clean_stage_lora_edit_instruction(value)


def resolve_media_path(value: object, *, field_name: str, meta_file: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"StageLoRA metadata requires non-empty {field_name}.")
    path = osp.expanduser(value.strip())
    if not osp.isabs(path):
        path = osp.join(osp.dirname(meta_file), path)
    path = osp.abspath(path)
    if not osp.isfile(path):
        raise FileNotFoundError(f"StageLoRA {field_name} does not exist: {path}")
    return path


def shared_sample_indices(
    meta: Mapping[str, Any],
    total_frames: int,
    sampling_fps: float,
) -> np.ndarray:
    begin_frame_id = int(meta["begin_frame_id"])
    sample_frames = int(meta["sample_frames"])
    sample_seconds = (sample_frames - 1) / float(sampling_fps)
    window_end = begin_frame_id + math.ceil(float(meta["fps"]) * sample_seconds)
    if window_end <= begin_frame_id or window_end > min(
        int(meta["end_frame_id"]), int(total_frames)
    ):
        raise StageLoRAPairAlignmentError(
            "shared absolute frame window exceeds the declared paired interval: "
            f"begin={begin_frame_id}, end={window_end}, "
            f"declared_end={meta['end_frame_id']}, total_frames={total_frames}."
        )
    return np.linspace(
        begin_frame_id,
        window_end - 1,
        sample_frames,
        dtype=int,
    )
