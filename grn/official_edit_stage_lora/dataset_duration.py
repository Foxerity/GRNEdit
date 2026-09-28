"""Canonical duration and VAE-frame policy for StageLoRA data."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class StageLoRATemporalWindow:
    sample_seconds: float
    sample_frames: int
    latent_frames: int


def resolve_stage_lora_temporal_window(
    mapped_seconds: float,
    *,
    available_seconds: float,
    sampling_fps: float,
    temporal_compress_rate: int,
    minimum_seconds: float,
    maximum_seconds: float,
) -> StageLoRATemporalWindow:
    """Bound a clip to available data and align it to ``rate * N + 1``."""

    mapped_seconds = float(mapped_seconds)
    available_seconds = float(available_seconds)
    sampling_fps = float(sampling_fps)
    temporal_compress_rate = int(temporal_compress_rate)
    minimum_seconds = float(minimum_seconds)
    maximum_seconds = float(maximum_seconds)
    values = {
        "mapped_seconds": mapped_seconds,
        "available_seconds": available_seconds,
        "sampling_fps": sampling_fps,
        "minimum_seconds": minimum_seconds,
        "maximum_seconds": maximum_seconds,
    }
    invalid = [name for name, value in values.items() if not math.isfinite(value)]
    if invalid:
        raise ValueError(f"StageLoRA duration values must be finite: {invalid}")
    if (
        mapped_seconds < 0
        or available_seconds <= 0
        or sampling_fps <= 0
        or temporal_compress_rate <= 0
        or maximum_seconds <= 0
        or minimum_seconds > maximum_seconds
    ):
        raise ValueError(
            "StageLoRA temporal sampling configuration is invalid: "
            f"mapped={mapped_seconds}, available={available_seconds}, "
            f"fps={sampling_fps}, rate={temporal_compress_rate}, "
            f"minimum={minimum_seconds}, maximum={maximum_seconds}."
        )

    if mapped_seconds < minimum_seconds:
        raise ValueError(
            f"mapped duration {mapped_seconds:g}s is below the configured minimum "
            f"{minimum_seconds:g}s."
        )
    bounded_seconds = min(mapped_seconds, maximum_seconds, available_seconds)
    frame_budget = int(bounded_seconds * sampling_fps + 1)
    sample_frames = (
        (frame_budget - 1) // temporal_compress_rate
    ) * temporal_compress_rate + 1
    sample_seconds = (sample_frames - 1) / sampling_fps
    if sample_seconds < minimum_seconds:
        raise ValueError(
            f"VAE-aligned duration {sample_seconds:g}s is below the configured "
            f"minimum {minimum_seconds:g}s."
        )
    return StageLoRATemporalWindow(
        sample_seconds=sample_seconds,
        sample_frames=sample_frames,
        latent_frames=(sample_frames - 1) // temporal_compress_rate + 1,
    )


__all__ = ["StageLoRATemporalWindow", "resolve_stage_lora_temporal_window"]
