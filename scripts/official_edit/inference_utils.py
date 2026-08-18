from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import os.path as osp
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import cv2
import imageio
from PIL import Image

from grn.dataset.official_t2iv.dataset_edit_pair import list_official_edit_jsonls
from grn.dataset.official_t2iv.dataset_joint_vi import transform
from grn.models.umt5.t5 import T5EncoderModel
from grn.schedules.dynamic_resolution import get_dynamic_resolution_meta
from grn.utils.video_decoder import EncodedVideoDecord
from grn.utils_t2iv.hbq_util_t2iv import raw_feature2bit_label
from scripts.official_edit.resource_utils import resolve_t5_paths


def images2video(frames, fps: int, save_filepath: str) -> str:
    """Write BGR uint8 frames as an image or MP4."""
    save_dir = osp.dirname(save_filepath)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    if len(frames) == 1:
        output_path = osp.splitext(save_filepath)[0] + ".jpg"
        cv2.imwrite(output_path, frames[0])
    else:
        output_path = save_filepath
        imageio.mimsave(output_path, frames[..., ::-1], fps=fps)
    return output_path


def _str2bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid bool value: {value!r}")


def _safe_name(value: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._-")
    return name or "dataset"


def _short_hash(value: str, length: int = 10) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def _sample_seed(base_seed: int, uid: str) -> int:
    offset = int(hashlib.sha256(str(uid).encode("utf-8")).hexdigest()[:8], 16)
    return (int(base_seed) + offset) % (2**32 - 1)


def _require_file(path: str, role: str) -> None:
    if not path or not osp.isfile(path):
        raise FileNotFoundError(f"{role} does not exist: {path}")


def _split_repeat(value: str) -> Tuple[str, float]:
    if "@" not in value:
        return value.strip(), 1.0
    name, repeat = value.rsplit("@", 1)
    repeat_value = float(repeat)
    if repeat_value <= 0:
        raise ValueError(f"Repeat must be positive in {value!r}.")
    return name.strip(), repeat_value


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    _require_file(str(path), "metadata file")
    rows = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} must be a JSON object.")
            rows.append(row)
    return rows


def _select_indices(count: int, repeat: float, rng: np.random.Generator) -> np.ndarray:
    if count <= 0:
        return np.empty((0,), dtype=np.int64)
    full_repeats = int(math.floor(repeat))
    fractional_count = int(count * (repeat - full_repeats))
    parts = []
    if full_repeats:
        parts.append(np.tile(np.arange(count, dtype=np.int64), full_repeats))
    if fractional_count:
        fractional = np.arange(count, dtype=np.int64)
        rng.shuffle(fractional)
        parts.append(fractional[:fractional_count])
    if not parts:
        return np.empty((0,), dtype=np.int64)
    selected = np.concatenate(parts)
    rng.shuffle(selected)
    return selected


def _official_caption(meta: Dict[str, Any]) -> str:
    caption = str(meta.get("tarsier2_caption") or "").strip()
    if not caption:
        value = meta.get("caption")
        if isinstance(value, list) and value:
            first = value[0]
            caption = str(first.get("content") if isinstance(first, dict) else first).strip()
        elif isinstance(value, str):
            caption = value.strip()
    if not caption:
        raise ValueError("Official edit metadata is missing caption text.")
    quality_prompt = str(meta.get("quality_prompt") or "").strip()
    return f"{caption} {quality_prompt}".strip()


def _load_official_dataset_samples(
    *,
    root: str,
    subdirs: str,
    num_samples_per_dataset: int,
    shuffle: bool,
    seed: int,
) -> List[Dict[str, Any]]:
    root_path = Path(root)
    if not root_path.is_dir():
        raise FileNotFoundError(f"Metadata root does not exist: {root}")
    # A missing subdir means "use the metadata root itself".  Using the
    # textual root here would join a relative root to itself below (for
    # example data/metadata/data/metadata).
    specs = [item.strip() for item in subdirs.split(",") if item.strip()] if subdirs else ["."]
    rng = np.random.default_rng(seed)
    samples: List[Dict[str, Any]] = []
    for spec in specs:
        name_or_path, repeat = _split_repeat(spec)
        dataset_dir = Path(name_or_path)
        if not dataset_dir.is_absolute():
            dataset_dir = root_path / name_or_path
        jsonl_paths = sorted(Path(item) for item in list_official_edit_jsonls(str(dataset_dir)))
        if not jsonl_paths:
            raise FileNotFoundError(f"No JSONL metadata found in {dataset_dir}.")
        rows: List[Tuple[str, int, Dict[str, Any]]] = []
        for jsonl_path in jsonl_paths:
            rows.extend(
                (str(jsonl_path), row_index, row)
                for row_index, row in enumerate(_read_jsonl(jsonl_path))
            )
        selected = _select_indices(len(rows), repeat, rng)
        if shuffle:
            rng.shuffle(selected)
        if num_samples_per_dataset > 0:
            selected = selected[:num_samples_per_dataset]
        dataset_name = _safe_name(dataset_dir.name)
        for logical_index, row_id in enumerate(selected.tolist()):
            meta_file, row_index, raw_meta = rows[int(row_id)]
            meta = dict(raw_meta)
            meta["meta_file"] = meta_file
            meta["meta_line_no"] = row_index + 1
            source_path = str(meta.get("source_path") or "").strip()
            _require_file(source_path, "Source video")
            meta["source_path"] = source_path
            uid_source = f"{dataset_name}|{meta_file}|{row_index}|{logical_index}"
            samples.append(
                {
                    "dataset": dataset_name,
                    "index": int(row_index),
                    "logical_index": int(logical_index),
                    "uid": f"{logical_index:06d}_{_short_hash(uid_source)}",
                    "meta": meta,
                }
            )
    return samples


def _rank_info(cli_gpus: int, workers_per_gpu: int = 1) -> Tuple[int, int, int, int]:
    if "LOCAL_RANK" not in os.environ:
        return 0, 1, 0, 0
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ.get("RANK", local_rank))
    world_size = int(
        os.environ.get("WORLD_SIZE", max(1, cli_gpus) * max(1, workers_per_gpu))
    )
    return rank, world_size, local_rank, local_rank % max(1, cli_gpus)


def _load_text_encoder(args: SimpleNamespace, device: torch.device):
    checkpoint_path, tokenizer_path = resolve_t5_paths(args.t5_path)
    return T5EncoderModel(
        text_len=args.t5_max_tokens,
        dtype=torch.bfloat16,
        device=device,
        checkpoint_path=checkpoint_path,
        tokenizer_path=tokenizer_path,
        enable_fsdp=False,
    )


def _extract_model_state(path: str, use_ema: bool) -> Dict[str, torch.Tensor]:
    _require_file(path, "Checkpoint")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    trainer_state = checkpoint.get("trainer") if isinstance(checkpoint, dict) else None
    if not isinstance(trainer_state, dict):
        raise ValueError("GRNEdit inference requires a complete training checkpoint.")
    state_name = "gpt_ema_fsdp" if use_ema else "gpt_fsdp"
    state = trainer_state.get(state_name)
    if not isinstance(state, dict) or not state:
        raise KeyError(f"Checkpoint trainer state has no non-empty {state_name}.")
    return state


def _infer_block_chunks_from_state(state: Dict[str, torch.Tensor]) -> Optional[int]:
    chunk_indices = []
    for key in state:
        clean_key = str(key)
        changed = True
        while changed:
            changed = False
            for prefix in ("module.", "_orig_mod.", "_fsdp_wrapped_module."):
                if clean_key.startswith(prefix):
                    clean_key = clean_key[len(prefix):]
                    changed = True
        match = re.match(r"block_chunks\.(\d+)\.module\.", clean_key)
        if match:
            chunk_indices.append(int(match.group(1)))
    return max(chunk_indices) + 1 if chunk_indices else None


def _encode_t5(text_encoder, texts: Sequence[str], args: SimpleNamespace, device: torch.device):
    del args
    features = text_encoder(list(texts), device)
    lengths = [int(feature.shape[0]) for feature in features]
    return torch.cat(features, dim=0).float(), lengths


def _mapped_sample_frames(meta: Dict[str, Any], args: SimpleNamespace) -> int:
    for key in ("begin_frame_id", "end_frame_id", "fps"):
        if key not in meta:
            raise KeyError(f"Official edit metadata is missing {key}.")
    real_duration = (int(meta["end_frame_id"]) - int(meta["begin_frame_id"])) / float(meta["fps"])
    mapped_duration = int(np.round(real_duration / args.duration_resolution)) * args.duration_resolution
    max_duration = (int(args.video_frames) - 1) // int(args.video_fps)
    if mapped_duration > max_duration:
        if int(args.drop_long_video):
            raise ValueError(f"Sample duration {mapped_duration} exceeds {max_duration}.")
        mapped_duration = max_duration
    sample_frames = int(mapped_duration * int(args.video_fps) + 1)
    if sample_frames <= 1:
        raise ValueError(f"Invalid sample_frames={sample_frames}.")
    return sample_frames


def _tensor_t3hw_to_bgr(video_t3hw: torch.Tensor) -> np.ndarray:
    video = ((video_t3hw.float() + 1.0) / 2.0).clamp(0, 1)
    video = video.permute(0, 2, 3, 1).mul(255).to(torch.uint8).cpu().numpy()
    return video[..., ::-1].copy()


def _read_video_like_training(
    *,
    path: str,
    meta: Dict[str, Any],
    args: SimpleNamespace,
    dynamic_resolution_h_w,
    h_div_w_templates,
    target_hw: Optional[Tuple[int, int]] = None,
) -> Tuple[torch.Tensor, np.ndarray, float, Tuple[int, int]]:
    _require_file(path, "Video")
    sample_frames = int(meta["sample_frames"])
    video = EncodedVideoDecord(path, osp.basename(path), num_threads=0)
    try:
        begin_frame_id = int(meta["begin_frame_id"])
        start = max(0.0, begin_frame_id / float(video._fps))
        end = start + (sample_frames - 1) / int(args.video_fps)
        if end > video.duration + 0.2:
            raise ValueError(
                f"Training-style clip exceeds video duration: "
                f"end={end}, duration={video.duration}, path={path}"
            )
        end = min(end, video.duration)
        raw_video_rgb, _ = video.get_clip(start, end, sample_frames)
    finally:
        video.close()
    if len(raw_video_rgb) != sample_frames:
        raise RuntimeError(f"Decoded {len(raw_video_rgb)} frames, expected {sample_frames}: {path}")
    height, width = raw_video_rgb[0].shape[:2]
    ratio = height / width
    ratio_template = float(h_div_w_templates[np.argmin(np.abs(h_div_w_templates - ratio))])
    target_height, target_width = (
        dynamic_resolution_h_w[ratio_template][args.pn]["pixel"] if target_hw is None else target_hw
    )
    frames = [
        transform(Image.fromarray(frame).convert("RGB"), int(target_height), int(target_width))
        for frame in raw_video_rgb
    ]
    video_t3hw = torch.stack(frames, dim=0).contiguous()
    return (
        video_t3hw,
        _tensor_t3hw_to_bgr(video_t3hw),
        ratio_template,
        (int(target_height), int(target_width)),
    )


def _encode_source_bits(
    vae,
    video_t3hw: torch.Tensor,
    args: SimpleNamespace,
    device: torch.device,
) -> torch.Tensor:
    input_cthw = video_t3hw.permute(1, 0, 2, 3).contiguous().to(device, non_blocking=True)
    with torch.no_grad(), torch.amp.autocast(
        "cuda",
        enabled=device.type == "cuda",
        dtype=torch.float32,
    ):
        raw_features, _, _ = vae.encode_for_raw_features(
            input_cthw.unsqueeze(0),
            scale_schedule=None,
            slice=args.use_slice,
        )
    return raw_feature2bit_label(raw_features[0], hbq_round=args.hbq_round)
