"""Portable Stage-LoRA checkpoint loading and architecture recovery."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .topology import StageLoRATopology

INFERENCE_RUNTIME_FIELDS = (
    "use_reprompt_text",
    "video_caption_type",
    "text_channels",
    "t5_max_tokens",
    "norm_eps",
    "num_of_label_value",
    "pn",
    "train_h_div_w_list",
    "apply_spatial_patchify",
    "dynamic_scale_schedule",
    "video_frames",
    "video_fps",
    "min_video_frames",
    "temporal_compress_rate",
    "duration_resolution",
    "detail_scale_dim",
    "detail_num_lvl",
    "hbq_round",
    "rope2d_normalized_by_hw",
    "use_ada_layer_norm",
    "add_scale_token",
    "refine_mode",
    "use_slice",
    "corruption_alpha",
    "log_norm_mean",
    "log_norm_sigma",
    "train_max_token_len",
    "tf32",
)


def make_inference_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Export only portable inference settings, never training paths or data metadata."""
    normalized = normalize_stage_lora_checkpoint(manifest)
    return {
        f"stage_lora_runtime_{name}": runtime_value(normalized, name)
        for name in INFERENCE_RUNTIME_FIELDS
    }


def normalize_stage_lora_checkpoint(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Translate only the current main-repository Stage I naming at the boundary."""

    def normalize(value):
        if not isinstance(value, Mapping):
            return value
        result = {}
        for key, item in value.items():
            name = key
            if isinstance(name, str) and name.startswith("stage1_lora_"):
                name = "stage_lora_" + name[len("stage1_lora_") :]
            elif isinstance(name, str) and name.startswith("stage1_"):
                name = "stage_lora_" + name[len("stage1_") :]
            result[name] = normalize(item)
        if result.get("adapter_name") == "stage1":
            result["adapter_name"] = "stage_lora"
        return result

    normalized = normalize(payload)
    trainer = normalized.get("trainer", {})
    if isinstance(trainer.get("gpt_fsdp"), Mapping):
        trainer["gpt_fsdp"] = normalize_model_state(trainer["gpt_fsdp"])
    optimizer = trainer.get("gpt_fsdp_opt")
    if isinstance(optimizer, Mapping):
        optimizer["state"] = {
            _parameter_name(name): value for name, value in optimizer["state"].items()
        }
        optimizer["param_groups"] = [
            {**group, "params": [_parameter_name(name) for name in group["params"]]}
            for group in optimizer["param_groups"]
        ]
    return normalized


def _parameter_name(name):
    if not isinstance(name, str):
        return name  # Ordinary optimizer checkpoints use integer parameter IDs.
    wrappers = ("module.", "_orig_mod.", "_fsdp_wrapped_module.")
    while name.startswith(wrappers):
        name = name.split(".", 1)[1]
    return name.replace(".lora_A.stage1.", ".lora_A.stage_lora.").replace(
        ".lora_B.stage1.", ".lora_B.stage_lora."
    )


def normalize_model_state(state: Mapping[str, Any]) -> dict[str, Any]:
    normalized = {}
    for key, tensor in state.items():
        name = _parameter_name(key)
        if name in normalized:
            raise ValueError(f"Duplicate normalized checkpoint key: {name}")
        normalized[name] = tensor
    return normalized


def recover_topology(
    metadata: Mapping[str, Any], state: Mapping[str, Any]
) -> StageLoRATopology:
    """Recover branch positions and ranks from tensors, checking their metadata.

    LoRA alpha, dropout and text-conditioning semantics are not encoded in the
    weights. They must remain in checkpoint metadata; no default is guessed.
    """

    topology = StageLoRATopology.from_dict(metadata)
    active = tuple(
        sorted(
            int(match.group(1))
            for name in state
            if (match := re.fullmatch(r"source_word_embeds\.(\d+)\.weight", name))
        )
    )
    if not active:
        raise ValueError("Checkpoint contains no Stage-LoRA source branches.")
    num_chunks = len(topology.source_chunk_mask)
    if active[-1] >= num_chunks:
        raise ValueError("Source branch index is outside the recorded backbone chunks.")
    mask = tuple(index in active for index in range(num_chunks))
    if mask != topology.source_chunk_mask or active != topology.active_chunk_ids:
        raise ValueError("Source branch tensors disagree with the checkpoint topology.")

    down = state.get("source_text_gate.down.weight")
    if down is None or tuple(down.shape) != (
        topology.source_text_gate_rank,
        topology.text_channels,
    ):
        raise ValueError(
            "Source text-gate tensor dimensions disagree with the topology."
        )
    gate_ids = tuple(
        sorted(
            int(match.group(1))
            for name in state
            if (
                match := re.fullmatch(
                    r"source_text_gate\.up_by_chunk\.(\d+)\.weight", name
                )
            )
        )
    )
    if gate_ids != active:
        raise ValueError(
            "Source projections and text gates have different active chunks."
        )

    targets = {}
    for name, tensor in state.items():
        match = re.fullmatch(r"(.+)\.lora_A(?:\.stage_lora)?\.weight", name)
        if match:
            targets[match.group(1)] = int(tensor.shape[0])
    if set(targets) != set(topology.lora_targets):
        raise ValueError(
            "LoRA tensor targets disagree with the recorded target modules."
        )
    if set(targets.values()) != {topology.lora_rank}:
        raise ValueError("LoRA tensor ranks disagree with the recorded rank.")
    for target in targets:
        full_name = f"{target}.lora_B.stage_lora.weight"
        compact_name = f"{target}.lora_B.weight"
        tensor = state.get(full_name, state.get(compact_name))
        if tensor is None or int(tensor.shape[1]) != topology.lora_rank:
            raise ValueError(f"Missing or incompatible LoRA B tensor: {target}")
    return topology


@dataclass
class LoadedStageLoRACheckpoint:
    manifest: dict[str, Any]
    topology: StageLoRATopology
    state: dict[str, Any] | None
    adapter_dir: Path | None
    payload: dict[str, Any]


def load_stage_lora_checkpoint(
    path: str | Path,
    *,
    runtime_path: str | Path | None = None,
) -> LoadedStageLoRACheckpoint:
    """Load a current full trainer checkpoint or an explicit compact export.

    Full checkpoints are self-contained model weights. Compact exports require
    the base GRN weights and a runtime.json manifest for inference.
    """

    import torch

    checkpoint_path = Path(path).expanduser().resolve()
    if checkpoint_path.is_dir() and (checkpoint_path / "SHA256SUMS").is_file():
        from grn.utils.checkpoint_parts import resolve_checkpoint_path

        checkpoint_path = resolve_checkpoint_path(checkpoint_path)
    if checkpoint_path.is_dir():
        from safetensors.torch import load_file

        config_path = checkpoint_path / "stage_lora_config.json"
        if not config_path.is_file():
            config_path = checkpoint_path / "stage1_config.json"
        metadata = json.loads(config_path.read_text(encoding="utf-8"))
        runtime_file = (
            Path(runtime_path) if runtime_path else checkpoint_path / "runtime.json"
        )
        if not runtime_file.is_file():
            raise FileNotFoundError(
                "Compact inference needs runtime.json from its training checkpoint; "
                "supply --runtime_path for a main-repository export without it."
            )
        runtime = json.loads(runtime_file.read_text(encoding="utf-8"))
        manifest = normalize_stage_lora_checkpoint(runtime.get("args", runtime))
        state = normalize_model_state(
            {
                **load_file(str(checkpoint_path / "adapter_model.safetensors")),
                **load_file(str(checkpoint_path / "source_branch.safetensors")),
            }
        )
        topology = recover_topology(metadata, state)
        return LoadedStageLoRACheckpoint(manifest, topology, None, checkpoint_path, {})

    payload = torch.load(
        str(checkpoint_path), map_location="cpu", weights_only=False, mmap=True
    )
    if not isinstance(payload, Mapping):
        raise ValueError(
            "Stage-LoRA inference requires a full trainer checkpoint or a compact export."
        )
    payload = normalize_stage_lora_checkpoint(payload)
    manifest, trainer = payload.get("args"), payload.get("trainer")
    if not isinstance(manifest, Mapping) or not isinstance(trainer, Mapping):
        raise ValueError("Stage-LoRA checkpoint is missing args or trainer state.")
    metadata, state = trainer.get("stage_lora_topology"), trainer.get("gpt_fsdp")
    if not isinstance(metadata, Mapping) or not isinstance(state, Mapping) or not state:
        raise ValueError(
            "Stage-LoRA checkpoint needs trainer.stage_lora_topology and trainer.gpt_fsdp."
        )
    topology = recover_topology(metadata, state)
    return LoadedStageLoRACheckpoint(
        dict(manifest), topology, dict(state), None, payload
    )


def runtime_value(manifest: Mapping[str, Any], name: str) -> Any:
    key = f"stage_lora_runtime_{name}"
    if key not in manifest:
        raise ValueError(f"Checkpoint is missing the required inference setting: {key}")
    return manifest[key]
