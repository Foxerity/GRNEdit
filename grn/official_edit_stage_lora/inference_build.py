"""Stage-LoRA model and tokenizer construction for inference."""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping

import torch
import torch.nn as nn

from grn.models.hbq_tokenizer import HBQ_Tokenizer

from .lora import (
    PEFT_VERSION,
    TrainableParameterStats,
    configure_stage_lora_trainable_parameters,
    inject_stage_lora,
    load_compact_stage_lora,
)
from .model import (
    StageLoRAGRN,
    create_stage_lora_backbone,
)
from .topology import (
    ADAPTER_NAME,
    OFFICIAL_GRN_COMMIT,
    StageLoRATopology,
    build_lora_targets,
    parse_chunk_mask,
)


OFFICIAL_BASE_FILENAMES = {
    "GRN2b": "GRN_T2V_2B_FSA_sft147400_ema.pth",
}
OFFICIAL_VAE_FILENAME = "HBQ_image_video_tokenizer_64dim_M4_20260626.ckpt"


@contextmanager
def disable_torch_default_parameter_reset() -> Iterator[None]:
    """Temporarily skip only the two reset methods required by the protocol."""

    linear_reset = nn.Linear.reset_parameters
    layer_norm_reset = nn.LayerNorm.reset_parameters
    nn.Linear.reset_parameters = lambda self: None
    nn.LayerNorm.reset_parameters = lambda self: None
    try:
        yield
    finally:
        nn.Linear.reset_parameters = linear_reset
        nn.LayerNorm.reset_parameters = layer_norm_reset


def resolve_base_checkpoint(
    model_name: str, weights_dir: str | os.PathLike[str], override: str = ""
) -> Path:
    if override:
        path = Path(override).expanduser().resolve()
    else:
        filename = OFFICIAL_BASE_FILENAMES.get(model_name)
        if filename is None:
            raise ValueError(
                f"{model_name} has no official StageLoRA default checkpoint; pass --base_checkpoint"
            )
        path = (Path(weights_dir).expanduser() / filename).resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"StageLoRA base checkpoint does not exist: {path}. "
            "A complete official or later-pretrained GRN checkpoint is mandatory."
        )
    return path


def resolve_vae_checkpoint(
    weights_dir: str | os.PathLike[str], override: str = ""
) -> Path:
    path = (
        Path(override).expanduser().resolve()
        if override
        else (Path(weights_dir).expanduser() / OFFICIAL_VAE_FILENAME).resolve()
    )
    if not path.is_file():
        raise FileNotFoundError(f"official VAE checkpoint does not exist: {path}")
    return path


def _is_raw_state_dict(value: Any) -> bool:
    return (
        isinstance(value, Mapping)
        and bool(value)
        and all(
            isinstance(key, str) and isinstance(tensor, torch.Tensor)
            for key, tensor in value.items()
        )
    )


def extract_supported_model_state(checkpoint: Any) -> Mapping[str, torch.Tensor]:
    """Accept only official raw state dicts or this project's trainer/FSDP envelope."""

    if _is_raw_state_dict(checkpoint):
        return checkpoint
    if isinstance(checkpoint, Mapping) and isinstance(
        checkpoint.get("trainer"), Mapping
    ):
        state = checkpoint["trainer"].get("gpt_fsdp")
        if _is_raw_state_dict(state):
            return state
        raise RuntimeError(
            "trainer checkpoint is missing a full tensor state at trainer.gpt_fsdp"
        )
    raise RuntimeError(
        "unsupported GRN checkpoint structure; expected a raw state dict or "
        "the current project envelope containing trainer.gpt_fsdp"
    )


def load_base_checkpoint_strict(model, checkpoint_path):
    checkpoint = torch.load(
        str(checkpoint_path), map_location="cpu", weights_only=False
    )
    model.load_state_dict(
        extract_supported_model_state(checkpoint), strict=True, assign=True
    )


def load_visual_tokenizer(
    args: Any, device: str | torch.device | None = None
) -> HBQ_Tokenizer:
    """Latest official HBQ construction and strict state loading."""

    target_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    vae_path = resolve_vae_checkpoint(
        getattr(args, "weights_dir", "weights"), getattr(args, "vae_path", "")
    )
    args.vae_path = str(vae_path)
    vae = HBQ_Tokenizer(
        args=args,
        latent_channels=args.detail_scale_dim,
        encoder_out_type="feature_tanh",
    )
    vae.eval().requires_grad_(False)
    checkpoint = torch.load(
        str(vae_path), map_location=target_device, weights_only=False
    )
    if not isinstance(checkpoint, Mapping) or not (
        "ema" in checkpoint or "vae" in checkpoint
    ):
        raise RuntimeError(f"unsupported official VAE checkpoint structure: {vae_path}")
    state = checkpoint["ema"] if "ema" in checkpoint else checkpoint["vae"]
    vae.load_state_dict(state, strict=True, assign=True)
    return vae.to(target_device).eval().requires_grad_(False)


def _official_model_kwargs(
    args: Any, vae: HBQ_Tokenizer, inference_mode: bool
) -> dict[str, Any]:
    return {
        "vae_local": vae,
        "text_channels": args.Ct5,
        "text_maxlen": args.tlen,
        "norm_eps": args.norm_eps,
        "checkpointing": args.enable_checkpointing,
        "use_flex_attn": args.use_flex_attn,
        "num_of_label_value": args.num_of_label_value,
        "rope2d_normalized_by_hw": getattr(args, "rope2d_normalized_by_hw", 0),
        "pn": args.pn,
        "train_h_div_w_list": None,
        "apply_spatial_patchify": args.apply_spatial_patchify,
        "video_frames": args.video_frames,
        "other_args": args,
        "inference_mode": inference_mode,
    }


def _checkpoint_topology(checkpoint_path):
    from .checkpoint import load_stage_lora_checkpoint

    return load_stage_lora_checkpoint(checkpoint_path).topology


def build_stage_lora_unwrapped(
    args: Any,
    vae: HBQ_Tokenizer,
    *,
    inference_mode: bool = False,
    load_base: bool = True,
) -> tuple[StageLoRAGRN, StageLoRATopology, TrainableParameterStats]:
    """Build the complete static topology before compile/FSDP/optimizer."""

    model_name = str(args.model)
    base_path = (
        resolve_base_checkpoint(
            model_name, args.weights_dir, getattr(args, "base_checkpoint", "")
        )
        if load_base
        else Path(getattr(args, "base_checkpoint", "") or ".")
    )
    with disable_torch_default_parameter_reset():
        model = create_stage_lora_backbone(
            model_name, **_official_model_kwargs(args, vae, inference_mode)
        )
    if load_base:
        load_base_checkpoint_strict(model, base_path)

    mask, active = parse_chunk_mask(
        args.stage_lora_source_chunk_mask, len(model.block_chunks)
    )
    model.install_source_conditioning(
        mask,
        text_gate_rank=int(args.stage_lora_source_text_gate_rank),
    )
    exact_targets = build_lora_targets(model, args.stage_lora_scope, active)
    inject_stage_lora(
        model,
        exact_targets,
        rank=int(args.stage_lora_rank),
        alpha=float(args.stage_lora_alpha),
        dropout=float(args.stage_lora_dropout),
        adapter_name=ADAPTER_NAME,
    )
    stats = configure_stage_lora_trainable_parameters(model)
    topology = StageLoRATopology(
        source_chunk_mask=mask,
        active_chunk_ids=active,
        source_text_gate_rank=int(model.source_text_gate.rank),
        text_channels=int(model.source_text_gate.text_channels),
        lora_scope=str(args.stage_lora_scope),
        lora_rank=int(args.stage_lora_rank),
        lora_alpha=float(args.stage_lora_alpha),
        lora_dropout=float(args.stage_lora_dropout),
        lora_targets=exact_targets,
        adapter_name=ADAPTER_NAME,
        base_model=model_name,
        base_checkpoint=str(base_path),
        official_grn_commit=OFFICIAL_GRN_COMMIT,
        peft_version=PEFT_VERSION,
    )
    model.stage_lora_topology = topology

    resume_checkpoint = str(getattr(args, "resume_checkpoint", "") or "")
    if resume_checkpoint:
        topology.assert_matches(_checkpoint_topology(resume_checkpoint))
    adapter_dir = str(getattr(args, "adapter_dir", "") or "")
    if adapter_dir:
        if resume_checkpoint:
            raise ValueError(
                "--adapter_dir and --resume_checkpoint are mutually exclusive"
            )
        load_compact_stage_lora(adapter_dir, model, topology)
    _print_topology(topology, stats, model)
    return model, topology, stats


def _print_topology(
    topology: StageLoRATopology, stats: TrainableParameterStats, model: StageLoRAGRN
) -> None:
    suffix_counts = {
        suffix: sum(name.endswith(suffix) for name in topology.lora_targets)
        for suffix in (
            "attn.q_proj",
            "attn.k_proj",
            "attn.v_proj",
            "attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        )
    }
    selected_chunks = (
        len(model.block_chunks)
        if topology.lora_scope == "all_blocks"
        else len(topology.active_chunk_ids)
    )
    selected_blocks = len(topology.lora_targets) // 7
    print(
        "[StageLoRA topology] "
        f"mask={','.join(str(int(value)) for value in topology.source_chunk_mask)}, "
        f"active={topology.active_chunk_ids}, "
        f"text_gate=raw_edit_mean_rms/r{topology.source_text_gate_rank}, "
        f"lora_scope={topology.lora_scope}, "
        f"chunks={selected_chunks}, blocks={selected_blocks}, targets={len(topology.lora_targets)}, "
        f"suffix_counts={suffix_counts}",
        flush=True,
    )
    print(
        "[StageLoRA parameters] "
        f"total={stats.total:,}, lora={stats.lora:,}, source={stats.source:,}, "
        f"trainable={stats.trainable:,} ({stats.trainable_tensors} tensors)",
        flush=True,
    )
