from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Sequence, Tuple

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["XFORMERS_FORCE_DISABLE_TRITON"] = "1"

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from timm.models import create_model

from scripts.official_edit import inference_utils as base_infer
from scripts.official_edit import resource_utils
from scripts.official_edit.inference_utils import images2video
from grn.official_edit_stage15.model import (  # noqa: F401
    GRN2bOfficialEditStage15,
    STAGE15_SOURCE_INJECTION_MASK,
    STAGE15_SOURCE_INJECTION_MASK_JSON,
)
from grn.official_edit_stage15.text_conditioning import (
    STAGE15_TEXT_PAIR_MARKER,
    STAGE15_TEXT_SCHEMA,
    STAGE15_TEXT_SEPARATOR_TOKENS,
    add_stage15_t2v_prefix,
    clean_stage15_edited_prompt,
    clean_stage15_reprompt,
)
from grn.official_t2iv_edit.build import load_visual_tokenizer
from grn.official_t2iv_edit.global_refine import get_visual_rope_embeds
from grn.official_t2iv_edit.model_config import default_block_chunks_for_model


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def _str2bool(value: Any) -> bool:
    return base_infer._str2bool(value)


def _maybe_launch_distributed(args: argparse.Namespace) -> None:
    total_processes = max(1, int(args.GPUS)) * max(1, int(args.workers))
    if total_processes <= 1 or "LOCAL_RANK" in os.environ:
        return
    cmd = [
        "torchrun",
        "--standalone",
        f"--nproc_per_node={total_processes}",
        str(Path(__file__).resolve()),
        *sys.argv[1:],
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO_ROOT}{os.pathsep}{env.get('PYTHONPATH', '')}"
    print("[stage15 launcher] " + " ".join(cmd), flush=True)
    raise SystemExit(subprocess.call(cmd, env=env))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GRNEdit Stage I inference.")
    parser.add_argument("--checkpoint_path", default=_env("CHECKPOINT_PATH", ""))
    parser.add_argument("--weights_dir", default=_env("WEIGHTS_DIR", str(REPO_ROOT / "weights")))
    parser.add_argument(
        "--output_dir",
        default=_env("OUTPUT_DIR", str(REPO_ROOT / "outputs" / "official_stage15_layerwise_source_conditioned_infer")),
    )
    parser.add_argument("--official_meta_root", default=_env("EDIT_OFFICIAL_META_ROOT", ""))
    parser.add_argument("--official_meta_subdirs", default=_env("EDIT_OFFICIAL_META_SUBDIRS", ""))
    parser.add_argument("--num_samples_per_dataset", type=int, default=int(_env("NUM_SAMPLES_PER_DATASET", "4")))
    parser.add_argument("--shuffle", type=_str2bool, default=_str2bool(_env("SHUFFLE", "1")))
    parser.add_argument("--sample_seed", type=int, default=int(_env("SAMPLE_SEED", "42")))
    parser.add_argument("--seed", type=int, default=int(_env("SEED", "1234")))
    parser.add_argument("--GPUS", type=int, default=int(_env("GPUS", "1")))
    parser.add_argument("--workers", type=int, default=int(_env("WORKERS", "2")))
    parser.add_argument("--model", default=_env("MODEL", "GRN2bOfficialEditStage15"))
    parser.add_argument(
        "--block_chunks",
        type=int,
        default=int(_env("BLOCK_CHUNKS", str(default_block_chunks_for_model(_env("MODEL", "GRN2bOfficialEditStage15"))))),
    )
    parser.add_argument("--pn", default=_env("PN", "0.41M"))
    parser.add_argument("--video_fps", type=int, default=int(_env("VIDEO_FPS", "20")))
    parser.add_argument("--video_frames", type=int, default=int(_env("VIDEO_FRAMES", "61")))
    parser.add_argument("--duration_resolution", type=float, default=float(_env("DURATION_RESOLUTION", "0.25")))
    parser.add_argument("--temperature", type=float, default=float(_env("TEMPERATURE", "1.0")))
    parser.add_argument("--max_infer_steps", type=int, default=int(_env("MAX_INFER_STEPS", "50")))
    parser.add_argument("--complexity_aware_Tmin", type=int, default=int(_env("COMPLEXITY_AWARE_TMIN", "10")))
    parser.add_argument(
        "--complexity_aware_Tmax",
        type=int,
        default=int(_env("COMPLEXITY_AWARE_TMAX", "0")),
        help="0 means follow --max_infer_steps.",
    )
    parser.add_argument("--snr_shift", type=float, default=float(_env("SNR_SHIFT", "1.0")))
    parser.add_argument("--use_slow_attn", type=_str2bool, default=_str2bool(_env("USE_SLOW_ATTN", "0")))
    parser.add_argument("--use_reprompt_text", type=_str2bool, default=_str2bool(_env("USE_REPROMPT", "0")))
    parser.add_argument("--t5_max_tokens", type=int, default=int(_env("T5_MAX_TOKENS", "512")))
    parser.add_argument(
        "--stage15_source_residual_modulation",
        choices=("text_pt",),
        default=_env("STAGE15_SOURCE_RESIDUAL_MODULATION", "text_pt"),
    )
    parser.add_argument(
        "--stage15_source_injection_mask",
        default=_env("STAGE15_SOURCE_INJECTION_MASK", STAGE15_SOURCE_INJECTION_MASK_JSON),
        help=f"GRNEdit endpoint injection mask: {STAGE15_SOURCE_INJECTION_MASK_JSON}.",
    )
    parser.add_argument("--use_ema", type=_str2bool, default=_str2bool(_env("USE_EMA", "1")))
    parser.add_argument("--vae_path", default="")
    parser.add_argument("--t5_path", default="")
    args = parser.parse_args()
    try:
        parsed_mask = json.loads(args.stage15_source_injection_mask)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"STAGE15_SOURCE_INJECTION_MASK must be {STAGE15_SOURCE_INJECTION_MASK_JSON}."
        ) from exc
    if (
        not isinstance(parsed_mask, list)
        or any(
            not (isinstance(item, bool) or (isinstance(item, int) and item in (0, 1)))
            for item in parsed_mask
        )
        or tuple(bool(item) for item in parsed_mask) != STAGE15_SOURCE_INJECTION_MASK
    ):
        raise ValueError(
            f"GRNEdit supports only endpoint source injection {STAGE15_SOURCE_INJECTION_MASK_JSON}."
        )
    return args


def _resolve_stage15_checkpoint_runtime(cli: argparse.Namespace) -> None:
    checkpoint = torch.load(cli.checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
    saved = checkpoint.get("args") if isinstance(checkpoint, dict) else None
    if not isinstance(saved, dict):
        raise RuntimeError("Stage I inference requires checkpoint/args runtime metadata.")
    required = (
        "model",
        "block_chunks",
        "stage15_source_injection_mask",
        "stage15_source_residual_modulation",
        "stage15_use_reprompt_text",
        "stage15_text_schema",
        "t5_max_tokens",
    )
    missing = [field for field in required if field not in saved]
    if missing:
        raise RuntimeError(f"Stage I checkpoint is missing runtime settings: {missing}.")
    if str(saved["model"]) != "GRN2bOfficialEditStage15":
        raise RuntimeError(f"Stage I release supports only GRN2bOfficialEditStage15, got {saved['model']!r}.")
    saved_mask = [int(item) for item in json.loads(str(saved["stage15_source_injection_mask"]))]
    requested_mask = [int(item) for item in json.loads(str(cli.stage15_source_injection_mask))]
    comparisons = {
        "model": (str(cli.model), str(saved["model"])),
        "block_chunks": (int(cli.block_chunks), int(saved["block_chunks"])),
        "source_injection_mask": (requested_mask, saved_mask),
        "source_residual_modulation": (
            str(cli.stage15_source_residual_modulation),
            str(saved["stage15_source_residual_modulation"]),
        ),
        "use_reprompt_text": (int(bool(cli.use_reprompt_text)), int(bool(saved["stage15_use_reprompt_text"]))),
        "t5_max_tokens": (int(cli.t5_max_tokens), int(saved["t5_max_tokens"])),
    }
    mismatches = [
        f"{field}: requested={requested!r}, checkpoint={persisted!r}"
        for field, (requested, persisted) in comparisons.items()
        if requested != persisted
    ]
    if mismatches:
        raise RuntimeError("Stage I inference configuration conflicts with the checkpoint:\n- " + "\n- ".join(mismatches))
    if str(saved["stage15_text_schema"]) != STAGE15_TEXT_SCHEMA:
        raise RuntimeError("Stage I checkpoint uses an incompatible text-conditioning schema.")
    resource_args = SimpleNamespace(vae_path=cli.vae_path, t5_path=cli.t5_path)
    for field, value in resource_utils.compute_external_resource_fingerprints(resource_args).items():
        setattr(resource_args, field, value)
    resource_utils.validate_external_resource_fingerprints(saved, resource_args, "Stage I checkpoint")


def _clean_state_key(key: str) -> str:
    key = str(key)
    prefixes = ("module.", "_orig_mod.", "_fsdp_wrapped_module.")
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix):]
                changed = True
    return key


def _infer_stage15_deep_count_from_state(clean_keys: set) -> int:
    indices = []
    prefix = "deep_source_word_embeds."
    suffix = ".weight"
    for key in clean_keys:
        if key.startswith(prefix) and key.endswith(suffix):
            raw_index = key[len(prefix):-len(suffix)]
            if raw_index.isdigit():
                indices.append(int(raw_index))
    if not indices:
        return 0
    indices = sorted(indices)
    expected = list(range(indices[-1] + 1))
    if indices != expected:
        raise RuntimeError(f"Stage1.5 deep source embed indices are not contiguous: {indices}")
    return len(indices)


def _validate_stage15_state(state: Dict[str, torch.Tensor]) -> None:
    clean_keys = {_clean_state_key(key) for key in state.keys()}
    required = {
        "source_word_embed.weight",
        "source_word_embed.bias",
    }
    deep_count = _infer_stage15_deep_count_from_state(clean_keys)
    if deep_count <= 0:
        raise RuntimeError("Stage1.5 inference requires deep_source_word_embeds.* keys; got none.")
    for index in range(deep_count):
        required.add(f"deep_source_word_embeds.{index}.weight")
        required.add(f"deep_source_word_embeds.{index}.bias")
    missing = sorted(required - clean_keys)
    if missing:
        raise RuntimeError(
            "Stage1.5 inference requires a full Stage1.5 source-conditioned checkpoint. "
            f"Missing required keys: {missing}."
        )


def _normalize_state_keys(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    normalized: Dict[str, torch.Tensor] = {}
    for key, value in state.items():
        clean_key = _clean_state_key(key)
        if clean_key in normalized and normalized[clean_key] is not value:
            raise RuntimeError(f"duplicate checkpoint key after prefix normalization: {clean_key}")
        normalized[clean_key] = value
    return normalized


def _load_stage15_checkpoint_state(checkpoint_path: str, *, use_ema: bool) -> Dict[str, torch.Tensor]:
    print(
        f"[load Stage1.5 state] checkpoint={checkpoint_path}, use_ema={bool(use_ema)}",
        flush=True,
    )
    state = _normalize_state_keys(base_infer._extract_model_state(checkpoint_path, use_ema=bool(use_ema)))
    print(f"[load Stage1.5 state] keys={len(state)}", flush=True)
    _validate_stage15_state(state)
    return state


def _load_stage15_state_into_model(
    model,
    state: Dict[str, torch.Tensor],
    *,
    checkpoint_path: str,
    expected_block_chunks: int,
) -> None:
    block_chunks = base_infer._infer_block_chunks_from_state(state)
    if block_chunks is not None and int(block_chunks) != int(expected_block_chunks):
        raise RuntimeError(
            "Stage1.5 checkpoint block chunk count does not match the resident inference model: "
            f"checkpoint={checkpoint_path}, checkpoint_chunks={block_chunks}, "
            f"model_chunks={expected_block_chunks}. Multi-checkpoint inference requires one architecture."
        )
    print(f"[load Stage1.5 checkpoint] loading state_dict from {checkpoint_path}", flush=True)
    load_msg = model.load_state_dict(state, strict=True)
    for attr in (
        "source_word_embed",
        "deep_source_word_embeds",
        "source_residual_modulations",
    ):
        if not hasattr(model, attr):
            raise RuntimeError(f"loaded model is missing Stage1.5 source residual attribute: {attr}")
    print(f"[load Stage1.5 checkpoint] {checkpoint_path}: {load_msg}", flush=True)


def _reload_stage15_model_checkpoint(model, checkpoint_path: str, *, use_ema: bool) -> None:
    state = _load_stage15_checkpoint_state(checkpoint_path, use_ema=use_ema)
    _load_stage15_state_into_model(
        model,
        state,
        checkpoint_path=checkpoint_path,
        expected_block_chunks=len(model.block_chunks),
    )


def _load_stage15_model_and_vae(args: SimpleNamespace, cli: argparse.Namespace, device: torch.device):
    vae = load_visual_tokenizer(args, device=device)
    state = _load_stage15_checkpoint_state(args.checkpoint_path, use_ema=bool(cli.use_ema))
    block_chunks = base_infer._infer_block_chunks_from_state(state)
    if block_chunks is None:
        block_chunks = int(cli.block_chunks)
    if str(args.model) != "GRN2bOfficialEditStage15":
        raise ValueError(f"Stage I inference requires GRN2bOfficialEditStage15, got {args.model!r}.")

    gpt_kw = dict(
        pretrained=False,
        global_pool="",
        text_channels=args.Ct5,
        text_maxlen=args.tlen,
        norm_eps=args.norm_eps,
        top_p=args.tp,
        top_k=args.tk,
        tau=args.tau,
        checkpointing=args.enable_checkpointing,
        pad_to_multiplier=args.pad_to_multiplier,
        use_flex_attn=args.use_flex_attn,
        num_of_label_value=args.num_of_label_value,
        pn=args.pn,
        train_h_div_w_list=None,
        apply_spatial_patchify=args.apply_spatial_patchify,
        dynamic_scale_schedule=args.dynamic_scale_schedule,
        video_frames=args.video_frames,
        other_args=args,
        vae_local=vae,
        inference_mode=True,
        block_chunks=block_chunks,
    )
    print(
        f"[load Stage1.5 GRN] model={args.model}, pn={args.pn}, video_frames={args.video_frames}, "
        f"block_chunks={block_chunks}",
        flush=True,
    )
    model = create_model(args.model, **gpt_kw).to(device=device)
    model.eval().requires_grad_(False)
    _load_stage15_state_into_model(
        model,
        state,
        checkpoint_path=args.checkpoint_path,
        expected_block_chunks=int(block_chunks),
    )
    return vae, model


def _build_stage15_infer_args(cli: argparse.Namespace, device: torch.device) -> SimpleNamespace:
    use_reprompt_text = bool(cli.use_reprompt_text)
    text_total_tokens = int(cli.t5_max_tokens)
    max_infer_steps = int(cli.max_infer_steps)
    requested_tmax = int(cli.complexity_aware_Tmax)
    if max_infer_steps <= 1:
        raise ValueError(f"--max_infer_steps must be > 1, got {max_infer_steps}.")
    if requested_tmax <= 0:
        complexity_aware_Tmax = max_infer_steps
    elif requested_tmax == max_infer_steps:
        complexity_aware_Tmax = requested_tmax
    else:
        raise ValueError(
            "--complexity_aware_Tmax must equal --max_infer_steps for official Stage1.5 edit inference. "
            f"got complexity_aware_Tmax={requested_tmax}, max_infer_steps={max_infer_steps}."
        )
    args = SimpleNamespace(
        model=cli.model,
        vae_path=cli.vae_path,
        t5_path=cli.t5_path,
        checkpoint_path=cli.checkpoint_path,
        Ct5=4096,
        tlen=text_total_tokens,
        t5_max_tokens=int(cli.t5_max_tokens),
        norm_eps=1e-6,
        tp=0.0,
        tk=0.0,
        tau=float(cli.temperature),
        enable_checkpointing="full-block",
        pad_to_multiplier=128,
        use_flex_attn=True,
        num_of_label_value=2,
        pn=cli.pn,
        train_h_div_w_list="[]",
        apply_spatial_patchify=0,
        dynamic_scale_schedule="GRN_vae_stride16",
        video_frames=int(cli.video_frames),
        video_fps=int(cli.video_fps),
        fps=int(cli.video_fps),
        temporal_compress_rate=4,
        duration_resolution=float(cli.duration_resolution),
        min_video_frames=-1,
        drop_long_video=0,
        detail_scale_dim=64,
        semantic_scale_dim=64,
        detail_num_lvl=2,
        semantic_num_lvl=2,
        hbq_round=4,
        rope_type="3d",
        use_ada_layer_norm=0,
        add_scale_token=1,
        add_class_token=0,
        vae_encoder_out_type="feature_tanh",
        refine_mode="ar_discrete_GRN_bit",
        alpha=1001,
        simple_text_proj=1,
        use_fsq_cls_head=1,
        use_slice=1,
        use_slow_attn=bool(cli.use_slow_attn),
        max_infer_steps=max_infer_steps,
        min_infer_steps=max_infer_steps,
        complexity_aware_Tmin=int(cli.complexity_aware_Tmin),
        complexity_aware_Tmax=complexity_aware_Tmax,
        complexity_aware_k=0,
        complexity_aware_b=50,
        complexity_aware_wp=5,
        snr_shift=float(cli.snr_shift),
        gt_leak=-1,
        meta="",
        seed=int(cli.seed),
        other_device=device,
        device=str(device),
        text_channels=4096,
        checkpoint_type="torch",
        bf16=0,
        use_reprompt_text=0,
        stage15_use_reprompt_text=int(use_reprompt_text),
        stage15_source_residual_modulation=str(cli.stage15_source_residual_modulation).strip().lower(),
        stage15_source_injection_mask=str(cli.stage15_source_injection_mask),
    )
    if str(args.model) != "GRN2bOfficialEditStage15":
        raise ValueError(f"Stage I inference requires GRN2bOfficialEditStage15, got {args.model!r}.")
    return args


def _make_t5_text_cond_tuple(*, t5_compact: torch.Tensor, t5_lens: Sequence[int], caption_nums: Sequence[int]):
    cu_seqlens = [0]
    for length in t5_lens:
        cu_seqlens.append(cu_seqlens[-1] + int(length))
    return (
        t5_compact,
        list(t5_lens),
        torch.tensor(cu_seqlens, dtype=torch.int32),
        max(t5_lens) if t5_lens else 0,
        list(caption_nums),
    )


def _make_stage15_text_pair_cond_tuple(
    *,
    edited_compact: torch.Tensor,
    edited_lens: Sequence[int],
    reprompt_compact: torch.Tensor,
    reprompt_lens: Sequence[int],
    main_lens: Sequence[int],
    caption_nums: Sequence[int],
):
    if not (len(edited_lens) == len(reprompt_lens) == len(main_lens)):
        raise ValueError(
            "Stage1.5 text pair lens mismatch: "
            f"edited={len(edited_lens)}, reprompt={len(reprompt_lens)}, main={len(main_lens)}"
        )
    cu_seqlens = [0]
    for length in main_lens:
        cu_seqlens.append(cu_seqlens[-1] + int(length))
    return (
        edited_compact,
        list(main_lens),
        torch.tensor(cu_seqlens, dtype=torch.int32),
        max(main_lens) if main_lens else 0,
        list(caption_nums),
        list(edited_lens),
        reprompt_compact,
        list(reprompt_lens),
        STAGE15_TEXT_PAIR_MARKER,
    )


def _encode_t5_with_limit(text_encoder, texts: Sequence[str], args: SimpleNamespace, device: torch.device, max_tokens: int):
    original_seq_len = getattr(text_encoder.tokenizer, "seq_len", None)
    max_tokens = int(max_tokens)
    if max_tokens <= 0:
        raise ValueError(f"T5 max token length must be positive, got {max_tokens}.")
    text_encoder.tokenizer.seq_len = max_tokens
    try:
        features, lens = base_infer._encode_t5(text_encoder, list(texts), args, device)
    finally:
        text_encoder.tokenizer.seq_len = original_seq_len
    split_features = list(torch.split(features, list(lens), dim=0))
    if len(split_features) != len(texts):
        raise ValueError(f"T5 feature/text count mismatch: {len(split_features)} != {len(texts)}")
    return split_features


def _resolve_reprompt_for_meta(
    *,
    meta: Dict[str, Any],
    prompt: str,
    cli: argparse.Namespace,
) -> Tuple[str, Dict[str, Any]]:
    existing = clean_stage15_reprompt(meta.get("reprompt", ""))
    if existing:
        return existing, {"reprompt_source": "jsonl"}
    context_key = f"{meta.get('meta_file', '<unknown>')}:{meta.get('meta_line_no', meta.get('row_index', '<unknown>'))}"
    raise ValueError(
        "Reprompt conditioning is enabled, but the JSONL record has no precomputed 'reprompt': "
        f"{context_key}."
    )


def _build_text_condition(
    *,
    prompt: str,
    reprompt: str,
    text_encoder,
    args: SimpleNamespace,
    device: torch.device,
    use_reprompt: bool = True,
):
    if not getattr(args, "stage15_use_reprompt_text", 0) or not use_reprompt:
        text = add_stage15_t2v_prefix(prompt)
        feature = _encode_t5_with_limit(text_encoder, [text], args, device, int(args.t5_max_tokens))[0]
        return _make_t5_text_cond_tuple(t5_compact=feature, t5_lens=[feature.shape[0]], caption_nums=[1])

    edited_prompt = add_stage15_t2v_prefix(clean_stage15_edited_prompt(prompt))
    reprompt = clean_stage15_reprompt(reprompt)
    edited_feature, reprompt_feature = _encode_t5_with_limit(
        text_encoder,
        [edited_prompt, reprompt],
        args,
        device,
        int(args.t5_max_tokens),
    )
    remaining = int(args.t5_max_tokens) - int(edited_feature.shape[0])
    if int(reprompt_feature.shape[0]) <= 0 or remaining <= STAGE15_TEXT_SEPARATOR_TOKENS:
        reprompt_feature = reprompt_feature[:0]
    else:
        reprompt_feature = reprompt_feature[:remaining - STAGE15_TEXT_SEPARATOR_TOKENS]
    main_len = (
        int(edited_feature.shape[0])
        + (STAGE15_TEXT_SEPARATOR_TOKENS if int(reprompt_feature.shape[0]) > 0 else 0)
        + int(reprompt_feature.shape[0])
    )
    return _make_stage15_text_pair_cond_tuple(
        edited_compact=edited_feature,
        edited_lens=[edited_feature.shape[0]],
        reprompt_compact=reprompt_feature,
        reprompt_lens=[reprompt_feature.shape[0]],
        main_lens=[main_len],
        caption_nums=[1],
    )


def _infer_one(
    *,
    sample: Dict[str, Any],
    args: SimpleNamespace,
    cli: argparse.Namespace,
    vae,
    model,
    text_encoder,
    device: torch.device,
    seed: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    meta = dict(sample["meta"])
    meta["pn"] = args.pn
    meta["first_frame_condition"] = False
    meta["sample_frames"] = base_infer._mapped_sample_frames(meta, args)
    prompt = base_infer._official_caption(meta)
    reprompt_info: Dict[str, Any] = {"reprompt_source": "disabled"}
    if getattr(args, "stage15_use_reprompt_text", 0):
        reprompt, reprompt_info = _resolve_reprompt_for_meta(meta=meta, prompt=prompt, cli=cli)
    else:
        reprompt = ""
    dynamic_resolution_h_w, h_div_w_templates = base_infer.get_dynamic_resolution_meta(
        args.dynamic_scale_schedule,
        args.train_h_div_w_list,
        args.video_frames,
    )

    source_t3hw, _, h_div_w_template, _ = base_infer._read_video_like_training(
        path=str(meta["source_path"]),
        meta=meta,
        args=args,
        dynamic_resolution_h_w=dynamic_resolution_h_w,
        h_div_w_templates=h_div_w_templates,
    )
    source_labels = base_infer._encode_source_bits(vae, source_t3hw, args, device)

    # Training builds the GRN schedule from the actual VAE latent shape:
    #   T, H, W = raw_features[-1].shape[-3:]
    # Do the same in inference. Deriving T from sample_frames can drift from the
    # tokenizer's real temporal output and causes source residual shape mismatch.
    source_latent_shape = tuple(int(item) for item in source_labels.shape[-3:])
    latent_t = source_latent_shape[0]
    scale_schedule = dynamic_resolution_h_w[h_div_w_template][args.pn]["pt2scale_schedule"][latent_t]
    scale_schedule = [tuple(scale_schedule[0])]
    if tuple(scale_schedule[0]) != source_latent_shape:
        raise RuntimeError(
            "Stage1.5 source latent shape does not match official scale schedule: "
            f"source_latent_shape={source_latent_shape}, scale_schedule[0]={tuple(scale_schedule[0])}, "
            f"sample_frames={meta['sample_frames']}, pn={args.pn}, h_div_w_template={h_div_w_template}."
        )
    args.mapped_h_div_w_template = h_div_w_template
    args.first_full_spatial_size_scale_index = 0
    args.tower_split_index = 1
    args.meta = f"{sample['dataset']}:{sample['index']}"

    cond = _build_text_condition(
        prompt=prompt,
        reprompt=reprompt,
        text_encoder=text_encoder,
        args=args,
        device=device,
    )
    if args.refine_mode != "ar_discrete_GRN_bit":
        raise ValueError(f"Stage1.5 source-conditioned inference requires ar_discrete_GRN_bit, got {args.refine_mode!r}.")
    with (
        torch.no_grad(),
        torch.cuda.amp.autocast(enabled=device.type == "cuda", dtype=torch.bfloat16, cache_enabled=True),
    ):
        generated = model.autoregressive_infer(
            vae=vae,
            scale_schedule=scale_schedule,
            label_B_or_BLT=[cond],
            tau_list=[float(cli.temperature)],
            args=args,
            get_visual_rope_embeds=get_visual_rope_embeds,
            stage15_source_labels=source_labels,
        )
    generated_bgr = generated[0].detach().cpu().numpy()
    return generated_bgr, {
        "dataset": sample["dataset"],
        "index": sample["index"],
        "meta_line_no": meta.get("meta_line_no", None),
        "prompt": prompt,
        "reprompt": reprompt,
        **reprompt_info,
        "seed": int(seed),
        "sample_frames": int(meta["sample_frames"]),
        "temperature": float(cli.temperature),
        "max_infer_steps": int(args.max_infer_steps),
        "stage15_layerwise_source_residual": True,
    }


def main() -> None:
    cli = _parse_args()
    _maybe_launch_distributed(cli)
    if not cli.checkpoint_path:
        raise ValueError("Set CHECKPOINT_PATH or --checkpoint_path for Stage1.5 edit inference.")

    weights_dir = osp.abspath(cli.weights_dir)
    cli.vae_path = cli.vae_path or osp.join(weights_dir, "HBQ_tokenizer_64dim_M4.ckpt")
    cli.t5_path = cli.t5_path or osp.join(weights_dir, "umt5-xxl")
    base_infer._require_file(cli.vae_path, "VAE checkpoint")
    base_infer._require_file(cli.checkpoint_path, "Stage1.5 GRN checkpoint")
    _resolve_stage15_checkpoint_runtime(cli)

    rank, world_size, _, gpu_index = base_infer._rank_info(cli.GPUS, cli.workers)
    if not torch.cuda.is_available():
        raise RuntimeError("official GRN Stage1.5 edit inference requires CUDA.")
    torch.cuda.set_device(gpu_index)
    device = torch.device(f"cuda:{gpu_index}")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    official_samples = base_infer._load_official_dataset_samples(
        root=cli.official_meta_root,
        subdirs=cli.official_meta_subdirs,
        num_samples_per_dataset=cli.num_samples_per_dataset,
        shuffle=cli.shuffle,
        seed=cli.sample_seed,
    )
    samples = official_samples
    if not samples:
        raise ValueError("No samples selected from EDIT_OFFICIAL_META_ROOT/SUBDIRS.")
    rank_samples = samples[rank::world_size]
    print(f"[rank {rank}/{world_size}] selected={len(samples)}, rank_samples={len(rank_samples)}, gpu={gpu_index}", flush=True)

    output_dir = Path(cli.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not rank_samples:
        print(f"[rank {rank}/{world_size}] no samples assigned; skip model loading.", flush=True)
        return

    args = _build_stage15_infer_args(cli, device)
    text_encoder = base_infer._load_text_encoder(args, device)
    vae, model = _load_stage15_model_and_vae(args, cli, device)
    result_jsonl = output_dir / f"rank_{rank:03d}_results.jsonl"
    with result_jsonl.open("w", encoding="utf-8") as f:
        for sample in rank_samples:
            uid = str(sample.get("uid") or f"{int(sample['logical_index']):06d}_{int(sample['index']):06d}")
            seed = int(cli.seed) + int(sample["logical_index"]) + rank * 100000
            dataset_dir = output_dir / base_infer._safe_name(sample["dataset"])
            dataset_dir.mkdir(parents=True, exist_ok=True)
            stem = f"sample_{uid}_row_{int(sample['index']):06d}_seed_{seed}"
            print(f"[stage15 infer] dataset={sample['dataset']} index={sample['index']} seed={seed}", flush=True)
            start = time.time()
            generated, info = _infer_one(
                sample=sample,
                args=args,
                cli=cli,
                vae=vae,
                model=model,
                text_encoder=text_encoder,
                device=device,
                seed=seed,
            )
            video_path = dataset_dir / f"{stem}.mp4"
            images2video(generated, fps=args.fps, save_filepath=str(video_path))
            info["output_path"] = str(video_path)
            info["uid"] = uid
            info["elapsed_sec"] = round(time.time() - start, 3)
            f.write(json.dumps(info, ensure_ascii=False) + "\n")
            f.flush()


if __name__ == "__main__":
    main()
