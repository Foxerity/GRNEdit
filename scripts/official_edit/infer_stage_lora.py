"""Stage-LoRA video editing with checkpoint-derived model configuration."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint_path",
        required=True,
        help="Full .pth checkpoint, numbered split-checkpoint directory, or compact adapter directory.",
    )
    parser.add_argument("--weights_dir", default=str(REPO_ROOT / "weights"))
    parser.add_argument(
        "--base_checkpoint",
        default="",
        help="Base GRN weights; needed only by compact adapters.",
    )
    parser.add_argument(
        "--runtime_path",
        default="",
        help="Runtime JSON for a compact export lacking runtime.json.",
    )
    parser.add_argument("--vae_path", default="")
    parser.add_argument("--t5_path", default="")
    parser.add_argument("--official_meta_root", required=True)
    parser.add_argument(
        "--official_meta_subdirs",
        default="",
        help="Comma-separated metadata subdirectories, optionally name@repeat.",
    )
    parser.add_argument(
        "--output_dir", default=str(REPO_ROOT / "outputs" / "stage_lora")
    )
    parser.add_argument(
        "--num_samples_per_dataset", type=int, default=4, help="0 selects all rows."
    )
    parser.add_argument("--sample_seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--GPUS", type=int, default=1)
    parser.add_argument("--shuffle", type=int, choices=(0, 1), default=1)
    parser.add_argument("--skip_existing", type=int, choices=(0, 1), default=1)
    parser.add_argument(
        "--guidance_mode", choices=("none", "standard", "cfg1"), default="none"
    )
    parser.add_argument("--guidance_scale", type=float, default=3.0)
    parser.add_argument("--negative_prompt", default="")
    parser.add_argument("--cfg_interval", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max_infer_steps", type=int, default=50)
    parser.add_argument("--snr_shift", type=float, default=1.0)
    parser.add_argument("--use_slow_attn", type=int, choices=(0, 1), default=0)
    cli = parser.parse_args(argv)
    if cli.GPUS < 1 or cli.max_infer_steps < 2 or cli.temperature <= 0:
        parser.error(
            "GPUS >= 1, max_infer_steps >= 2, and temperature > 0 are required."
        )
    if cli.num_samples_per_dataset < 0:
        parser.error("num_samples_per_dataset must be non-negative.")
    return cli


def _runtime_args(cli, checkpoint, device):
    from grn.official_edit_stage_lora.checkpoint import runtime_value

    topology, manifest = checkpoint.topology, checkpoint.manifest
    if topology.base_model != "GRN2b":
        raise ValueError("The public Stage-LoRA inference release supports GRN2b only.")
    read = lambda name: runtime_value(manifest, name)
    args = SimpleNamespace(
        model=topology.base_model,
        weights_dir=str(Path(cli.weights_dir).expanduser().resolve()),
        base_checkpoint=cli.base_checkpoint,
        adapter_dir=str(checkpoint.adapter_dir or ""),
        resume_checkpoint="",
        vae_path=cli.vae_path,
        t5_path=cli.t5_path or str(Path(cli.weights_dir) / "umt5-xxl"),
        stage_lora_source_chunk_mask=topology.source_chunk_mask,
        stage_lora_source_text_gate_rank=topology.source_text_gate_rank,
        stage_lora_scope=topology.lora_scope,
        stage_lora_rank=topology.lora_rank,
        stage_lora_alpha=topology.lora_alpha,
        stage_lora_dropout=topology.lora_dropout,
        stage_lora_use_reprompt_text=bool(read("use_reprompt_text")),
        video_caption_type=str(read("video_caption_type")),
        Ct5=int(read("text_channels")),
        tlen=int(read("t5_max_tokens")),
        t5_max_tokens=int(read("t5_max_tokens")),
        norm_eps=float(read("norm_eps")),
        enable_checkpointing=None,
        use_flex_attn=False,
        num_of_label_value=int(read("num_of_label_value")),
        pn=str(read("pn")),
        train_h_div_w_list=json.dumps(read("train_h_div_w_list")),
        apply_spatial_patchify=int(read("apply_spatial_patchify")),
        dynamic_scale_schedule=str(read("dynamic_scale_schedule")),
        video_frames=int(read("video_frames")),
        video_fps=int(read("video_fps")),
        fps=int(read("video_fps")),
        min_video_frames=int(read("min_video_frames")),
        temporal_compress_rate=int(read("temporal_compress_rate")),
        duration_resolution=float(read("duration_resolution")),
        detail_scale_dim=int(read("detail_scale_dim")),
        detail_num_lvl=int(read("detail_num_lvl")),
        hbq_round=int(read("hbq_round")),
        rope2d_normalized_by_hw=int(read("rope2d_normalized_by_hw")),
        use_ada_layer_norm=int(read("use_ada_layer_norm")),
        add_scale_token=int(read("add_scale_token")),
        vae_encoder_out_type="feature_tanh",
        refine_mode=str(read("refine_mode")),
        use_slice=int(read("use_slice")),
        alpha=float(read("corruption_alpha")),
        log_norm_mean=float(read("log_norm_mean")),
        log_norm_sigma=float(read("log_norm_sigma")),
        train_max_token_len=int(read("train_max_token_len")),
        pad_to_multiplier=128,
        rope_type="3d",
        semantic_scale_dim=64,
        semantic_num_lvl=2,
        use_slow_attn=bool(cli.use_slow_attn),
        bf16=True,
        max_infer_steps=cli.max_infer_steps,
        min_infer_steps=cli.max_infer_steps,
        complexity_aware_Tmin=min(10, cli.max_infer_steps),
        complexity_aware_Tmax=cli.max_infer_steps,
        cfg_type=f"cfg_interval_{cli.cfg_interval}",
        snr_shift=cli.snr_shift,
        device=str(device),
        other_device=device,
        meta="",
    )
    if args.refine_mode != "ar_discrete_GRN_bit" or args.add_scale_token != 1:
        raise ValueError(
            "Stage-LoRA requires binary refinement and one timestep token."
        )
    return args


def _select_indices(count, repeat, shuffle, rng):
    import numpy as np

    full_repeats = int(repeat)
    fractional_count = int(count * (repeat - full_repeats))
    parts = []
    if full_repeats:
        parts.append(np.tile(np.arange(count, dtype=np.int64), full_repeats))
    if fractional_count:
        fractional = np.arange(count, dtype=np.int64)
        rng.shuffle(fractional)
        parts.append(fractional[:fractional_count])
    if not parts:
        return np.empty(0, dtype=np.int64)
    selected = parts[0] if len(parts) == 1 else np.concatenate(parts)
    if shuffle:
        rng.shuffle(selected)
    return selected


def _load_samples(cli):
    import numpy as np

    from grn.dataset.official_t2iv.dataset_edit_pair import list_official_edit_jsonls

    root = Path(cli.official_meta_root).expanduser().resolve()
    specs = [
        item.strip() for item in cli.official_meta_subdirs.split(",") if item.strip()
    ] or ["."]
    rng, samples = np.random.default_rng(cli.sample_seed), []
    for spec in specs:
        name, repeat = spec.rsplit("@", 1) if "@" in spec else (spec, "1")
        repeat = float(repeat)
        if not math.isfinite(repeat) or repeat <= 0:
            raise ValueError(f"Invalid metadata repeat: {spec}")
        directory = root / name
        rows = []
        for path in sorted(list_official_edit_jsonls(str(directory))):
            with open(path, encoding="utf-8-sig") as handle:
                rows.extend(
                    (Path(path), line_no, json.loads(line))
                    for line_no, line in enumerate(handle, 1)
                    if line.strip()
                )
        if not rows:
            raise ValueError(f"No JSONL samples found in {directory}")
        selected = _select_indices(len(rows), repeat, cli.shuffle, rng)
        if cli.num_samples_per_dataset:
            selected = selected[: cli.num_samples_per_dataset]
        samples.extend((directory.name, *rows[index]) for index in selected)
    return samples


def _read_source(row, meta_file, args):
    """Use the same absolute frame-index and VAE alignment policy as training."""
    import numpy as np
    import torch
    from grn.official_edit_stage_lora.data_utils import (
        extract_single_text,
        resolve_media_path,
        shared_sample_indices,
    )
    from grn.official_edit_stage_lora.dataset_duration import (
        resolve_stage_lora_temporal_window,
    )
    from grn.official_edit_stage_lora.text_conditioning import clean_stage_lora_reprompt
    from PIL import Image

    from grn.dataset.official_t2iv.dataset_joint_vi import transform
    from grn.utils.video_decoder import EncodedVideoDecord

    source_path = resolve_media_path(
        row.get("source_path"), field_name="source_path", meta_file=str(meta_file)
    )
    meta = dict(row)
    begin, end, fps = (
        int(meta["begin_frame_id"]),
        int(meta["end_frame_id"]),
        float(meta["fps"]),
    )
    if begin < 0 or end <= begin or not math.isfinite(fps) or fps <= 0:
        raise ValueError("Source video metadata has an invalid frame interval.")
    duration = (end - begin) / fps
    mapped = round(duration / args.duration_resolution) * args.duration_resolution
    window = resolve_stage_lora_temporal_window(
        mapped,
        available_seconds=duration,
        sampling_fps=args.video_fps,
        temporal_compress_rate=args.temporal_compress_rate,
        minimum_seconds=(args.min_video_frames - 1) // args.video_fps,
        maximum_seconds=(args.video_frames - 1) // args.video_fps,
    )
    meta["sample_frames"] = window.sample_frames
    video = EncodedVideoDecord(
        source_path, Path(source_path).name, num_threads=0, fault_tol=0
    )
    try:
        indices = shared_sample_indices(meta, len(video._av_reader), args.video_fps)
        frames = video._av_reader.get_batch(indices).asnumpy()
    finally:
        video.close()
    height, width = frames.shape[1:3]
    template = float(
        args.h_div_w_templates[
            np.argmin(np.abs(args.h_div_w_templates - height / width))
        ]
    )
    target_h, target_w = args.dynamic_resolution_h_w[template][args.pn]["pixel"]
    tensor = torch.stack(
        [
            transform(
                Image.fromarray(frame).convert("RGB"), int(target_h), int(target_w)
            )
            for frame in frames
        ]
    ).contiguous()
    prompt = extract_single_text(meta[args.video_caption_type], args.video_caption_type)
    reprompt = (
        clean_stage_lora_reprompt(meta.get("reprompt"))
        if args.stage_lora_use_reprompt_text
        else ""
    )
    return tensor, template, prompt, reprompt, indices.tolist()


def _load_model(args, checkpoint, device):
    from grn.official_edit_stage_lora.inference_build import (
        build_stage_lora_unwrapped,
        load_visual_tokenizer,
    )

    vae = load_visual_tokenizer(args, device=device)
    args.vae = vae
    model, constructed, _ = build_stage_lora_unwrapped(
        args,
        vae,
        inference_mode=True,
        load_base=checkpoint.adapter_dir is not None,
    )
    checkpoint.topology.assert_matches(constructed)
    if checkpoint.state is not None:
        model.load_state_dict(checkpoint.state, strict=True, assign=True)
    model = model.to(device).eval().requires_grad_(False)
    for block in model.unregistered_blocks:
        block.bfloat16()
    return vae, model


def _infer_one(sample, args, cli, vae, model, text_encoder, negative, device, seed):
    import numpy as np
    import torch
    from grn.official_edit_stage_lora.inference_ops import (
        get_visual_rope_embeds,
        source_raw_feature_to_inference_tokens,
    )
    from grn.official_edit_stage_lora.text_conditioning import (
        encode_stage_lora_text_with_edit_context,
    )

    dataset, meta_file, line_no, row = sample
    source, template, prompt, reprompt, indices = _read_source(row, meta_file, args)
    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
        raw_features, _, _ = vae.encode_for_raw_features(
            source.permute(1, 0, 2, 3).unsqueeze(0).to(device),
            scale_schedule=None,
            slice=bool(args.use_slice),
        )
    raw_source = raw_features[0]
    shape = tuple(raw_source.shape[-3:])
    schedule = [
        tuple(
            args.dynamic_resolution_h_w[template][args.pn]["pt2scale_schedule"][
                shape[0]
            ][0]
        )
    ]
    if schedule[0] != shape:
        raise ValueError(
            f"Source latent and refinement geometry disagree: {shape} vs {schedule[0]}"
        )
    source_tokens = source_raw_feature_to_inference_tokens(
        raw_source, hbq_round=args.hbq_round
    )
    condition = encode_stage_lora_text_with_edit_context(
        text_encoder,
        [[prompt]],
        [[reprompt]] if args.stage_lora_use_reprompt_text else None,
        use_reprompt=args.stage_lora_use_reprompt_text,
        device=device,
        t5_max_tokens=args.t5_max_tokens,
    )
    if cli.guidance_mode == "none":
        negative = condition  # Unused: no second T5 encoding or negative model stream.
    elif cli.guidance_mode == "cfg1":
        negative = replace(negative, edit_context=condition.edit_context)
    mode = "standard" if cli.guidance_mode == "cfg1" else cli.guidance_mode
    args.mapped_h_div_w_template = template
    args.meta = f"{meta_file}:{line_no}"
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        _, _, generated = model.autoregressive_infer(
            vae=vae,
            scale_schedule=schedule,
            label_B_or_BLT=[condition.backbone[:4]],
            negative_label_B_or_BLT=negative.backbone[:4],
            source_x_BLC=source_tokens,
            edit_text_context=condition.edit_context,
            negative_edit_text_context=negative.edit_context,
            cfg_list=[1.0 if mode == "none" else cli.guidance_scale],
            tau_list=[cli.temperature],
            args=args,
            get_visual_rope_embeds=get_visual_rope_embeds,
            guidance_mode=mode,
        )
    return generated[0].detach().cpu().numpy(), {
        "dataset": dataset,
        "meta_file": str(meta_file),
        "meta_line_no": line_no,
        "prompt": prompt,
        "reprompt": reprompt,
        "sampled_frame_indices": indices,
        "seed": seed,
        "guidance_mode": cli.guidance_mode,
    }


def main():
    cli = _parse_args()
    if cli.GPUS > 1 and "LOCAL_RANK" not in os.environ:
        raise SystemExit(
            subprocess.call(
                [
                    sys.executable,
                    "-m",
                    "torch.distributed.run",
                    "--standalone",
                    f"--nproc_per_node={cli.GPUS}",
                    str(Path(__file__).resolve()),
                    *sys.argv[1:],
                ]
            )
        )
    import torch
    from grn.official_edit_stage_lora.checkpoint import (
        load_stage_lora_checkpoint,
        runtime_value,
    )
    from grn.official_edit_stage_lora.text_conditioning import (
        encode_stage_lora_negative_text,
    )

    from grn.schedules.dynamic_resolution import get_dynamic_resolution_meta
    from scripts.official_edit.inference_utils import _load_text_encoder, images2video

    rank, world = (
        int(os.environ.get("RANK", "0")),
        int(os.environ.get("WORLD_SIZE", "1")),
    )
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    samples = list(enumerate(_load_samples(cli)))[rank::world]
    if not samples:
        return
    if not torch.cuda.is_available():
        raise RuntimeError("Stage-LoRA inference requires CUDA and FlashAttention-4.")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    checkpoint = load_stage_lora_checkpoint(
        cli.checkpoint_path, runtime_path=cli.runtime_path or None
    )
    args = _runtime_args(cli, checkpoint, device)
    args.dynamic_resolution_h_w, args.h_div_w_templates = get_dynamic_resolution_meta(
        args.dynamic_scale_schedule, args.train_h_div_w_list, args.video_frames
    )
    torch.backends.cuda.matmul.allow_tf32 = bool(
        runtime_value(checkpoint.manifest, "tf32")
    )
    torch.backends.cudnn.allow_tf32 = bool(runtime_value(checkpoint.manifest, "tf32"))
    torch.set_float32_matmul_precision("high")
    print(
        f"[Stage-LoRA] model={checkpoint.topology.base_model}, branches={list(checkpoint.topology.active_chunk_ids)}, LoRA={checkpoint.topology.lora_scope}/r{checkpoint.topology.lora_rank}",
        flush=True,
    )
    vae, model = _load_model(args, checkpoint, device)
    del checkpoint
    text_encoder = _load_text_encoder(args, device)
    negative = None
    if cli.guidance_mode != "none":
        negative = encode_stage_lora_negative_text(
            text_encoder,
            [[cli.negative_prompt]],
            device=device,
            t5_max_tokens=args.t5_max_tokens,
        )
    output = Path(cli.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / f"rank_{rank:03d}_results.jsonl").open(
        "a", encoding="utf-8"
    ) as handle:
        for index, sample in samples:
            seed = cli.seed + index
            video_path = output / f"sample_{index:06d}_seed_{seed}.mp4"
            if cli.skip_existing and video_path.is_file():
                continue
            generated, metadata = _infer_one(
                sample, args, cli, vae, model, text_encoder, negative, device, seed
            )
            metadata["output_path"] = images2video(
                generated, fps=args.fps, save_filepath=str(video_path)
            )
            handle.write(json.dumps(metadata, ensure_ascii=False) + "\n")
            handle.flush()


if __name__ == "__main__":
    main()
