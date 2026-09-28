"""Official GRN bit-refinement backbone with Stage-LoRA chunk hooks.

Derived from ByteDance/GRN fe71a70e82b9f25303590fcb769212a81ab11b9d.
Attention, RoPE, transformer chunks and classification heads share the public
GRN implementation. Only the source hook and supported edit sampling path live here.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from grn.official_t2iv_edit.basic import SelfAttnBlock
from grn.official_t2iv_edit.model import (
    MultipleLayers,
    FsqHead,
    TimestepEmbedder,
    build_attn_mask,
)
from grn.official_t2iv_edit.rope import precompute_rope3d_freqs_grid
from grn.schedules.dynamic_resolution import get_dynamic_resolution_meta
from grn.utils_t2iv.hbq_util_t2iv import (
    multiclass_labels2onehot_input,
    bit_label2raw_feature,
)
from grn.utils_t2iv.sequence_parallel import SequenceParallelManager as sp_manager
from grn.utils_t2iv.sequence_parallel import (
    sp_gather_sequence_by_dim,
    sp_split_sequence_by_dim,
)


class GRN(nn.Module):
    def __init__(
        self,
        vae_local: Any,
        arch: str = "var",
        qwen_qkvo_bias: bool = False,
        text_channels: int = 0,
        text_maxlen: int = 0,
        embed_dim: int = 1024,
        depth: int = 16,
        num_key_value_heads: int = -1,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        drop_path_rate: float = 0.0,
        norm_eps: float = 1e-6,
        block_chunks: int = 1,
        checkpointing: Optional[str] = None,
        use_flex_attn: bool = False,
        num_of_label_value: int = 2,
        rope2d_normalized_by_hw: int = 0,
        pn: Optional[str] = None,
        video_frames: int = 1,
        always_training_scales: int = 20,
        apply_spatial_patchify: int = 0,
        inference_mode: bool = False,
        other_args: Optional[Any] = None,
        **kwargs: Any,
    ):
        super().__init__()
        # 1. Model Configuration
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.arch = arch
        self.mlp_ratio = mlp_ratio
        self.norm_eps = norm_eps
        self.drop_path_rate = drop_path_rate
        self.use_flex_attn = use_flex_attn
        self.checkpointing = checkpointing
        self.inference_mode = inference_mode
        self.other_args = other_args

        # 2. Embedding & Scale Configuration
        self.vae_embed_dim = vae_local.codebook_dim
        self.apply_spatial_patchify = apply_spatial_patchify
        self.text_channels = text_channels
        self.text_maxlen = text_maxlen
        self.is_text_to_image = text_channels != 0

        classifier_head_dim = other_args.detail_scale_dim
        classifier_head_lvl = other_args.detail_num_lvl
        hbq_round = other_args.hbq_round

        if (
            other_args.refine_mode != "ar_discrete_GRN_bit"
            or self.apply_spatial_patchify
        ):
            raise ValueError(
                "Stage-LoRA supports unpatchified binary GRN refinement only"
            )
        if other_args.use_ada_layer_norm or other_args.add_scale_token != 1:
            raise ValueError("Stage-LoRA requires RMSNorm blocks and one PT token")
        self.visual_embedding_in_dim = hbq_round * vae_local.codebook_dim * 2
        classifier_head_dim = hbq_round * vae_local.codebook_dim

        # 3. Dynamic Resolution & Video Specifics
        self.video_frames = video_frames
        self.always_training_scales = always_training_scales
        self.num_of_label_value = num_of_label_value
        self.rope2d_normalized_by_hw = rope2d_normalized_by_hw

        self.dynamic_resolution_h_w, self.h_div_w_templates = (
            get_dynamic_resolution_meta(
                other_args.dynamic_scale_schedule,
                other_args.train_h_div_w_list,
                other_args.video_frames,
            )
        )
        self.train_h_div_w_list = self.h_div_w_templates
        print(f"train_h_div_w_list: {self.train_h_div_w_list}")

        # 4. Utilities

        # 5. Model Components (Projections, Embeddings)
        self.norm0_cond = nn.Identity()
        self.text_proj = nn.Linear(self.text_channels, self.embed_dim)

        # RoPE grid initialization
        with torch.amp.autocast("cuda", dtype=torch.float32):
            self.rope2d_freqs_grid = precompute_rope3d_freqs_grid(
                dim=self.embed_dim // self.num_heads,
                rope2d_normalized_by_hw=self.rope2d_normalized_by_hw,
                activated_h_div_w_templates=self.train_h_div_w_list,
                max_scales=1010,  # never used
                max_frames=int(
                    self.video_frames / other_args.temporal_compress_rate + 1
                ),
                max_height=1800 // 8,
                max_width=1800 // 8,
                text_maxlen=self.text_maxlen,
                args=other_args,
            )

        self.word_embed = nn.Linear(self.visual_embedding_in_dim, self.embed_dim)
        self.head = FsqHead(
            hidden_dim=self.embed_dim,
            fsq_dim=classifier_head_dim,
            fsq_lvl=classifier_head_lvl,
            use_ada_layer_norm=other_args.use_ada_layer_norm,
        )

        if other_args.add_scale_token > 0:
            self.pt_embedder = TimestepEmbedder(self.embed_dim)

        # 6. Transformer Blocks
        self.unregistered_blocks = []
        for block_idx in range(depth):
            block = SelfAttnBlock(
                embed_dim=self.embed_dim,
                num_heads=num_heads,
                num_key_value_heads=num_key_value_heads,
                mlp_ratio=mlp_ratio,
                use_flex_attn=use_flex_attn,
                qwen_qkvo_bias=qwen_qkvo_bias,
                use_ada_layer_norm=other_args.use_ada_layer_norm,
            )
            self.unregistered_blocks.append(block)

        self.num_block_chunks = block_chunks or 1
        self.num_blocks_in_a_chunk = depth // self.num_block_chunks
        assert self.num_blocks_in_a_chunk * self.num_block_chunks == depth, (
            "Depth must be divisible by block_chunks"
        )

        self.block_chunks = nn.ModuleList(
            [
                MultipleLayers(
                    self.unregistered_blocks,
                    self.num_blocks_in_a_chunk,
                    i * self.num_blocks_in_a_chunk,
                )
                for i in range(self.num_block_chunks)
            ]
        )

        print(
            f"    [Model Config] embed_dim={embed_dim}, num_heads={num_heads}, depth={depth}, "
            f"mlp_ratio={mlp_ratio}, num_blocks_in_a_chunk={self.num_blocks_in_a_chunk}"
        )
        print(f"    drop_path_rate={drop_path_rate:g}", end="\n\n", flush=True)

    def before_block_chunk(self, chunk_id: int, hidden: torch.Tensor) -> torch.Tensor:
        """Return the hidden state passed to one transformer chunk.

        The vendored backbone keeps this identity hook as its only semantic extension.
        Stage-LoRA overrides it without changing packed lengths, RoPE, or attention metadata.
        """
        del chunk_id
        return hidden

    def get_loss_acc(
        self,
        hidden_states: torch.Tensor,
        hidden_states_mask: Optional[torch.Tensor],
        e: Optional[torch.Tensor],
        sequence_packing_scales: List[List[Tuple[int, int, int]]],
        gt: List[torch.Tensor],
        other_info_by_scale: List[Dict[str, Any]],
        return_last_hidden_states: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculate loss and accuracy for the predicted logits.

        Args:
            hidden_states: shaped (B, L, C)
            hidden_states_mask: Optional mask for hidden states
            e: scale or time embeddings
            sequence_packing_scales: List of scales for sequence packing
            gt: Ground truth labels
            other_info_by_scale: Meta information for each scale
            return_last_hidden_states: Whether to return the last hidden states

        Returns:
            Tuple of (logits_norm, loss_list, acc_list)
        """
        logits_norm = []
        logits_full = self.head(hidden_states, e)
        global_token_ptr, global_scale_ptr = 0, 0
        loss_list, acc_list = [], []

        for pack_scales in sequence_packing_scales:
            for pt, ph, pw in pack_scales:
                mul_pt_ph_pw = pt * ph * pw
                cur_bits = other_info_by_scale[global_scale_ptr]["cur_bits"]
                cur_lvl = other_info_by_scale[global_scale_ptr]["cur_lvl"]
                predict_tokens = other_info_by_scale[global_scale_ptr]["predict_tokens"]
                all_tokens = other_info_by_scale[global_scale_ptr]["all_tokens"]
                logits = logits_full[
                    :, global_token_ptr : global_token_ptr + predict_tokens
                ]
                logits = logits.reshape(
                    hidden_states.shape[0], mul_pt_ph_pw, cur_bits, cur_lvl
                )
                logits = logits.permute(
                    0, 3, 1, 2
                )  # [1, num_of_label_value, mul_pt_ph_pw, d]

                logits_norm.append(logits.abs().mean())

                # gt[global_scale_ptr]: [1, mul_pt_ph_pw, d]
                loss_this_scale = F.cross_entropy(
                    logits, gt[global_scale_ptr], reduction="none"
                )[0]  # [mul_pt_ph_pw, d]
                acc_this_scale = (logits.argmax(1) == gt[global_scale_ptr]).float()[
                    0
                ]  # [mul_pt_ph_pw, d]

                loss_list.append(loss_this_scale.mean(-1))
                acc_list.append(acc_this_scale.mean(-1))

                global_scale_ptr += 1
                global_token_ptr += all_tokens

        loss_tensor = (
            torch.cat(loss_list)
            if loss_list
            else torch.tensor([], device=hidden_states.device)
        )
        acc_tensor = (
            torch.cat(acc_list)
            if acc_list
            else torch.tensor([], device=hidden_states.device)
        )
        logits_norm_tensor = (
            torch.stack(logits_norm).mean()
            if logits_norm
            else torch.tensor(0.0, device=hidden_states.device)
        )

        return logits_norm_tensor, loss_tensor, acc_tensor

    def get_logits_during_infer(
        self, hidden_states: torch.Tensor, e: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Get logits during inference."""
        return self.head(hidden_states.float(), e)

    def forward(
        self,
        label_B_or_BLT: Union[
            torch.LongTensor, Tuple[torch.FloatTensor, torch.IntTensor, int]
        ],
        x_BLC: torch.Tensor,
        visual_rope_cache: Optional[List[torch.Tensor]] = None,
        sequece_packing_scales: Optional[List[List[Tuple[int, int, int]]]] = None,
        super_scale_lengths: Optional[List[int]] = None,
        other_info_by_scale: Optional[List[Dict[str, Any]]] = None,
        gt_BL: Optional[List[torch.Tensor]] = None,
        x_BLC_mask: Optional[torch.Tensor] = None,
        scale_or_time_ids: Optional[torch.Tensor] = None,
        return_last_hidden_states: bool = False,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Forward pass for the GRN model.

        Args:
            label_B_or_BLT: Text conditions or labels
            x_BLC: Input sequence hidden states
            visual_rope_cache: Cache for visual RoPE embeddings
            sequece_packing_scales: Scales for sequence packing
            super_scale_lengths: Lengths of super scales
            other_info_by_scale: Meta info for scales
            gt_BL: Ground truth
            x_BLC_mask: Mask for input sequence
            scale_or_time_ids: IDs for scale or time embeddings
            return_last_hidden_states: Whether to return last hidden states

        Returns:
            Tuple of (logits_norm, loss_list, acc_list, valid_sequence_ratio)
        """
        device = x_BLC[0].device

        # [1. get input sequence x_BLC]
        # word embedding
        sub_L_list = [item.shape[1] for item in x_BLC]
        cat_x_BLC = torch.cat(x_BLC, dim=1)
        with torch.amp.autocast("cuda", dtype=torch.float32):
            cat_x_BLC = self.word_embed(cat_x_BLC.float())
        x_BLC = list(torch.split(cat_x_BLC, sub_L_list, dim=1))

        # text tokens embedding
        kv_compact, lens, cu_seqlens_k, max_seqlen_k, _ = label_B_or_BLT
        with torch.amp.autocast("cuda", dtype=torch.float32):
            kv_compact = self.text_proj(kv_compact).contiguous()  # [sum(lens), C]
        kv_compact_splits = torch.split(kv_compact, lens, dim=0)

        # scale tokens embedding
        scale_token_ids = torch.tensor(
            [info["scale_token_id"] for info in other_info_by_scale], device=device
        )
        with torch.amp.autocast("cuda", dtype=torch.float32):
            pt_tokens = self.pt_embedder((scale_token_ids))  # [num_scales, C]

        # construct final X_BLC input, [visual token, text token, scale token]
        x_BLC_lists = []
        for i in range(len(x_BLC)):
            x_BLC_lists.extend(
                [x_BLC[i], kv_compact_splits[i].unsqueeze(0), pt_tokens[i][None, None]]
            )
        x_BLC = torch.cat(x_BLC_lists, dim=1)

        valid_sequence_ratio = x_BLC.shape[1] / self.other_args.train_max_token_len
        attn_bias_or_two_vector = None

        # calculate finalrope cache, [visual token, text token, scale token]
        self.rope2d_freqs_grid["freqs_text"] = self.rope2d_freqs_grid["freqs_text"].to(
            x_BLC.device
        )
        rope_cache_list = []
        for i in range(len(visual_rope_cache)):
            rope_cache_list.append(visual_rope_cache[i])
            rope_cache_list.append(
                self.rope2d_freqs_grid["freqs_text"][:, :, :, :, : lens[i]]
            )
            rope_cache_list.append(
                self.rope2d_freqs_grid["freqs_text"][
                    :, :, :, :, 512 : 512 + self.other_args.add_scale_token
                ]
            )
        rope_cache = torch.cat(
            rope_cache_list, dim=4
        )  # (2, 1, 1, 1, seq_len, head_dim / 2)
        assert rope_cache.shape[4] == x_BLC.shape[1], (
            f"{rope_cache.shape[4]} != {x_BLC.shape[1]}"
        )
        rope_cache = rope_cache[
            :, 0
        ].permute(
            0, 1, 3, 2, 4
        )  # (2, 1, 1, 1, seq_len, head_dim / 2) -> (2, 1, 1, seq_len, head_dim / 2) -> (2, 1, seq_len, 1, head_dim / 2)

        e, e0 = None, None

        # [2. block loop]
        checkpointing_full_block = self.checkpointing == "full-block" and self.training

        if sp_manager.sp_on():
            # [B, raw_L, C] --> [B, raw_L/sp_size, C]
            x_BLC = sp_split_sequence_by_dim(x_BLC, 1)

        cu_seqlens = (
            torch.tensor([0] + super_scale_lengths, device=device)
            .cumsum(-1)
            .to(torch.int32)
        )
        max_seqlen = max(super_scale_lengths)
        for i, chunk in enumerate(self.block_chunks):  # this path
            x_BLC = self.before_block_chunk(i, x_BLC)
            x_BLC = chunk(
                x=x_BLC,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                e0=e0,
                attn_bias_or_two_vector=attn_bias_or_two_vector,
                checkpointing_full_block=checkpointing_full_block,
                rope2d_freqs_grid=rope_cache,
            )

        if sp_manager.sp_on():
            # [B, raw_L/sp_size, C] --> [B, raw_L, C]
            x_BLC = sp_gather_sequence_by_dim(x_BLC, 1)

        # [3. unpad the seqlen dim, and then get logits]
        logits_norm, loss_list, acc_list = self.get_loss_acc(
            x_BLC,
            x_BLC_mask,
            e,
            sequece_packing_scales,
            gt_BL,
            other_info_by_scale,
            return_last_hidden_states,
        )
        return logits_norm, loss_list, acc_list, valid_sequence_ratio

    def prepare_text_conditions(
        self,
        label_B_or_BLT: Tuple[torch.Tensor, ...],
        negative_label_B_or_BLT: Optional[Tuple[torch.Tensor, ...]],
        use_cfg: bool = False,
    ) -> Tuple[torch.Tensor, List[int]]:
        """Prepare text conditions for inference."""
        kv_compact, lens, cu_seqlens_k, max_seqlen_k = label_B_or_BLT
        if use_cfg:
            kv_compact_un, lens_un, cu_seqlens_k_un, max_seqlen_k_un = (
                negative_label_B_or_BLT
            )
            kv_compact = torch.cat((kv_compact, kv_compact_un), dim=0)
            cu_seqlens_k = torch.cat(
                (cu_seqlens_k, cu_seqlens_k_un[1:] + cu_seqlens_k[-1]), dim=0
            )
            max_seqlen_k = max(max_seqlen_k, max_seqlen_k_un)
            lens = lens + lens_un
        kv_compact = self.text_proj(kv_compact).contiguous()
        return kv_compact, lens

    def embeds_codes2input(self, last_stage: torch.Tensor) -> torch.Tensor:
        """Embed discrete codes into continuous input representations."""
        last_stage = last_stage.reshape(
            *last_stage.shape[:2], -1
        )  # [B, d, t*h*w] or [B, 4d, t*h*w]
        last_stage = torch.permute(
            last_stage, [0, 2, 1]
        )  # [B, t*h*w, d] or [B, t*h*w, 4d]
        last_stage = self.word_embed(last_stage)  # norm0_ve is Identity
        return last_stage

    def _run_inference_transformer(
        self,
        hidden: torch.Tensor,
        *,
        sequence_lengths: List[int],
        rope_cache: torch.Tensor,
        block_chunks: Any,
        use_slow_attn: bool,
    ) -> torch.Tensor:
        """Run one or more packed inference streams through the transformer."""

        cu_seqlens = (
            torch.tensor(
                [0] + [int(length) for length in sequence_lengths],
                device=hidden.device,
            )
            .cumsum(-1)
            .to(torch.int32)
        )
        max_seqlen = max(sequence_lengths)
        attn_mask = (
            build_attn_mask(sequence_lengths, hidden.device) if use_slow_attn else None
        )
        for block_idx, block in enumerate(block_chunks):
            hidden = self.before_block_chunk(block_idx, hidden)
            hidden = block(
                x=hidden,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                e0=None,
                attn_bias_or_two_vector=attn_mask,
                rope2d_freqs_grid=rope_cache,
            )
        return self.get_logits_during_infer(hidden, e=None)

    def _inference_stream_logits(
        self,
        cond_hidden: torch.Tensor,
        uncond_hidden: Optional[torch.Tensor],
        *,
        cond_length: int,
        uncond_length: Optional[int],
        cond_rope_cache: torch.Tensor,
        uncond_rope_cache: Optional[torch.Tensor],
        block_chunks: Any,
        use_slow_attn: bool,
        guidance_mode: str,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Evaluate the official single-stream or packed two-stream CFG path."""

        if guidance_mode == "none":
            cond_logits = self._run_inference_transformer(
                cond_hidden,
                sequence_lengths=[cond_length],
                rope_cache=cond_rope_cache,
                block_chunks=block_chunks,
                use_slow_attn=use_slow_attn,
            )
            return cond_logits, None
        if guidance_mode != "standard":
            raise ValueError(f"unsupported GRN guidance mode: {guidance_mode!r}")
        if uncond_hidden is None or uncond_length is None or uncond_rope_cache is None:
            raise ValueError("standard CFG requires a negative inference stream")

        packed_logits = self._run_inference_transformer(
            torch.cat((cond_hidden, uncond_hidden), dim=1),
            sequence_lengths=[cond_length, uncond_length],
            rope_cache=torch.cat((cond_rope_cache, uncond_rope_cache), dim=2),
            block_chunks=block_chunks,
            use_slow_attn=use_slow_attn,
        )
        return (
            packed_logits[:, :cond_length],
            packed_logits[:, cond_length : cond_length + uncond_length],
        )

    @torch.no_grad()
    def autoregressive_infer(
        self,
        *,
        vae: Any,
        scale_schedule: List[Tuple[int, int, int]],
        label_B_or_BLT: Any,
        negative_label_B_or_BLT: Any = None,
        cfg_list: Optional[List[float]] = None,
        tau_list: Optional[List[float]] = None,
        args: Any,
        get_visual_rope_embeds: Any,
        guidance_mode: str = "none",
    ) -> Tuple[list, list, torch.Tensor]:
        """Single-scale bit refinement; CFG-1 is standard CFG with shared edit gates.

        The caller seeds torch once per sample, matching the training repository.
        Every iteration samples bits, then blends them with the initial noise by
        the shifted cosine keep-probability; the last sampled bits are decoded.
        """
        from .inference_ops import shift_pt

        if len(scale_schedule) != 1:
            raise ValueError("Stage-LoRA inference uses one latent video scale")
        if guidance_mode not in {"none", "standard"}:
            raise ValueError("guidance_mode must be none or standard")
        if args.max_infer_steps <= 0 or args.complexity_aware_Tmax <= 1:
            raise ValueError(
                "Refinement requires positive steps and complexity_aware_Tmax > 1"
            )
        use_cfg = guidance_mode == "standard"
        guidance = float(cfg_list[0]) if cfg_list else 1.0
        temperature = float(tau_list[0]) if tau_list else 1.0
        if temperature <= 0:
            raise ValueError("Sampling temperature must be positive")
        cfg_interval = float(args.cfg_type.split("_")[-1])
        pt, ph, pw = scale_schedule[0]
        visual_length = pt * ph * pw
        prefix, lengths = self.prepare_text_conditions(
            label_B_or_BLT[0], negative_label_B_or_BLT, use_cfg
        )
        device, dtype = prefix.device, prefix.dtype
        prefix = torch.split(prefix, lengths, dim=0)
        self.rope2d_freqs_grid["freqs_text"] = self.rope2d_freqs_grid["freqs_text"].to(
            device
        )
        text_rope = self.rope2d_freqs_grid["freqs_text"]
        visual_rope = get_visual_rope_embeds(
            self.rope2d_freqs_grid,
            (pt, ph, pw),
            device,
            args.mapped_h_div_w_template,
            t_offset=0,
        )
        pt_rope = text_rope[:, :, :, :, 512:513]
        stream_lengths = [visual_length + length + 1 for length in lengths]
        rope_caches = [
            torch.cat((visual_rope, text_rope[:, :, :, :, :length], pt_rope), dim=4)[
                :, 0
            ].permute(0, 1, 3, 2, 4)
            for length in lengths
        ]
        labels_shape = (1, args.detail_scale_dim * args.hbq_round, pt, ph, pw)
        initial_noise = torch.randint(0, 2, labels_shape, device=device, dtype=dtype)
        mixed = initial_noise
        next_pt = 0.0
        for step in range(args.max_infer_steps):
            cur_pt = next_pt
            is_last_step = np.abs(cur_pt - 1) < 0.02
            visual = self.embeds_codes2input(multiclass_labels2onehot_input(mixed, 2))
            pt_token = self.pt_embedder(
                torch.tensor([cur_pt], device=device)
            ).unsqueeze(0)
            streams = [
                torch.cat((visual, text.unsqueeze(0), pt_token), dim=1)
                for text in prefix
            ]
            positive, negative = self._inference_stream_logits(
                streams[0],
                streams[1] if use_cfg else None,
                cond_length=stream_lengths[0],
                uncond_length=stream_lengths[1] if use_cfg else None,
                cond_rope_cache=rope_caches[0],
                uncond_rope_cache=rope_caches[1] if use_cfg else None,
                block_chunks=self.block_chunks,
                use_slow_attn=bool(args.use_slow_attn),
                guidance_mode=guidance_mode,
            )
            positive = positive.reshape(1, stream_lengths[0], -1, 2)[:, :visual_length]
            cfg = guidance if use_cfg and cur_pt >= cfg_interval else 1.0
            if cfg != 1.0:
                negative = negative.reshape(1, stream_lengths[1], -1, 2)[
                    :, :visual_length
                ]
                logits = negative + cfg * (positive - negative)
            else:
                logits = positive
            probabilities = logits.mul(1 / temperature).softmax(dim=-1)
            sampled = torch.multinomial(
                probabilities.reshape(-1, 2), num_samples=1, replacement=True
            )
            sampled = sampled.reshape(1, pt, ph, pw, -1).permute(0, 4, 1, 2, 3)
            progress = min(1.0, (step + 1) / (args.complexity_aware_Tmax - 1))
            keep_probability = (
                1 - np.cos(np.pi / 2 * shift_pt(progress, args.snr_shift))
            ) * 0.95
            keep = torch.rand(sampled.shape, device=device) < keep_probability
            mixed = torch.where(keep, sampled, initial_noise)
            next_pt = keep.float().mean().item()
            if is_last_step:
                break
        raw_feature = bit_label2raw_feature(sampled, hbq_round=args.hbq_round)
        video = vae.decode(raw_feature, slice=True)
        video = ((video + 1) / 2).clamp(0, 1).permute(0, 2, 3, 4, 1)
        video = video.mul(255).to(torch.uint8).flip(dims=(4,))
        return [], [], video
