import math
import time
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

import grn.utils_t2iv.dist as dist
from grn.official_t2iv_edit.basic import FastRMSNorm, SelfAttnBlock
from grn.official_t2iv_edit.rope import precompute_rope3d_freqs_grid
from grn.schedules.dynamic_resolution import get_dynamic_resolution_meta
from grn.utils_t2iv.sequence_parallel import SequenceParallelManager as sp_manager
from grn.utils_t2iv.sequence_parallel import sp_gather_sequence_by_dim, sp_split_sequence_by_dim


class MultipleLayers(nn.Module):
    """A sequential container for a chunk of multiple transformer blocks."""

    def __init__(self, layers: List[nn.Module], num_blocks: int, start_index: int):
        super().__init__()
        self.module = nn.ModuleList([
            layers[i] for i in range(start_index, start_index + num_blocks)
        ])

    def forward(
        self, x, cu_seqlens, max_seqlen, e0: Optional[torch.Tensor],
        attn_bias_or_two_vector: Optional[Any],
        checkpointing_full_block: bool = False, rope2d_freqs_grid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = x
        for m in self.module:
            if checkpointing_full_block:
                h = torch.utils.checkpoint.checkpoint(
                    m, h, cu_seqlens, max_seqlen, e0, attn_bias_or_two_vector,
                    rope2d_freqs_grid, use_reentrant=False
                )
            else:
                h = m(
                    h, cu_seqlens, max_seqlen, e0, attn_bias_or_two_vector,
                    rope2d_freqs_grid
                )
        return h


def sinusoidal_embedding_1d(dim: int, position: torch.Tensor) -> torch.Tensor:
    """
    Generate 1D sinusoidal embeddings.

    Args:
        dim (int): Embedding dimension (must be even).
        position (torch.Tensor): Position tensor of shape [B, L].

    Returns:
        torch.Tensor: Embeddings of shape [B, L, dim].
    """
    if dim % 2 != 0:
        raise ValueError(f"Embedding dimension must be even, got {dim}")

    half = dim // 2
    b, l = position.shape
    position = position.reshape(-1).type(torch.float64)

    sinusoid = torch.outer(
        position,
        torch.pow(10000, -torch.arange(half).to(position).div(half))
    )
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.reshape(b, l, dim)


class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        """Create sinusoidal timestep embeddings."""
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


def bld_to_bthwd(item: torch.Tensor, patch_time: int, patch_height: int, patch_width: int, apply_spatial_patchify: bool = False) -> torch.Tensor:
    """Reshape a sequence tensor to a spatial tensor."""
    batch_size = item.shape[0]
    return item.reshape(batch_size, patch_time, patch_height, patch_width, -1)


def clip_binary_logits_max_prob(logits: torch.Tensor, max_prob: float) -> torch.Tensor:
    """Limit binary-logit confidence without changing the preferred class."""
    if logits.shape[-1] != 2:
        raise ValueError(f"clip_binary_logits_max_prob expects binary logits, got shape={tuple(logits.shape)}")
    if not 0.5 <= float(max_prob) <= 1.0:
        raise ValueError(f"max_prob must be in [0.5, 1.0], got {max_prob}.")
    if float(max_prob) == 1.0:
        return logits
    max_prob_t = torch.as_tensor(float(max_prob), dtype=torch.float32, device=logits.device)
    max_margin = torch.log(max_prob_t / (1.0 - max_prob_t)).to(dtype=logits.dtype)
    center = logits.mean(dim=-1, keepdim=True)
    margin = (logits[..., 1:2] - logits[..., 0:1]).clamp(min=-max_margin, max=max_margin)
    return torch.cat((center - 0.5 * margin, center + 0.5 * margin), dim=-1)


def build_attn_mask(seqlens, device):
    attn_mask = torch.zeros((1, 1, sum(seqlens), sum(seqlens)), dtype=torch.bool, device=device)
    q_start = 0
    for i in range(len(seqlens)):
        q_len = seqlens[i]
        q_end = q_start + q_len
        attn_mask[:, :, q_start:q_end, q_start:q_end] = True
        q_start = q_end
    return attn_mask


class FsqHead(nn.Module):
    """Classification head for Finite Scalar Quantization (FSQ)."""

    def __init__(self, hidden_dim: int, fsq_dim: int, fsq_lvl: int, use_ada_layer_norm: bool, eps: float = 1e-6):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, fsq_dim * fsq_lvl)
        self.norm = FastRMSNorm(hidden_dim)

    def forward(self, x: torch.Tensor, e: Optional[torch.Tensor] = None) -> torch.Tensor:
        with torch.amp.autocast('cuda', dtype=torch.float32):
            return self.proj(self.norm(x))


def get_scale_token_rope_offset(args: Any) -> int:
    return int(getattr(args, 'tlen', 512) or 512)


class GRN(nn.Module):
    def __init__(
        self,
        vae_local: Any,
        arch: str = 'var',
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
        pad_to_multiplier: int = 0,
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

        if other_args.refine_mode in ['ar_discrete_GRN_ind']:
            self.visual_embedding_in_dim = vae_local.codebook_dim * (2**hbq_round)
            classifier_head_dim = vae_local.codebook_dim
        elif other_args.refine_mode in ['ar_discrete_GRN_bit']:
            self.visual_embedding_in_dim = hbq_round * vae_local.codebook_dim * 2
            classifier_head_dim = hbq_round * vae_local.codebook_dim
        else:
            self.visual_embedding_in_dim = vae_local.codebook_dim

        if self.apply_spatial_patchify:
            self.visual_embedding_in_dim *= 4

        # 3. Dynamic Resolution & Video Specifics
        self.video_frames = video_frames
        self.always_training_scales = always_training_scales
        self.num_of_label_value = num_of_label_value
        self.rope2d_normalized_by_hw = rope2d_normalized_by_hw

        self.dynamic_resolution_h_w, self.h_div_w_templates = get_dynamic_resolution_meta(
            other_args.dynamic_scale_schedule, other_args.train_h_div_w_list, other_args.video_frames
        )
        self.train_h_div_w_list = self.h_div_w_templates
        print(f"train_h_div_w_list: {self.train_h_div_w_list}")

        # 4. Utilities
        self.entrophy_statistics = []
        self.top_p, self.top_k = 1.0, 100
        self.rng = torch.Generator(device=dist.get_device())
        self.maybe_record_function = nullcontext
        self.infer_ts = None

        # 5. Model Components (Projections, Embeddings)
        self.norm0_cond = nn.Identity()
        self.text_proj = nn.Linear(self.text_channels, self.embed_dim)

        if self.other_args.use_ada_layer_norm:
            self.scale_or_time_dim = 256
            self.scale_or_time_embedding = nn.Sequential(
                nn.Linear(self.scale_or_time_dim, self.embed_dim), nn.SiLU(), nn.Linear(self.embed_dim, self.embed_dim),
            )
            self.scale_or_time_projection = nn.Sequential(nn.SiLU(), nn.Linear(self.embed_dim, self.embed_dim * 6))

        tmp_h_div_w_template = self.train_h_div_w_list[0]

        # RoPE grid initialization
        with torch.amp.autocast('cuda', dtype=torch.float32):
            self.rope2d_freqs_grid = precompute_rope3d_freqs_grid(
                dim=self.embed_dim // self.num_heads,
                rope2d_normalized_by_hw=self.rope2d_normalized_by_hw,
                activated_h_div_w_templates=self.train_h_div_w_list,
                max_scales=1010,
                max_frames=int(self.video_frames / other_args.temporal_compress_rate + 1),
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
        assert self.num_blocks_in_a_chunk * self.num_block_chunks == depth, "Depth must be divisible by block_chunks"

        self.block_chunks = nn.ModuleList([
            MultipleLayers(self.unregistered_blocks, self.num_blocks_in_a_chunk, i * self.num_blocks_in_a_chunk)
            for i in range(self.num_block_chunks)
        ])

        print(f"    [Model Config] embed_dim={embed_dim}, num_heads={num_heads}, depth={depth}, "
              f"mlp_ratio={mlp_ratio}, num_blocks_in_a_chunk={self.num_blocks_in_a_chunk}")
        print(f"    drop_path_rate={drop_path_rate:g}", end='\n\n', flush=True)

    def get_loss_acc(
        self,
        hidden_states: torch.Tensor,
        hidden_states_mask: Optional[torch.Tensor],
        e: Optional[torch.Tensor],
        sequence_packing_scales: List[List[Tuple[int, int, int]]],
        gt: List[torch.Tensor],
        other_info_by_scale: List[Dict[str, Any]],
        return_last_hidden_states: bool
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
                cur_bits = other_info_by_scale[global_scale_ptr]['cur_bits']
                cur_lvl = other_info_by_scale[global_scale_ptr]['cur_lvl']
                predict_tokens = other_info_by_scale[global_scale_ptr]['predict_tokens']
                all_tokens = other_info_by_scale[global_scale_ptr]['all_tokens']
                logits = logits_full[:, global_token_ptr:global_token_ptr + predict_tokens]
                logits = logits.reshape(hidden_states.shape[0], mul_pt_ph_pw, cur_bits, cur_lvl)
                logits = logits.permute(0, 3, 1, 2) # [1, num_of_label_value, mul_pt_ph_pw, d]

                logits_norm.append(logits.abs().mean())

                # gt[global_scale_ptr]: [1, mul_pt_ph_pw, d]
                loss_this_scale = F.cross_entropy(logits, gt[global_scale_ptr], reduction='none')[0] # [mul_pt_ph_pw, d]
                acc_this_scale = (logits.argmax(1) == gt[global_scale_ptr]).float()[0] # [mul_pt_ph_pw, d]

                loss_list.append(loss_this_scale.mean(-1))
                acc_list.append(acc_this_scale.mean(-1))

                global_scale_ptr += 1
                global_token_ptr += all_tokens

        loss_tensor = torch.cat(loss_list) if loss_list else torch.tensor([], device=hidden_states.device)
        acc_tensor = torch.cat(acc_list) if acc_list else torch.tensor([], device=hidden_states.device)
        logits_norm_tensor = torch.stack(logits_norm).mean() if logits_norm else torch.tensor(0.0, device=hidden_states.device)

        return logits_norm_tensor, loss_tensor, acc_tensor

    def get_logits_during_infer(self, hidden_states: torch.Tensor, e: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Get logits during inference."""
        return self.head(hidden_states.float(), e)

    def _project_text_conditions(
        self,
        label_B_or_BLT: Tuple[torch.Tensor, ...],
    ) -> Tuple[torch.Tensor, List[int], torch.Tensor, int, Any]:
        kv_compact, lens, cu_seqlens_k, max_seqlen_k, caption_nums = label_B_or_BLT
        with torch.amp.autocast('cuda', dtype=torch.float32):
            kv_compact = self.text_proj(kv_compact).contiguous()
        return kv_compact, list(lens), cu_seqlens_k, int(max_seqlen_k), caption_nums

    def forward(
        self,
        label_B_or_BLT: Union[torch.LongTensor, Tuple[torch.FloatTensor, torch.IntTensor, int]],
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
        with torch.amp.autocast('cuda', dtype=torch.float32):
            cat_x_BLC = self.word_embed(cat_x_BLC.float())
        x_BLC = list(torch.split(cat_x_BLC, sub_L_list, dim=1))

        # text tokens embedding
        kv_compact, lens, cu_seqlens_k, max_seqlen_k, _ = self._project_text_conditions(label_B_or_BLT)
        kv_compact_splits = torch.split(kv_compact, lens, dim=0)

        # scale tokens embedding
        scale_token_ids = torch.tensor([info["scale_token_id"] for info in other_info_by_scale], device=device)
        with torch.amp.autocast("cuda", dtype=torch.float32):
            pt_tokens = self.pt_embedder((scale_token_ids)) # [num_scales, C]

        # construct final X_BLC input, [visual token, text token, scale token]
        x_BLC_lists = []
        for i in range(len(x_BLC)):
            x_BLC_lists.extend([x_BLC[i], kv_compact_splits[i].unsqueeze(0), pt_tokens[i][None, None]])
        x_BLC = torch.cat(x_BLC_lists, dim=1)

        valid_sequence_ratio = x_BLC.shape[1] / self.other_args.train_max_token_len
        attn_bias_or_two_vector = None

        # calculate finalrope cache, [visual token, text token, scale token]
        self.rope2d_freqs_grid['freqs_text'] = self.rope2d_freqs_grid['freqs_text'].to(x_BLC.device)
        rope_cache_list = []
        scale_token_rope_offset = get_scale_token_rope_offset(self.other_args)
        for i in range(len(visual_rope_cache)):
            rope_cache_list.append(visual_rope_cache[i])
            rope_cache_list.append(self.rope2d_freqs_grid['freqs_text'][:,:,:,:,:lens[i]])
            rope_cache_list.append(
                self.rope2d_freqs_grid['freqs_text'][
                    :, :, :, :, scale_token_rope_offset:scale_token_rope_offset + self.other_args.add_scale_token
                ]
            )
        rope_cache = torch.cat(rope_cache_list, dim=4) # (2, 1, 1, 1, seq_len, head_dim / 2)
        assert rope_cache.shape[4] == x_BLC.shape[1], f'{rope_cache.shape[4]} != {x_BLC.shape[1]}'
        rope_cache = rope_cache[:,0].permute(0, 1, 3, 2, 4) # (2, 1, 1, 1, seq_len, head_dim / 2) -> (2, 1, 1, seq_len, head_dim / 2) -> (2, 1, seq_len, 1, head_dim / 2)

        # calculate time or scale embeddings
        if self.other_args.use_ada_layer_norm:
            with torch.amp.autocast('cuda', dtype=torch.float32):
                e = self.scale_or_time_embedding(sinusoidal_embedding_1d(self.scale_or_time_dim, scale_or_time_ids).float()) # [1, visual_seq_len,] -> [1, visual_seq_len, 256] -> [1, visual_seq_len, C]
                if e.shape[1] < x_BLC.shape[1]:
                    e = F.pad(e, (0,0,0,x_BLC.shape[1]-e.shape[1]), 'constant', 0.) # [1, visual_seq_len, C] -> [1, L, C]
                e0 = self.scale_or_time_projection(e).unflatten(2, (6, self.C)) # [1, L, C] -> [1, L, 6C] -> [1, L, 6, C]
                assert e.dtype == torch.float32 and e0.dtype == torch.float32
        else:
            e, e0 = None, None

        # [2. block loop]
        checkpointing_full_block = self.checkpointing == 'full-block' and self.training

        if sp_manager.sp_on():
            # [B, raw_L, C] --> [B, raw_L/sp_size, C]
            x_BLC = sp_split_sequence_by_dim(x_BLC, 1)

        cu_seqlens = torch.tensor([0]+super_scale_lengths, device=device).cumsum(-1).to(torch.int32)
        max_seqlen = max(super_scale_lengths)
        for i, chunk in enumerate(self.block_chunks):
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
        logits_norm, loss_list, acc_list = self.get_loss_acc(x_BLC, x_BLC_mask, e, sequece_packing_scales, gt_BL, other_info_by_scale, return_last_hidden_states)
        return logits_norm, loss_list, acc_list, valid_sequence_ratio

    def summed_codes2images(self, vae: Any, summed_codes: torch.Tensor) -> torch.Tensor:
        """Decode summed codes into images using the VAE."""
        t1 = time.time()
        img = vae.decode(summed_codes, slice=True)
        img = (img + 1) / 2
        img = torch.clamp(img, 0, 1)
        img = img.permute(0, 2, 3, 4, 1) # [bs, 3, t, h, w] -> [bs, t, h, w, 3]
        img = img.mul_(255).to(torch.uint8).flip(dims=(4,))
        print(f"Decode takes {time.time() - t1:.1f}s")
        return img # bgr order

    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = False, assign: bool = False) -> Any:
        return super().load_state_dict(state_dict=state_dict, strict=strict, assign=assign)

    def special_init(self, **kwargs: Any) -> None:
        """Apply special initialization to specific layers."""
        std = 0.02
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=std)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=std)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()

    def extra_repr(self) -> str:
        return f'drop_path_rate={self.drop_path_rate}'

    def get_layer_id_and_scale_exp(self, para_name: str) -> Any:
        raise NotImplementedError

TIMM_KEYS = {'img_size', 'pretrained', 'pretrained_cfg', 'pretrained_cfg_overlay', 'global_pool'}
