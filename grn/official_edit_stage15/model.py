import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from timm.models import register_model

from grn.official_t2iv_edit.model import (
    GRN as OfficialEditGRN,
    TIMM_KEYS,
    bld_to_bthwd,
    build_attn_mask,
    get_scale_token_rope_offset,
    sinusoidal_embedding_1d,
)
from grn.official_t2iv_edit.basic import FastRMSNorm
from grn.official_edit_stage15.text_conditioning import (
    STAGE15_TEXT_PAIR_MARKER,
    STAGE15_TEXT_SEPARATOR_TOKENS,
)
from grn.utils_t2iv.hbq_util_t2iv import multiclass_labels2onehot_input
from grn.utils_t2iv.sequence_parallel import SequenceParallelManager as sp_manager
from grn.utils_t2iv.sequence_parallel import sp_gather_sequence_by_dim, sp_split_sequence_by_dim


STAGE15_SOURCE_INJECTION_MASK = (True, False, False, False, False, False, True)
STAGE15_SOURCE_INJECTION_MASK_JSON = '[1,0,0,0,0,0,1]'


class SourceResidualTextPtModulation(nn.Module):
    """Text- and progress-conditioned modulation for one source residual stream."""

    def __init__(self, hidden_dim: int, bottleneck_dim: int = 256):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.context_norm = FastRMSNorm(hidden_dim * 2)
        self.in_proj = nn.Linear(hidden_dim * 2, bottleneck_dim)
        self.act = nn.SiLU()
        self.out_proj = nn.Linear(bottleneck_dim, hidden_dim * 2)
        self.reset_output()

    def reset_output(self) -> None:
        with torch.no_grad():
            self.out_proj.weight.zero_()
            if self.out_proj.bias is not None:
                self.out_proj.bias.zero_()

    def forward(
        self,
        source_hidden: torch.Tensor,
        text_context: torch.Tensor,
        pt_context: torch.Tensor,
    ) -> torch.Tensor:
        if source_hidden.ndim != 3:
            raise ValueError(f'source_hidden must be [B,L,C], got {tuple(source_hidden.shape)}')
        out_dtype = source_hidden.dtype
        if text_context.ndim != 2 or pt_context.ndim != 2:
            raise ValueError(
                'text_context and pt_context must be [B,C]: '
                f'text={tuple(text_context.shape)}, pt={tuple(pt_context.shape)}'
            )
        if text_context.shape != pt_context.shape:
            raise ValueError(f'text/pt context shape mismatch: {tuple(text_context.shape)} != {tuple(pt_context.shape)}')
        if source_hidden.shape[0] != text_context.shape[0] or source_hidden.shape[-1] != text_context.shape[-1]:
            raise ValueError(
                'source/context shape mismatch: '
                f'source={tuple(source_hidden.shape)}, text={tuple(text_context.shape)}'
            )

        context = torch.cat([text_context.detach(), pt_context.detach()], dim=-1)
        context = context.to(device=source_hidden.device, dtype=source_hidden.dtype)
        with torch.amp.autocast('cuda', dtype=torch.float32):
            scale, gate_delta = self.out_proj(self.act(self.in_proj(self.context_norm(context.float())))).chunk(2, dim=-1)
            gate = gate_delta
            residual = source_hidden.float() * gate[:, None, :] * (1.0 + scale[:, None, :])
        return residual.to(dtype=out_dtype)


class Stage15LayerwiseSourceConditionedGRN(OfficialEditGRN):
    """Official GRN with entry and per-chunk text/pt-aware source residuals."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.source_word_embed = nn.Linear(self.visual_embedding_in_dim, self.embed_dim)

        num_deep_injections = max(0, int(self.num_block_chunks) - 1)
        self.deep_source_word_embeds = nn.ModuleList(
            [nn.Linear(self.visual_embedding_in_dim, self.embed_dim) for _ in range(num_deep_injections)]
        )
        self.source_residual_modulations = nn.ModuleList(
            [SourceResidualTextPtModulation(self.embed_dim) for _ in range(num_deep_injections + 1)]
        )

    def _copy_word_embed_to_source(self, source_embed: nn.Linear) -> None:
        source_embed.weight.copy_(self.word_embed.weight)
        if self.word_embed.bias is not None and source_embed.bias is not None:
            source_embed.bias.copy_(self.word_embed.bias)
        elif source_embed.bias is not None:
            source_embed.bias.zero_()

    def init_stage15_source_from_word_embed(self) -> None:
        with torch.no_grad():
            self._copy_word_embed_to_source(self.source_word_embed)
            for source_embed in self.deep_source_word_embeds:
                self._copy_word_embed_to_source(source_embed)
            for modulation in self.source_residual_modulations:
                modulation.reset_output()

    def freeze_disabled_source_modules(self) -> None:
        mask = self._stage15_source_injection_mask()
        modules = [(self.source_word_embed, self.source_residual_modulations[0])]
        modules.extend(
            zip(self.deep_source_word_embeds, self.source_residual_modulations[1:])
        )
        for enabled, pair in zip(mask, modules):
            for module in pair:
                module.requires_grad_(bool(enabled))

    def special_init(self, **kwargs: Any) -> None:
        super().special_init(**kwargs)
        self.init_stage15_source_from_word_embed()

    def _stage15_source_required_keys(self) -> set:
        keys = {
            'source_word_embed.weight',
            'source_word_embed.bias',
        }
        for i in range(len(self.deep_source_word_embeds)):
            keys.add(f'deep_source_word_embeds.{i}.weight')
            keys.add(f'deep_source_word_embeds.{i}.bias')
        return keys

    def _stage15_modulation_required_keys(self) -> set:
        return {f'source_residual_modulations.{key}' for key in self.source_residual_modulations.state_dict().keys()}

    def _stage15_required_keys(self) -> set:
        return self._stage15_source_required_keys() | self._stage15_modulation_required_keys()

    @staticmethod
    def _normalize_checkpoint_key(key: str) -> str:
        prefixes = ('module.', '_orig_mod.', '_fsdp_wrapped_module.')
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if key.startswith(prefix):
                    key = key[len(prefix):]
                    changed = True
        return key

    @classmethod
    def _normalize_checkpoint_state_dict(cls, state_dict: Dict[str, Any]) -> Dict[str, Any]:
        normalized = {}
        owners = {}
        for key, value in state_dict.items():
            normalized_key = cls._normalize_checkpoint_key(str(key))
            if normalized_key in normalized and owners[normalized_key] != key:
                raise RuntimeError(
                    'Ambiguous Stage1.5 checkpoint keys after prefix normalization: '
                    f'{owners[normalized_key]!r} and {key!r} both map to {normalized_key!r}.'
                )
            normalized[normalized_key] = value
            owners[normalized_key] = key
        return normalized

    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = False, assign: bool = False) -> Any:
        state_dict = self._normalize_checkpoint_state_dict(state_dict)
        required = self._stage15_required_keys()
        provided = required.intersection(state_dict)
        if provided and provided != required:
            missing = sorted(required - provided)
            raise RuntimeError(f'Incomplete GRNEdit Stage I checkpoint. Missing keys: {missing[:30]}.')

        result = super().load_state_dict(state_dict=state_dict, strict=False, assign=assign)
        if strict and (result.missing_keys or result.unexpected_keys):
            raise RuntimeError(
                'Strict GRNEdit Stage I load failed: '
                f'missing={result.missing_keys[:30]}, unexpected={result.unexpected_keys[:30]}.'
            )
        if not provided:
            self.init_stage15_source_from_word_embed()
        return result

    @staticmethod
    def _mean_text_context(text_tokens: torch.Tensor) -> torch.Tensor:
        if text_tokens.ndim != 2 or text_tokens.shape[0] <= 0:
            raise ValueError(f'text tokens must be non-empty [T,C], got {tuple(text_tokens.shape)}')
        return text_tokens.mean(dim=0)

    def _build_source_contexts(
        self,
        kv_compact_splits: Tuple[torch.Tensor, ...],
        pt_tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(kv_compact_splits) != pt_tokens.shape[0]:
            raise ValueError(f'text/pt context count mismatch: {len(kv_compact_splits)} != {pt_tokens.shape[0]}')
        text_context = torch.stack([self._mean_text_context(tokens) for tokens in kv_compact_splits], dim=0)
        if text_context.shape != pt_tokens.shape:
            raise ValueError(f'text/pt context shape mismatch: {tuple(text_context.shape)} != {tuple(pt_tokens.shape)}')
        return text_context, pt_tokens

    def _stage15_source_injection_mask(self) -> List[bool]:
        raw_mask = getattr(self.other_args, 'stage15_source_injection_mask', '')
        if raw_mask is None or raw_mask == '':
            raise ValueError('stage15_source_injection_mask is required for GRNEdit.')
        if isinstance(raw_mask, str):
            try:
                parsed = json.loads(raw_mask)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    'stage15_source_injection_mask must be a JSON list with one 0/1 value '
                    f'per block chunk, got {raw_mask!r}.'
                ) from exc
        else:
            parsed = raw_mask
        if not isinstance(parsed, (list, tuple)):
            raise ValueError(f'stage15_source_injection_mask must be a list, got {type(parsed).__name__}.')
        if len(parsed) != int(self.num_block_chunks):
            raise ValueError(
                'stage15_source_injection_mask length must match num_block_chunks: '
                f'{len(parsed)} != {self.num_block_chunks}.'
            )
        mask = []
        for idx, item in enumerate(parsed):
            if isinstance(item, bool):
                mask.append(bool(item))
            elif isinstance(item, int) and item in (0, 1):
                mask.append(bool(item))
            else:
                raise ValueError(
                    'stage15_source_injection_mask entries must be bool or 0/1 integers, '
                    f'got index {idx}: {item!r}.'
                )
        if tuple(mask) != STAGE15_SOURCE_INJECTION_MASK:
            raise ValueError(
                'GRNEdit supports only endpoint source injection '
                f'{STAGE15_SOURCE_INJECTION_MASK_JSON}, got {list(map(int, mask))}.'
            )
        return mask

    def _stage15_source_injection_enabled(self, chunk_index: int) -> bool:
        if not 0 <= int(chunk_index) < int(self.num_block_chunks):
            raise ValueError(f'chunk_index out of range for Stage1.5 source injection: {chunk_index}.')
        return self._stage15_source_injection_mask()[int(chunk_index)]

    @staticmethod
    def _is_stage15_text_pair(label_B_or_BLT: Tuple[Any, ...]) -> bool:
        return (
            isinstance(label_B_or_BLT, tuple)
            and len(label_B_or_BLT) == 9
            and label_B_or_BLT[-1] == STAGE15_TEXT_PAIR_MARKER
        )

    def _project_stage15_text_conditions(
        self,
        label_B_or_BLT: Tuple[torch.Tensor, ...],
    ) -> Tuple[torch.Tensor, List[int], torch.Tensor, int, Any, torch.Tensor, List[int]]:
        if isinstance(label_B_or_BLT, tuple) and len(label_B_or_BLT) == 9 and label_B_or_BLT[-1] != STAGE15_TEXT_PAIR_MARKER:
            raise ValueError(f'Invalid Stage1.5 text pair marker: {label_B_or_BLT[-1]!r}')
        if not self._is_stage15_text_pair(label_B_or_BLT):
            kv_compact, lens, cu_seqlens_k, max_seqlen_k, caption_nums = self._project_text_conditions(label_B_or_BLT)
            return kv_compact, lens, cu_seqlens_k, max_seqlen_k, caption_nums, kv_compact, list(lens)

        (
            edited_compact,
            main_lens,
            main_cu_seqlens,
            main_max_seqlen,
            caption_nums,
            edited_lens,
            reprompt_compact,
            reprompt_lens,
            marker,
        ) = label_B_or_BLT
        if marker != STAGE15_TEXT_PAIR_MARKER:
            raise ValueError(f'Invalid Stage1.5 text pair marker: {marker!r}')
        main_lens = [int(item) for item in main_lens]
        edited_lens = [int(item) for item in edited_lens]
        reprompt_lens = [int(item) for item in reprompt_lens]
        if not (len(main_lens) == len(edited_lens) == len(reprompt_lens)):
            raise ValueError(
                'Stage1.5 text pair lens count mismatch: '
                f'main={len(main_lens)}, edited={len(edited_lens)}, reprompt={len(reprompt_lens)}'
            )
        if any(length <= 0 for length in edited_lens):
            raise ValueError(f'Stage1.5 edited text lens must be positive: {edited_lens}')
        for idx, (main_len, edited_len, reprompt_len) in enumerate(zip(main_lens, edited_lens, reprompt_lens)):
            expected = edited_len + (STAGE15_TEXT_SEPARATOR_TOKENS if reprompt_len > 0 else 0) + reprompt_len
            if main_len != expected:
                raise ValueError(
                    f'Stage1.5 main text length mismatch at sample {idx}: '
                    f'{main_len} != {edited_len} + sep({reprompt_len > 0}) + {reprompt_len}'
                )
        if sum(edited_lens) != int(edited_compact.shape[0]):
            raise ValueError(f'Stage1.5 edited compact length mismatch: {sum(edited_lens)} != {edited_compact.shape[0]}')
        if sum(reprompt_lens) != int(reprompt_compact.shape[0]):
            raise ValueError(
                f'Stage1.5 reprompt compact length mismatch: {sum(reprompt_lens)} != {reprompt_compact.shape[0]}'
            )

        with torch.amp.autocast('cuda', dtype=torch.float32):
            edited_projected = self.text_proj(edited_compact).contiguous()
            if int(reprompt_compact.shape[0]) > 0:
                reprompt_projected = self.text_proj(reprompt_compact).contiguous()
            else:
                reprompt_projected = edited_projected.new_zeros((0, edited_projected.shape[-1]))

        edited_ptr = 0
        reprompt_ptr = 0
        main_pieces = []
        gate_pieces = []
        for sample_idx, (edited_len, reprompt_len, main_len) in enumerate(zip(edited_lens, reprompt_lens, main_lens)):
            edited_tokens = edited_projected[edited_ptr:edited_ptr + edited_len]
            gate_pieces.append(edited_tokens)
            sample_pieces = [edited_tokens]
            if reprompt_len > 0:
                sample_pieces.append(edited_tokens.new_zeros((STAGE15_TEXT_SEPARATOR_TOKENS, edited_tokens.shape[-1])))
                sample_pieces.append(reprompt_projected[reprompt_ptr:reprompt_ptr + reprompt_len])
            sample_main = torch.cat(sample_pieces, dim=0)
            if int(sample_main.shape[0]) != main_len:
                raise ValueError(
                    f'Stage1.5 packed main text length mismatch at sample {sample_idx}: '
                    f'{sample_main.shape[0]} != {main_len}'
                )
            main_pieces.append(sample_main)
            edited_ptr += edited_len
            reprompt_ptr += reprompt_len
        if edited_ptr != edited_projected.shape[0] or reprompt_ptr != reprompt_projected.shape[0]:
            raise ValueError(
                'Stage1.5 text pair pointers do not consume compact tensors: '
                f'edited={edited_ptr}/{edited_projected.shape[0]}, reprompt={reprompt_ptr}/{reprompt_projected.shape[0]}'
            )
        return (
            torch.cat(main_pieces, dim=0),
            main_lens,
            main_cu_seqlens,
            int(main_max_seqlen),
            caption_nums,
            torch.cat(gate_pieces, dim=0),
            edited_lens,
        )

    def prepare_stage15_text_conditions(
        self,
        label_B_or_BLT: Tuple[torch.Tensor, ...],
    ) -> Tuple[torch.Tensor, List[int], torch.Tensor, List[int]]:
        kv_compact, lens, _, _, _, gate_compact, gate_lens = self._project_stage15_text_conditions(label_B_or_BLT)
        return kv_compact, lens, gate_compact, gate_lens

    def _check_source_visual_inputs(
        self,
        x_BLC: List[torch.Tensor],
        source_x_BLC: Optional[List[torch.Tensor]],
        other_info_by_scale: Optional[List[Dict[str, Any]]],
    ) -> List[int]:
        if source_x_BLC is None:
            raise ValueError('Stage15LayerwiseSourceConditionedGRN requires source_x_BLC.')
        if len(source_x_BLC) != len(x_BLC):
            raise ValueError(f'source_x_BLC length mismatch: {len(source_x_BLC)} != {len(x_BLC)}')
        sub_L_list = [item.shape[1] for item in x_BLC]
        source_sub_L_list = [item.shape[1] for item in source_x_BLC]
        if source_sub_L_list != sub_L_list:
            raise ValueError(f'source visual token lengths do not match input: {source_sub_L_list} != {sub_L_list}')
        if other_info_by_scale is None or len(other_info_by_scale) != len(x_BLC):
            raise ValueError(
                'Stage15LayerwiseSourceConditionedGRN requires one other_info_by_scale entry per visual span: '
                f'{0 if other_info_by_scale is None else len(other_info_by_scale)} != {len(x_BLC)}'
            )
        return sub_L_list

    def _embed_entry_visual_with_source(
        self,
        cat_x_BLC: torch.Tensor,
        cat_source_x_BLC: torch.Tensor,
        sub_L_list: List[int],
        text_context: torch.Tensor,
        pt_context: torch.Tensor,
    ) -> torch.Tensor:
        if tuple(cat_source_x_BLC.shape) != tuple(cat_x_BLC.shape):
            raise ValueError(
                'source_x_BLC shape mismatch: '
                f'source={tuple(cat_source_x_BLC.shape)}, input={tuple(cat_x_BLC.shape)}'
            )
        with torch.amp.autocast('cuda', dtype=torch.float32):
            cat_x_BLC = self.word_embed(cat_x_BLC.float())
        if not self._stage15_source_injection_enabled(0):
            return cat_x_BLC
        with torch.amp.autocast('cuda', dtype=torch.float32):
            source_hidden = self.source_word_embed(cat_source_x_BLC.float())
        x_splits = torch.split(cat_x_BLC, sub_L_list, dim=1)
        source_splits = torch.split(source_hidden, sub_L_list, dim=1)
        if len(x_splits) != text_context.shape[0] or len(x_splits) != pt_context.shape[0]:
            raise ValueError(
                'entry source residual context count mismatch: '
                f'visual_spans={len(x_splits)}, text={text_context.shape[0]}, pt={pt_context.shape[0]}'
            )
        pieces = []
        for visual_x, visual_source, text_ctx, pt_ctx in zip(x_splits, source_splits, text_context, pt_context):
            source_residual = self.source_residual_modulations[0](
                visual_source,
                text_ctx[None],
                pt_ctx[None],
            )
            pieces.append(visual_x + source_residual)
        cat_x_BLC = torch.cat(pieces, dim=1)
        return cat_x_BLC

    def _make_source_full_sequence(
        self,
        source_embed: nn.Linear,
        modulation: SourceResidualTextPtModulation,
        cat_source_x_BLC: torch.Tensor,
        sub_L_list: List[int],
        lens: List[int],
        text_context: torch.Tensor,
        pt_context: torch.Tensor,
    ) -> torch.Tensor:
        with torch.amp.autocast('cuda', dtype=torch.float32):
            source_hidden = source_embed(cat_source_x_BLC.float())
        source_hidden_splits = torch.split(source_hidden, sub_L_list, dim=1)
        if len(source_hidden_splits) != len(lens) or len(source_hidden_splits) != text_context.shape[0] or len(source_hidden_splits) != pt_context.shape[0]:
            raise ValueError(
                'deep source residual context count mismatch: '
                f'visual_spans={len(source_hidden_splits)}, lens={len(lens)}, '
                f'text={text_context.shape[0]}, pt={pt_context.shape[0]}'
            )
        pieces = []
        for visual_source, text_len, text_ctx, pt_ctx in zip(source_hidden_splits, lens, text_context, pt_context):
            source_residual = modulation(visual_source, text_ctx[None], pt_ctx[None])
            pieces.append(source_residual)
            pieces.append(visual_source.new_zeros((visual_source.shape[0], int(text_len), visual_source.shape[-1])))
            pieces.append(visual_source.new_zeros((visual_source.shape[0], 1, visual_source.shape[-1])))
        return torch.cat(pieces, dim=1)

    def _inject_deep_source(
        self,
        x_BLC: torch.Tensor,
        cat_source_x_BLC: torch.Tensor,
        sub_L_list: List[int],
        lens: List[int],
        text_context: torch.Tensor,
        pt_context: torch.Tensor,
        deep_index: int,
        sp_is_on: bool,
    ) -> torch.Tensor:
        if not self._stage15_source_injection_enabled(deep_index + 1):
            return x_BLC
        source_full = self._make_source_full_sequence(
            self.deep_source_word_embeds[deep_index],
            self.source_residual_modulations[deep_index + 1],
            cat_source_x_BLC,
            sub_L_list,
            lens,
            text_context,
            pt_context,
        )
        if sp_is_on:
            source_full = sp_split_sequence_by_dim(source_full, 1)
        if tuple(source_full.shape) != tuple(x_BLC.shape):
            raise ValueError(f'Stage1.5 source full sequence shape mismatch: {tuple(source_full.shape)} != {tuple(x_BLC.shape)}')
        return x_BLC + source_full

    def _labels_to_visual_tokens(self, labels: torch.Tensor, classes: int) -> torch.Tensor:
        visual_token_dim = labels.shape[1] * classes
        return multiclass_labels2onehot_input(labels, classes).reshape(labels.shape[0], visual_token_dim, -1).permute(0, 2, 1)

    def _embed_labels_with_entry_source(
        self,
        labels: torch.Tensor,
        source_labels: torch.Tensor,
        classes: int,
        text_context: torch.Tensor,
        pt_context: torch.Tensor,
    ) -> torch.Tensor:
        if tuple(labels.shape) != tuple(source_labels.shape):
            raise ValueError(
                'Stage1.5 source label shape mismatch: '
                f'labels={tuple(labels.shape)}, source={tuple(source_labels.shape)}'
            )
        visual_tokens = self._labels_to_visual_tokens(labels, classes)
        source_tokens = self._labels_to_visual_tokens(source_labels, classes)
        return self._embed_entry_visual_with_source(
            visual_tokens,
            source_tokens,
            [visual_tokens.shape[1]],
            text_context,
            pt_context,
        )

    def _make_infer_source_full_sequence(
        self,
        source_embed: nn.Linear,
        modulation: SourceResidualTextPtModulation,
        source_labels: torch.Tensor,
        classes: int,
        lens: List[int],
        text_context: torch.Tensor,
        pt_context: torch.Tensor,
    ) -> torch.Tensor:
        source_tokens = self._labels_to_visual_tokens(source_labels, classes)
        with torch.amp.autocast('cuda', dtype=torch.float32):
            source_hidden = source_embed(source_tokens.float())
        cond_source = modulation(source_hidden, text_context[0:1], pt_context[0:1])
        pieces = [
            cond_source,
            source_hidden.new_zeros((source_hidden.shape[0], int(lens[0]), source_hidden.shape[-1])),
            source_hidden.new_zeros((source_hidden.shape[0], 1, source_hidden.shape[-1])),
        ]
        return torch.cat(pieces, dim=1)

    def _run_stage15_block_chunks(
        self,
        x_BLC: torch.Tensor,
        cat_source_x_BLC: torch.Tensor,
        sub_L_list: List[int],
        lens: List[int],
        text_context: torch.Tensor,
        pt_context: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        e0: Optional[torch.Tensor],
        attn_bias_or_two_vector: Optional[Any],
        checkpointing_full_block: bool,
        rope_cache: torch.Tensor,
    ) -> torch.Tensor:
        sp_is_on = sp_manager.sp_on()
        if sp_is_on:
            x_BLC = sp_split_sequence_by_dim(x_BLC, 1)
        for chunk_index, chunk in enumerate(self.block_chunks):
            if chunk_index > 0 and self._stage15_source_injection_enabled(chunk_index):
                x_BLC = self._inject_deep_source(
                    x_BLC=x_BLC,
                    cat_source_x_BLC=cat_source_x_BLC,
                    sub_L_list=sub_L_list,
                    lens=lens,
                    text_context=text_context,
                    pt_context=pt_context,
                    deep_index=chunk_index - 1,
                    sp_is_on=sp_is_on,
                )
            x_BLC = chunk(
                x=x_BLC,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                e0=e0,
                attn_bias_or_two_vector=attn_bias_or_two_vector,
                checkpointing_full_block=checkpointing_full_block,
                rope2d_freqs_grid=rope_cache,
            )
        if sp_is_on:
            x_BLC = sp_gather_sequence_by_dim(x_BLC, 1)
        return x_BLC

    def _forward_stage15_hidden(
        self,
        label_B_or_BLT,
        x_BLC: List[torch.Tensor],
        source_x_BLC: Optional[List[torch.Tensor]] = None,
        visual_rope_cache: Optional[List[torch.Tensor]] = None,
        sequece_packing_scales: Optional[List[List[Tuple[int, int, int]]]] = None,
        super_scale_lengths: Optional[List[int]] = None,
        other_info_by_scale: Optional[List[Dict[str, Any]]] = None,
        x_BLC_mask: Optional[torch.Tensor] = None,
        scale_or_time_ids: Optional[torch.Tensor] = None,
        disable_checkpointing: bool = False,
    ) -> Dict[str, Any]:
        sub_L_list = self._check_source_visual_inputs(x_BLC, source_x_BLC, other_info_by_scale)
        device = x_BLC[0].device
        cat_x_BLC = torch.cat(x_BLC, dim=1)
        cat_source_x_BLC = torch.cat(source_x_BLC, dim=1)

        kv_compact, lens, cu_seqlens_k, max_seqlen_k, _, gate_compact, gate_lens = (
            self._project_stage15_text_conditions(label_B_or_BLT)
        )
        kv_compact_splits = torch.split(kv_compact, lens, dim=0)
        gate_compact_splits = torch.split(gate_compact, gate_lens, dim=0)

        scale_token_ids = torch.tensor([info["scale_token_id"] for info in other_info_by_scale], device=device)
        with torch.amp.autocast("cuda", dtype=torch.float32):
            pt_tokens = self.pt_embedder(scale_token_ids)
        text_context, pt_context = self._build_source_contexts(gate_compact_splits, pt_tokens)

        cat_x_BLC = self._embed_entry_visual_with_source(
            cat_x_BLC,
            cat_source_x_BLC,
            sub_L_list,
            text_context,
            pt_context,
        )
        x_BLC = list(torch.split(cat_x_BLC, sub_L_list, dim=1))

        x_BLC_lists = []
        visual_spans = []
        cursor = 0
        for i in range(len(x_BLC)):
            visual_len = int(x_BLC[i].shape[1])
            visual_spans.append((cursor, cursor + visual_len))
            x_BLC_lists.extend([x_BLC[i], kv_compact_splits[i].unsqueeze(0), pt_tokens[i][None, None]])
            cursor += visual_len + int(lens[i]) + int(self.other_args.add_scale_token)
        x_BLC = torch.cat(x_BLC_lists, dim=1)

        valid_sequence_ratio = x_BLC.shape[1] / self.other_args.train_max_token_len
        attn_bias_or_two_vector = None

        self.rope2d_freqs_grid['freqs_text'] = self.rope2d_freqs_grid['freqs_text'].to(x_BLC.device)
        rope_cache_list = []
        scale_token_rope_offset = get_scale_token_rope_offset(self.other_args)
        for i in range(len(visual_rope_cache)):
            rope_cache_list.append(visual_rope_cache[i])
            rope_cache_list.append(self.rope2d_freqs_grid['freqs_text'][:, :, :, :, :lens[i]])
            rope_cache_list.append(
                self.rope2d_freqs_grid['freqs_text'][
                    :, :, :, :, scale_token_rope_offset:scale_token_rope_offset + self.other_args.add_scale_token
                ]
            )
        rope_cache = torch.cat(rope_cache_list, dim=4)
        assert rope_cache.shape[4] == x_BLC.shape[1], f'{rope_cache.shape[4]} != {x_BLC.shape[1]}'
        rope_cache = rope_cache[:, 0].permute(0, 1, 3, 2, 4)

        if self.other_args.use_ada_layer_norm:
            with torch.amp.autocast('cuda', dtype=torch.float32):
                e = self.scale_or_time_embedding(
                    sinusoidal_embedding_1d(self.scale_or_time_dim, scale_or_time_ids).float()
                )
                if e.shape[1] < x_BLC.shape[1]:
                    e = F.pad(e, (0, 0, 0, x_BLC.shape[1] - e.shape[1]), 'constant', 0.)
                e0 = self.scale_or_time_projection(e).unflatten(2, (6, self.C))
                assert e.dtype == torch.float32 and e0.dtype == torch.float32
        else:
            e, e0 = None, None

        checkpointing_full_block = self.checkpointing == 'full-block' and self.training and not disable_checkpointing
        cu_seqlens = torch.tensor([0] + super_scale_lengths, device=device).cumsum(-1).to(torch.int32)
        max_seqlen = max(super_scale_lengths)
        x_BLC = self._run_stage15_block_chunks(
            x_BLC=x_BLC,
            cat_source_x_BLC=cat_source_x_BLC,
            sub_L_list=sub_L_list,
            lens=lens,
            text_context=text_context,
            pt_context=pt_context,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            e0=e0,
            attn_bias_or_two_vector=attn_bias_or_two_vector,
            checkpointing_full_block=checkpointing_full_block,
            rope_cache=rope_cache,
        )

        return {
            'hidden_states': x_BLC,
            'e': e,
            'valid_sequence_ratio': valid_sequence_ratio,
            'visual_spans': visual_spans,
            'visual_hidden_by_scale': [x_BLC[:, start:end] for start, end in visual_spans],
            'text_tokens_by_scale': list(kv_compact_splits),
            'edited_text_tokens_by_scale': list(gate_compact_splits),
            'x_BLC_mask': x_BLC_mask,
        }

    def forward(
        self,
        label_B_or_BLT,
        x_BLC: List[torch.Tensor],
        source_x_BLC: Optional[List[torch.Tensor]] = None,
        visual_rope_cache: Optional[List[torch.Tensor]] = None,
        sequece_packing_scales: Optional[List[List[Tuple[int, int, int]]]] = None,
        super_scale_lengths: Optional[List[int]] = None,
        other_info_by_scale: Optional[List[Dict[str, Any]]] = None,
        gt_BL: Optional[List[torch.Tensor]] = None,
        x_BLC_mask: Optional[torch.Tensor] = None,
        scale_or_time_ids: Optional[torch.Tensor] = None,
        return_last_hidden_states: bool = False,
        **kwargs: Any,
    ):
        hidden_data = self._forward_stage15_hidden(
            label_B_or_BLT=label_B_or_BLT,
            x_BLC=x_BLC,
            source_x_BLC=source_x_BLC,
            visual_rope_cache=visual_rope_cache,
            sequece_packing_scales=sequece_packing_scales,
            super_scale_lengths=super_scale_lengths,
            other_info_by_scale=other_info_by_scale,
            x_BLC_mask=x_BLC_mask,
            scale_or_time_ids=scale_or_time_ids,
        )

        logits_norm, loss_list, acc_list = self.get_loss_acc(
            hidden_data['hidden_states'],
            x_BLC_mask,
            hidden_data['e'],
            sequece_packing_scales,
            gt_BL,
            other_info_by_scale,
            return_last_hidden_states,
        )
        return logits_norm, loss_list, acc_list, hidden_data['valid_sequence_ratio']

    @torch.no_grad()
    def autoregressive_infer(
        self,
        vae: Optional[Any] = None,
        scale_schedule: Optional[List[Tuple[int, int, int]]] = None,
        label_B_or_BLT: Optional[List[Tuple[torch.Tensor, ...]]] = None,
        tau_list: Optional[List[float]] = None,
        args: Optional[Any] = None,
        get_visual_rope_embeds: Optional[Any] = None,
        stage15_source_labels: Optional[torch.Tensor] = None,
    ):
        """Official autoregressive inference with Stage1.5 layerwise source residuals."""
        if stage15_source_labels is None:
            raise ValueError('Stage1.5 inference requires stage15_source_labels.')
        if tau_list is None:
            tau_list = []
        from grn.schedules.global_refine import shift_pt

        assert len(tau_list) >= len(scale_schedule), "Not enough tau values for scales"

        for b in self.unregistered_blocks:
            b.attn.kv_caching(True)
        total_steps = args.max_infer_steps
        pbar = tqdm.tqdm(total=total_steps)
        block_chunks = self.block_chunks
        full_pt, ph, pw = scale_schedule[0]
        pt = full_pt
        visual_rope_cache = get_visual_rope_embeds(
            self.rope2d_freqs_grid,
            (pt, ph, pw),
            'cuda',
            args.mapped_h_div_w_template,
            t_offset=0,
        )

        self.rope2d_freqs_grid['freqs_text'] = self.rope2d_freqs_grid['freqs_text'].to('cuda')
        prefix_tokens, lens, gate_tokens, gate_lens = self.prepare_stage15_text_conditions(label_B_or_BLT[0])
        device = prefix_tokens.device
        infer_device, infer_dtype = prefix_tokens.device, prefix_tokens.dtype
        prefix_tokens = torch.split(prefix_tokens, lens, dim=0)
        gate_tokens = torch.split(gate_tokens, gate_lens, dim=0)
        text_context = torch.stack([self._mean_text_context(tokens) for tokens in gate_tokens], dim=0)
        rope_cache_text_cond = self.rope2d_freqs_grid['freqs_text'][:, :, :, :, :lens[0]]

        if args.refine_mode != 'ar_discrete_GRN_bit':
            raise ValueError(f'Stage1.5 inference only supports ar_discrete_GRN_bit, got {args.refine_mode!r}.')
        classes = 2
        labels_shape = (1, args.detail_scale_dim * args.hbq_round, pt, ph, pw)

        stage15_source_labels = stage15_source_labels.to(device=infer_device)
        if tuple(stage15_source_labels.shape) != tuple(labels_shape):
            raise ValueError(
                'stage15_source_labels shape mismatch: '
                f'{tuple(stage15_source_labels.shape)} != {tuple(labels_shape)}'
            )
        stage15_source_labels = stage15_source_labels.to(dtype=infer_dtype)

        mul_pt_ph_pw = pt * ph * pw
        repeat_idx = -1
        scale_token_rope_offset = get_scale_token_rope_offset(args)
        scale_token_rope_cache = self.rope2d_freqs_grid[
            'freqs_text'
        ][:, :, :, :, scale_token_rope_offset:scale_token_rope_offset + args.add_scale_token]
        assert len(scale_schedule) == 1
        tmp_seqlens = [mul_pt_ph_pw + lens[0] + args.add_scale_token]

        rope_cache_parts = [visual_rope_cache, rope_cache_text_cond, scale_token_rope_cache]
        rope_cache = torch.cat(rope_cache_parts, dim=4)
        rope_cache = rope_cache[:, 0].permute(0, 1, 3, 2, 4)

        cu_seqlens = torch.tensor([0] + tmp_seqlens, device=device).cumsum(-1).to(torch.int32)
        max_seqlen = max(tmp_seqlens)

        pure_rand_labels = torch.randint(low=0, high=classes, size=labels_shape, device=infer_device, dtype=infer_dtype)
        mixed_xt = pure_rand_labels.clone()
        next_pt = 0.
        attn_mask = build_attn_mask(tmp_seqlens, device) if args.use_slow_attn else None
        for cur_inner_round_si in range(args.max_infer_steps):
            cur_pt = next_pt
            is_last_step = np.abs(cur_pt - 1) < 0.02
            if cur_inner_round_si == 0:
                self.entrophy_statistics.append([])
            repeat_idx += 1
            pt_tokens = self.pt_embedder(torch.tensor([cur_pt], device=device)).unsqueeze(0)
            pt_context = pt_tokens[:, 0, :].expand(text_context.shape[0], -1)
            last_stage = self._embed_labels_with_entry_source(
                mixed_xt,
                stage15_source_labels,
                classes,
                text_context[0:1],
                pt_context[0:1],
            )
            last_stage = torch.cat((last_stage, prefix_tokens[0].unsqueeze(0), pt_tokens), dim=1)

            e, e0 = None, None
            for block_idx, b in enumerate(block_chunks):
                if block_idx > 0 and self._stage15_source_injection_enabled(block_idx):
                    source_full = self._make_infer_source_full_sequence(
                        self.deep_source_word_embeds[block_idx - 1],
                        self.source_residual_modulations[block_idx],
                        stage15_source_labels,
                        classes,
                        lens,
                        text_context,
                        pt_context,
                    )
                    if tuple(source_full.shape) != tuple(last_stage.shape):
                        raise ValueError(
                            'Stage1.5 inference source full sequence shape mismatch: '
                            f'{tuple(source_full.shape)} != {tuple(last_stage.shape)}'
                        )
                    last_stage = last_stage + source_full
                last_stage = b(
                    x=last_stage,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen,
                    e0=e0,
                    attn_bias_or_two_vector=attn_mask,
                    rope2d_freqs_grid=rope_cache,
                )
            logits = self.get_logits_during_infer(last_stage, e=e)
            tmp_bs, tmp_seq_len = logits.shape[:2]
            logits = logits.reshape(tmp_bs, tmp_seq_len, -1, args.detail_num_lvl)
            pred_cond_logits = logits[:, :mul_pt_ph_pw]
            pred_cond_probs = pred_cond_logits.softmax(-1)
            categories = pred_cond_logits.shape[-1]
            entrophy = (-pred_cond_probs * torch.log2(pred_cond_probs)).sum(-1).mean().item() / np.log2(categories)

            pt_unshift = (cur_inner_round_si + 1) / (args.complexity_aware_Tmax - 1)
            pt_shift = shift_pt(min(1., pt_unshift), args.snr_shift)
            next_pt = 1 - np.cos(np.pi / 2 * pt_shift)
            next_pt = next_pt * 0.999

            pred_cond_labels = torch.argmax(pred_cond_probs, dim=-1)
            pred_cond_labels = bld_to_bthwd(pred_cond_labels, pt, ph, pw)
            pred_logits = pred_cond_logits.mul(1 / tau_list[0])
            pred_probs = pred_logits.softmax(dim=-1)
            pred_labels = torch.argmax(pred_probs, dim=-1)
            pred_labels = bld_to_bthwd(pred_labels, pt, ph, pw)
            pred_sample_labels = torch.multinomial(
                pred_probs.view(-1, args.detail_num_lvl),
                num_samples=1,
                replacement=True,
            ).view(tmp_bs, mul_pt_ph_pw, -1)
            pred_sample_probs = torch.gather(
                pred_probs,
                dim=3,
                index=pred_sample_labels.unsqueeze(-1),
            ).squeeze(-1)
            pred_sample_probs = bld_to_bthwd(pred_sample_probs, pt, ph, pw)
            pred_sample_labels = bld_to_bthwd(pred_sample_labels, pt, ph, pw)

            assume_flip_ratio = (1 - cur_pt) / args.detail_num_lvl * 100.
            pred_zero_ratio = (pred_cond_labels == 0).sum() / pred_cond_labels.numel() * 100.
            pred_one_ratio = (pred_cond_labels == 1).sum() / pred_cond_labels.numel() * 100.
            mixed_xt_Bthwd_01 = mixed_xt.clone().permute(0, 2, 3, 4, 1)
            mixed_xt_Bthwd_01[mixed_xt_Bthwd_01 < 0] = 0
            pred_cond_flip_ratio = (pred_cond_labels != mixed_xt_Bthwd_01).sum() / pred_cond_labels.numel() * 100.
            pred_flip_ratio = (pred_labels != mixed_xt_Bthwd_01).sum() / pred_labels.numel() * 100.
            pred_sample_flip_ratio = (pred_sample_labels != mixed_xt_Bthwd_01).sum() / pred_sample_labels.numel() * 100.
            self.entrophy_statistics[-1].append({
                'cur_inner_round_si': cur_inner_round_si,
                'cur_pt': cur_pt,
                'entrophy': entrophy,
                'assume_flip_ratio': assume_flip_ratio,
                'pred_cond_flip_ratio': pred_cond_flip_ratio.item(),
                'pred_flip_ratio': pred_flip_ratio.item(),
                'pred_sample_flip_ratio': pred_sample_flip_ratio.item(),
                'pred_zero_ratio': pred_zero_ratio.item(),
                'pred_one_ratio': pred_one_ratio.item(),
                'meta': args.meta,
            })
            print(f'{repeat_idx=} {cur_inner_round_si=} {cur_pt=:.3f} {pred_sample_labels.shape=}')
            print(f'{assume_flip_ratio=:.2f}% {pred_cond_flip_ratio=:.2f}% {pred_flip_ratio=:.2f}% {pred_sample_flip_ratio=:.2f}%')
            pred_sample_labels = pred_sample_labels.permute(0, 4, 1, 2, 3)
            pred_sample_probs = pred_sample_probs.permute(0, 4, 1, 2, 3)
            use_predict_mask = torch.rand(pred_sample_labels.shape, device=device) < next_pt
            mixed_xt = torch.where(use_predict_mask, pred_sample_labels, pure_rand_labels)
            next_pt = use_predict_mask.float().mean().item()
            pbar.update(1)
            if is_last_step:
                break

        from grn.utils_t2iv.hbq_util_t2iv import bit_label2raw_feature
        approx_signal = bit_label2raw_feature(pred_sample_labels, hbq_round=args.hbq_round)
        for b in self.unregistered_blocks:
            b.attn.kv_caching(False)
        img = self.summed_codes2images(vae, approx_signal)
        return img


@register_model
def GRN2bOfficialEditStage15(
    depth: int = 28,
    block_chunks: int = 7,
    embed_dim: int = 2304,
    num_heads: int = 18,
    num_key_value_heads: int = 18,
    drop_path_rate: float = 0.0,
    **kwargs: Any,
) -> Stage15LayerwiseSourceConditionedGRN:
    return Stage15LayerwiseSourceConditionedGRN(
        arch='qwen',
        qwen_qkvo_bias=False,
        depth=depth,
        block_chunks=block_chunks,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_key_value_heads=num_key_value_heads,
        mlp_ratio=3.55,
        drop_path_rate=drop_path_rate,
        **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS},
    )
