"""Minimal source-conditioned extension of the vendored official GRN backbone."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator, Sequence

import torch
import torch.nn as nn

from grn.utils_t2iv.sequence_parallel import SequenceParallelManager as sp_manager
from grn.utils_t2iv.sequence_parallel import sp_split_sequence_by_dim

from .source_conditioning import StageLoRASourceBranch, StageLoRASourceTextGate
from .topology import parse_chunk_mask
from .backbone import GRN, MultipleLayers


_OFFICIAL_BACKBONE_CONFIGS = {
    "GRN2b": {
        "depth": 28,
        "block_chunks": 7,
        "embed_dim": 2304,
        "num_heads": 18,
        "num_key_value_heads": 18,
        "mlp_ratio": 3.55,
    }
}


class StageLoRAGRN(GRN):
    """Official GRN plus edit-text-conditioned source residual branches."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._stage_lora_source_residuals: dict[int, torch.Tensor] | None = None

    @property
    def source_conditioning_installed(self) -> bool:
        return hasattr(self, "source_word_embeds") and hasattr(self, "source_text_gate")

    def install_source_conditioning(
        self,
        source_chunk_mask: str | Sequence[int | bool],
        *,
        text_gate_rank: int,
    ) -> None:
        """Create only active branches, copying the already-loaded official word embedding."""

        if self.source_conditioning_installed:
            raise RuntimeError("StageLoRA source conditioning is already installed")
        mask, active = parse_chunk_mask(source_chunk_mask, len(self.block_chunks))
        self.source_word_embeds = nn.ModuleDict(
            {
                str(chunk_id): StageLoRASourceBranch(self.word_embed)
                for chunk_id in active
            }
        )
        self.source_text_gate = StageLoRASourceTextGate(
            text_channels=int(self.text_proj.in_features),
            hidden_channels=int(self.word_embed.out_features),
            rank=int(text_gate_rank),
            active_chunk_ids=active,
            device=self.word_embed.weight.device,
        )
        self.stage_lora_source_chunk_mask = mask
        self.active_chunk_ids = active

    def _require_source_topology(self) -> None:
        if not self.source_conditioning_installed:
            raise RuntimeError(
                "StageLoRA source conditioning must be installed after strict base loading"
            )

    @staticmethod
    def _validate_source_scales(
        source_x_BLC: Sequence[torch.Tensor],
        target_x_BLC: Sequence[torch.Tensor],
    ) -> tuple[torch.Tensor, ...]:
        source = tuple(source_x_BLC)
        target = tuple(target_x_BLC)
        if not source or len(source) != len(target):
            raise ValueError(
                "StageLoRA source/target scale count mismatch: "
                f"source={len(source)}, target={len(target)}"
            )
        for index, (source_scale, target_scale) in enumerate(zip(source, target)):
            if source_scale.ndim != 3 or target_scale.ndim != 3:
                raise ValueError(
                    f"StageLoRA visual scale {index} must use [B,L,C] tensors"
                )
            if tuple(source_scale.shape) != tuple(target_scale.shape):
                raise ValueError(
                    f"StageLoRA source/target visual scale {index} is not aligned: "
                    f"source={tuple(source_scale.shape)}, target={tuple(target_scale.shape)}"
                )
        return source

    @staticmethod
    def _pack_source_streams(
        embedded_scales: Sequence[torch.Tensor],
        text_lens: Sequence[int],
        stream_lengths: Sequence[int],
    ) -> torch.Tensor:
        if not (len(embedded_scales) == len(text_lens) == len(stream_lengths)):
            raise ValueError(
                "StageLoRA packed source metadata count mismatch: "
                f"visual={len(embedded_scales)}, text={len(text_lens)}, streams={len(stream_lengths)}"
            )
        packed = []
        for index, (visual, text_len, stream_len) in enumerate(
            zip(embedded_scales, text_lens, stream_lengths)
        ):
            text_len = int(text_len)
            stream_len = int(stream_len)
            pt_len = stream_len - visual.shape[1] - text_len
            if text_len < 0 or pt_len != 1:
                raise ValueError(
                    f"StageLoRA stream {index} does not match official [visual,text,one PT] layout: "
                    f"stream={stream_len}, visual={visual.shape[1]}, text={text_len}, pt={pt_len}"
                )
            zeros = visual.new_zeros(
                (visual.shape[0], text_len + pt_len, visual.shape[2])
            )
            packed.append(torch.cat((visual, zeros), dim=1))
        result = torch.cat(packed, dim=1)
        if sp_manager.sp_on():
            result = sp_split_sequence_by_dim(result, 1)
        return result

    def _training_source_residuals(
        self,
        source_scales: Sequence[torch.Tensor],
        edit_text_context: torch.Tensor,
        text_lens: Sequence[int],
        stream_lengths: Sequence[int],
    ) -> dict[int, torch.Tensor]:
        text_gate_deltas = self.source_text_gate(edit_text_context)
        residuals = {
            chunk_id: self._pack_source_streams(
                self.source_word_embeds[str(chunk_id)](
                    source_scales,
                    text_gate_deltas[chunk_id],
                ),
                text_lens,
                stream_lengths,
            )
            for chunk_id in self.active_chunk_ids
        }
        return residuals

    def _inference_source_residuals(
        self,
        source_scale: torch.Tensor,
        edit_text_context: torch.Tensor,
        text_lens: Sequence[int],
    ) -> dict[int, torch.Tensor]:
        text_lens = tuple(int(value) for value in text_lens)
        if edit_text_context.ndim != 2 or edit_text_context.shape[0] != len(text_lens):
            raise ValueError(
                "StageLoRA inference needs one source-text context per packed CFG stream: "
                f"contexts={tuple(edit_text_context.shape)}, text_lens={text_lens}"
            )
        visual_len = int(source_scale.shape[1])
        stream_lengths = tuple(visual_len + text_len + 1 for text_len in text_lens)
        source_streams = tuple(source_scale for _ in text_lens)
        text_gate_deltas = self.source_text_gate(edit_text_context)
        residuals = {}
        for chunk_id in self.active_chunk_ids:
            embedded = self.source_word_embeds[str(chunk_id)](
                source_streams,
                text_gate_deltas[chunk_id],
            )
            residuals[chunk_id] = self._pack_source_streams(
                embedded,
                text_lens,
                stream_lengths,
            )
        return residuals

    @contextmanager
    def _use_source_residuals(
        self, residuals: dict[int, torch.Tensor]
    ) -> Iterator[None]:
        if self._stage_lora_source_residuals is not None:
            raise RuntimeError(
                "nested or concurrent StageLoRA source contexts are not supported"
            )
        if tuple(residuals) != self.active_chunk_ids:
            raise RuntimeError(
                "StageLoRA source context does not match active chunk topology"
            )
        self._stage_lora_source_residuals = residuals
        try:
            yield
        finally:
            self._stage_lora_source_residuals = None

    def before_block_chunk(self, chunk_id: int, hidden: torch.Tensor) -> torch.Tensor:
        if (
            not self.source_conditioning_installed
            or chunk_id not in self.active_chunk_ids
        ):
            return hidden
        if self._stage_lora_source_residuals is None:
            raise RuntimeError(
                f"active StageLoRA chunk {chunk_id} was called without source evidence"
            )
        residual = self._stage_lora_source_residuals[chunk_id]
        if tuple(residual.shape) != tuple(hidden.shape):
            raise RuntimeError(
                f"StageLoRA source residual shape mismatch at chunk {chunk_id}: "
                f"source={tuple(residual.shape)}, hidden={tuple(hidden.shape)}"
            )
        residual = residual.to(device=hidden.device, dtype=hidden.dtype)
        return hidden + residual

    def forward(
        self,
        label_B_or_BLT: Any,
        x_BLC: Sequence[torch.Tensor],
        *,
        source_x_BLC: Sequence[torch.Tensor],
        edit_text_context: torch.Tensor,
        super_scale_lengths: Sequence[int],
        **kwargs: Any,
    ) -> Any:
        self._require_source_topology()
        source_scales = self._validate_source_scales(source_x_BLC, x_BLC)
        text_lens = tuple(int(value) for value in label_B_or_BLT[1])
        residuals = self._training_source_residuals(
            source_scales,
            edit_text_context,
            text_lens,
            super_scale_lengths,
        )
        with self._use_source_residuals(residuals):
            return super().forward(
                label_B_or_BLT,
                x_BLC,
                super_scale_lengths=list(super_scale_lengths),
                **kwargs,
            )

    @torch.no_grad()
    def autoregressive_infer(
        self,
        *,
        source_x_BLC: Sequence[torch.Tensor] | torch.Tensor,
        edit_text_context: torch.Tensor,
        negative_edit_text_context: torch.Tensor | None = None,
        label_B_or_BLT: Any,
        negative_label_B_or_BLT: Any = None,
        guidance_mode: str = "none",
        **kwargs: Any,
    ) -> Any:
        self._require_source_topology()
        source_scales = (
            (source_x_BLC,)
            if isinstance(source_x_BLC, torch.Tensor)
            else tuple(source_x_BLC)
        )
        if (
            len(source_scales) != 1
            or source_scales[0].ndim != 3
            or source_scales[0].shape[0] != 1
        ):
            raise ValueError("Stage-LoRA inference needs one [1,L,C] source scale")
        if len(label_B_or_BLT) != 1 or len(label_B_or_BLT[0][1]) != 1:
            raise ValueError("Stage-LoRA inference needs one positive text stream")
        text_lens = tuple(int(length) for length in label_B_or_BLT[0][1])
        contexts = edit_text_context
        if guidance_mode == "standard":
            if negative_label_B_or_BLT is None or negative_edit_text_context is None:
                raise ValueError(
                    "CFG requires a negative text condition and source-text context"
                )
            text_lens += tuple(int(length) for length in negative_label_B_or_BLT[1])
            contexts = torch.cat((contexts, negative_edit_text_context), dim=0)
        elif guidance_mode != "none":
            raise ValueError(
                "guidance_mode must be none or standard; CFG-1 supplies the positive source-text context to standard CFG"
            )
        residuals = self._inference_source_residuals(
            source_scales[0], contexts, text_lens
        )
        with self._use_source_residuals(residuals):
            return super().autoregressive_infer(
                label_B_or_BLT=label_B_or_BLT,
                negative_label_B_or_BLT=negative_label_B_or_BLT,
                guidance_mode=guidance_mode,
                **kwargs,
            )


def create_stage_lora_backbone(model_name: str, **kwargs: Any) -> StageLoRAGRN:
    """Construct the Python-effective latest official architecture without timm registration."""

    if model_name not in _OFFICIAL_BACKBONE_CONFIGS:
        raise ValueError(
            f"unsupported official StageLoRA backbone {model_name!r}; "
            f"expected one of {tuple(_OFFICIAL_BACKBONE_CONFIGS)}"
        )
    config = {**_OFFICIAL_BACKBONE_CONFIGS[model_name], **kwargs}
    return StageLoRAGRN(arch="qwen", qwen_qkvo_bias=False, **config)


__all__ = [
    "MultipleLayers",
    "StageLoRAGRN",
    "StageLoRASourceBranch",
    "StageLoRASourceTextGate",
    "create_stage_lora_backbone",
]
