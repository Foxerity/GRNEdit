"""Trainable source projection and edit-text gating for official StageLoRA.

For chunk ``k`` the injected visual residual is
``P_k(source) * (g_k + U_k(SiLU(D(RMS(mean(T5(edit)))))))``.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class StageLoRASourceTextGate(nn.Module):
    """Map one pooled edit instruction to an additive gate per active chunk.

    The shared down projection is normally initialized. Every chunk-specific up
    projection is zero initialized, so the complete dynamic gate is an exact
    no-op at construction while retaining a nonzero first-step gradient for the
    up projections.
    """

    def __init__(
        self,
        *,
        text_channels: int,
        hidden_channels: int,
        rank: int,
        active_chunk_ids: Sequence[int],
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if int(text_channels) <= 0 or int(hidden_channels) <= 0 or int(rank) <= 0:
            raise ValueError("StageLoRA source text-gate dimensions must be positive")
        active = tuple(int(chunk_id) for chunk_id in active_chunk_ids)
        if not active or len(set(active)) != len(active):
            raise ValueError("StageLoRA source text gate requires unique active chunks")

        self.text_channels = int(text_channels)
        self.hidden_channels = int(hidden_channels)
        self.rank = int(rank)
        self.active_chunk_ids = active
        self.down = nn.Linear(
            self.text_channels,
            self.rank,
            bias=False,
            device=device,
            dtype=torch.float32,
        )
        self.up_by_chunk = nn.ModuleDict(
            {
                str(chunk_id): nn.Linear(
                    self.rank,
                    self.hidden_channels,
                    bias=False,
                    device=device,
                    dtype=torch.float32,
                )
                for chunk_id in active
            }
        )
        for up in self.up_by_chunk.values():
            nn.init.zeros_(up.weight)

    def forward(self, edit_text_context: torch.Tensor) -> dict[int, torch.Tensor]:
        """Return ``[streams, hidden]`` additive gates keyed by absolute chunk."""

        if (
            edit_text_context.ndim != 2
            or edit_text_context.shape[1] != self.text_channels
        ):
            raise ValueError(
                "StageLoRA edit-text context must be [streams,text_channels]: "
                f"got={tuple(edit_text_context.shape)}, text_channels={self.text_channels}"
            )
        with torch.autocast(device_type=edit_text_context.device.type, enabled=False):
            context = edit_text_context.detach().float()
            context = context * torch.rsqrt(
                context.square().mean(dim=-1, keepdim=True) + 1e-6
            )
            latent = F.silu(self.down(context))
            return {
                chunk_id: self.up_by_chunk[str(chunk_id)](latent)
                for chunk_id in self.active_chunk_ids
            }


class StageLoRASourceBranch(nn.Linear):
    """Copied source projection with static and per-stream edit-text gates.

    All parameters stay directly owned by the root FSDP unit. The forward path
    therefore reads parameters only through live module attributes and never
    retains a parameter view across forwards.
    """

    def __init__(self, loaded_word_embed: nn.Linear) -> None:
        super().__init__(
            loaded_word_embed.in_features,
            loaded_word_embed.out_features,
            bias=loaded_word_embed.bias is not None,
            device=loaded_word_embed.weight.device,
            dtype=loaded_word_embed.weight.dtype,
        )
        self.gate = nn.Parameter(
            loaded_word_embed.weight.new_zeros(loaded_word_embed.out_features)
        )
        self.reset_from_word_embed(loaded_word_embed)

    def reset_from_word_embed(self, loaded_word_embed: nn.Linear) -> None:
        """Copy the loaded backbone projection and restore a no-op static gate."""

        if tuple(self.weight.shape) != tuple(loaded_word_embed.weight.shape):
            raise ValueError(
                "StageLoRA source/backbone word embedding weight shape mismatch: "
                f"source={tuple(self.weight.shape)}, "
                f"backbone={tuple(loaded_word_embed.weight.shape)}"
            )
        if (self.bias is None) != (loaded_word_embed.bias is None):
            raise ValueError(
                "StageLoRA source/backbone word embedding bias topology mismatch"
            )
        with torch.no_grad():
            self.weight.copy_(loaded_word_embed.weight)
            if self.bias is not None:
                self.bias.copy_(loaded_word_embed.bias)
            self.gate.zero_()

    def forward(
        self,
        source_scales: Sequence[torch.Tensor],
        text_gate_delta: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """Project concatenated source tokens, then gate each packed stream separately."""

        source = tuple(source_scales)
        if not source:
            raise ValueError(
                "StageLoRA source branch requires at least one visual stream"
            )
        if text_gate_delta.ndim != 2 or tuple(text_gate_delta.shape) != (
            len(source),
            self.out_features,
        ):
            raise ValueError(
                "StageLoRA text-gate/source-stream shape mismatch: "
                f"gate={tuple(text_gate_delta.shape)}, "
                f"expected={(len(source), self.out_features)}"
            )

        lengths = [scale.shape[1] for scale in source]
        with torch.autocast(device_type=source[0].device.type, enabled=False):
            projected = super().forward(torch.cat(source, dim=1).float())
            projected_scales = torch.split(projected, lengths, dim=1)
            effective_gate = self.gate.float().unsqueeze(0) + text_gate_delta.float()
            return tuple(
                (visual.float() * effective_gate[index].view(1, 1, -1)).to(
                    dtype=visual.dtype
                )
                for index, visual in enumerate(projected_scales)
            )


__all__ = ["StageLoRASourceBranch", "StageLoRASourceTextGate"]
