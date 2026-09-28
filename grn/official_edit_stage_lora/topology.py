"""Static StageLoRA parameter topology and strict checkpoint metadata."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, Tuple


OFFICIAL_GRN_COMMIT = "fe71a70e82b9f25303590fcb769212a81ab11b9d"
ADAPTER_NAME = "stage_lora"
STAGE_LORA_TOPOLOGY_FORMAT_VERSION = 3
STAGE_LORA_SOURCE_BRANCH_LAYOUT = (
    "linear_with_static_and_raw_edit_text_mean_rms_low_rank_gate_v1"
)
LORA_SUFFIXES = (
    "attn.q_proj",
    "attn.k_proj",
    "attn.v_proj",
    "attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)
LORA_SCOPES = ("all_blocks", "source_chunks")


def parse_chunk_mask(
    raw_mask: str | Sequence[int | bool], num_chunks: int
) -> Tuple[Tuple[bool, ...], Tuple[int, ...]]:
    """Parse the immutable source topology used by one model instance."""

    if num_chunks <= 0:
        raise ValueError(f"num_chunks must be positive, got {num_chunks}")
    if isinstance(raw_mask, str):
        text = raw_mask.strip()
        if not text:
            raise ValueError("--stage_lora_source_chunk_mask must not be empty")
        if text.startswith("[") or text.endswith("]"):
            if not (text.startswith("[") and text.endswith("]")):
                raise ValueError(
                    "--stage_lora_source_chunk_mask must be a comma-separated list or a JSON list"
                )
            try:
                decoded = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "--stage_lora_source_chunk_mask must be a valid JSON list of 0/1 values"
                ) from exc
            if not isinstance(decoded, list):
                raise ValueError(
                    "--stage_lora_source_chunk_mask JSON form must be a list"
                )
            raw_values = decoded
        else:
            raw_values = [part.strip() for part in text.split(",")]
        if any(value == "" for value in raw_values):
            raise ValueError(
                "--stage_lora_source_chunk_mask must be a comma-separated list of 0/1 values"
            )
        values_list = []
        for value in raw_values:
            if isinstance(value, str):
                if value not in {"0", "1"}:
                    raise ValueError(
                        "--stage_lora_source_chunk_mask accepts only 0 and 1"
                    )
                values_list.append(value == "1")
            elif isinstance(value, (bool, int)) and value in (0, 1, False, True):
                values_list.append(bool(value))
            else:
                raise ValueError("--stage_lora_source_chunk_mask accepts only 0 and 1")
        values = tuple(values_list)
    else:
        raw_values = tuple(raw_mask)
        if not raw_values:
            raise ValueError("stage_lora source chunk mask must not be empty")
        for value in raw_values:
            if not isinstance(value, (bool, int)) or value not in (0, 1, False, True):
                raise ValueError(
                    "stage_lora source chunk mask accepts only bool or integer 0/1 values"
                )
        values = tuple(bool(value) for value in raw_values)

    if len(values) != num_chunks:
        raise ValueError(
            "stage_lora source chunk mask length must match the constructed backbone: "
            f"mask={len(values)}, block_chunks={num_chunks}"
        )
    active = tuple(index for index, enabled in enumerate(values) if enabled)
    if not active:
        raise ValueError(
            "stage_lora source chunk mask must activate at least one chunk"
        )
    return values, active


def build_lora_targets(
    model: Any, scope: str, active_chunk_ids: Iterable[int]
) -> Tuple[str, ...]:
    """Enumerate exact PEFT targets, validating all seven official block projections."""

    import torch.nn as nn

    if scope not in LORA_SCOPES:
        raise ValueError(
            f"unsupported StageLoRA LoRA scope {scope!r}; expected one of {LORA_SCOPES}"
        )
    active = tuple(int(chunk_id) for chunk_id in active_chunk_ids)
    selected = (
        tuple(range(len(model.block_chunks))) if scope == "all_blocks" else active
    )
    if not selected:
        raise ValueError(
            "StageLoRA LoRA target selection must contain at least one chunk"
        )
    if len(set(selected)) != len(selected) or any(
        chunk_id < 0 or chunk_id >= len(model.block_chunks) for chunk_id in selected
    ):
        raise ValueError(f"invalid StageLoRA LoRA chunk ids: {selected}")

    targets = []
    for chunk_id in selected:
        chunk = model.block_chunks[chunk_id]
        for local_block_id, _ in enumerate(chunk.module):
            prefix = f"block_chunks.{chunk_id}.module.{local_block_id}"
            for suffix in LORA_SUFFIXES:
                name = f"{prefix}.{suffix}"
                module = model.get_submodule(name)
                if not isinstance(module, nn.Linear):
                    raise TypeError(
                        f"LoRA target {name!r} is {type(module).__name__}, expected torch.nn.Linear"
                    )
                targets.append(name)

    return tuple(targets)


@dataclass(frozen=True)
class StageLoRATopology:
    source_chunk_mask: Tuple[bool, ...]
    active_chunk_ids: Tuple[int, ...]
    source_text_gate_rank: int
    text_channels: int
    lora_scope: str
    lora_rank: int
    lora_alpha: float
    lora_dropout: float
    lora_targets: Tuple[str, ...]
    adapter_name: str
    base_model: str
    base_checkpoint: str
    official_grn_commit: str
    peft_version: str

    def __post_init__(self) -> None:
        mask, active = parse_chunk_mask(
            self.source_chunk_mask, len(self.source_chunk_mask)
        )
        if mask != self.source_chunk_mask or active != self.active_chunk_ids:
            raise ValueError(
                "active_chunk_ids must be exactly derived from source_chunk_mask"
            )
        if self.source_text_gate_rank <= 0 or self.text_channels <= 0:
            raise ValueError("source text-gate rank and text channels must be positive")
        if self.lora_scope not in LORA_SCOPES:
            raise ValueError(f"invalid lora_scope {self.lora_scope!r}")
        if self.lora_rank <= 0 or self.lora_alpha <= 0:
            raise ValueError("LoRA rank and alpha must be positive")
        if not 0.0 <= self.lora_dropout < 1.0:
            raise ValueError("LoRA dropout must be in [0, 1)")
        if not self.lora_targets or len(set(self.lora_targets)) != len(
            self.lora_targets
        ):
            raise ValueError("lora_targets must be non-empty and unique")
        if self.adapter_name != ADAPTER_NAME:
            raise ValueError(f"StageLoRA adapter name must be {ADAPTER_NAME!r}")
        for field_name in (
            "base_model",
            "base_checkpoint",
            "official_grn_commit",
            "peft_version",
        ):
            if not str(getattr(self, field_name)).strip():
                raise ValueError(
                    f"{field_name} must be recorded in StageLoRA topology metadata"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": STAGE_LORA_TOPOLOGY_FORMAT_VERSION,
            "source_branch_layout": STAGE_LORA_SOURCE_BRANCH_LAYOUT,
            "source_chunk_mask": [int(value) for value in self.source_chunk_mask],
            "active_chunk_ids": list(self.active_chunk_ids),
            "source_text_gate_rank": self.source_text_gate_rank,
            "text_channels": self.text_channels,
            "lora_scope": self.lora_scope,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "lora_targets": list(self.lora_targets),
            "adapter_name": self.adapter_name,
            "base_model": self.base_model,
            "base_checkpoint": self.base_checkpoint,
            "official_grn_commit": self.official_grn_commit,
            "peft_version": self.peft_version,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "StageLoRATopology":
        if int(value.get("format_version", -1)) != STAGE_LORA_TOPOLOGY_FORMAT_VERSION:
            raise ValueError(
                f"unsupported StageLoRA topology format_version={value.get('format_version')!r}"
            )
        if value.get("source_branch_layout") != STAGE_LORA_SOURCE_BRANCH_LAYOUT:
            raise ValueError(
                "unsupported StageLoRA source_branch_layout="
                f"{value.get('source_branch_layout')!r}"
            )
        mask_values = tuple(value["source_chunk_mask"])
        mask, active = parse_chunk_mask(mask_values, len(mask_values))
        recorded_active = tuple(int(item) for item in value["active_chunk_ids"])
        if active != recorded_active:
            raise ValueError(
                "checkpoint active_chunk_ids do not match source_chunk_mask"
            )
        return cls(
            source_chunk_mask=mask,
            active_chunk_ids=active,
            source_text_gate_rank=int(value["source_text_gate_rank"]),
            text_channels=int(value["text_channels"]),
            lora_scope=str(value["lora_scope"]),
            lora_rank=int(value["lora_rank"]),
            lora_alpha=float(value["lora_alpha"]),
            lora_dropout=float(value["lora_dropout"]),
            lora_targets=tuple(str(item) for item in value["lora_targets"]),
            adapter_name=ADAPTER_NAME
            if value["adapter_name"] == "stage1"
            else str(value["adapter_name"]),
            base_model=str(value["base_model"]),
            base_checkpoint=str(value["base_checkpoint"]),
            official_grn_commit=str(value["official_grn_commit"]),
            peft_version=str(value["peft_version"]),
        )

    def assert_matches(self, other: "StageLoRATopology") -> None:
        # File locations and library provenance are not model topology. In
        # particular, a released adapter must remain portable across machines.
        ignored = {"base_checkpoint", "peft_version"}
        actual, expected = self.to_dict(), other.to_dict()
        differing = [
            key
            for key in actual
            if key not in ignored and actual[key] != expected.get(key)
        ]
        if differing:
            raise ValueError(
                f"Stage-LoRA checkpoint topology differs: {', '.join(differing)}"
            )


def save_topology_json(topology: StageLoRATopology, path: str | Path) -> None:
    Path(path).write_text(
        json.dumps(topology.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def load_topology_json(path: str | Path) -> StageLoRATopology:
    return StageLoRATopology.from_dict(
        json.loads(Path(path).read_text(encoding="utf-8"))
    )
