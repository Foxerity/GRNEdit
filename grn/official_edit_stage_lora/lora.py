"""PEFT LoRA installation, trainability policy, and compact StageLoRA state."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping, Sequence

import torch

from .topology import (
    ADAPTER_NAME,
    StageLoRATopology,
    load_topology_json,
    save_topology_json,
)


PEFT_VERSION = "0.19.1"


def _peft_api():
    import peft
    from peft import LoraConfig, inject_adapter_in_model
    from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict

    if peft.__version__ != PEFT_VERSION:
        raise RuntimeError(
            f"StageLoRA requires peft=={PEFT_VERSION}, found {peft.__version__}. "
            "Install the project-pinned version before constructing the adapter."
        )
    return (
        LoraConfig,
        inject_adapter_in_model,
        get_peft_model_state_dict,
        set_peft_model_state_dict,
    )


def inject_stage_lora(
    model: torch.nn.Module,
    exact_targets: Sequence[str],
    rank: int,
    alpha: float,
    dropout: float,
    adapter_name: str = ADAPTER_NAME,
) -> torch.nn.Module:
    """Inject vanilla LoRA in place without changing the GRN model type."""

    if adapter_name != ADAPTER_NAME:
        raise ValueError(f"StageLoRA adapter name must be {ADAPTER_NAME!r}")
    if rank <= 0 or alpha <= 0 or not 0.0 <= dropout < 1.0:
        raise ValueError(
            f"invalid LoRA configuration: rank={rank}, alpha={alpha}, dropout={dropout}"
        )
    targets = tuple(exact_targets)
    if not targets or len(set(targets)) != len(targets):
        raise ValueError("exact LoRA targets must be non-empty and unique")

    LoraConfig, inject_adapter_in_model, _, _ = _peft_api()
    config = LoraConfig(
        r=int(rank),
        lora_alpha=float(alpha),
        lora_dropout=float(dropout),
        target_modules=list(targets),
        bias="none",
        init_lora_weights=True,
    )
    result = inject_adapter_in_model(config, model, adapter_name=adapter_name)
    if result is not model:
        raise RuntimeError(
            "PEFT low-level injection unexpectedly replaced the GRN model instance"
        )
    return model


def is_lora_parameter(name: str) -> bool:
    return ".lora_A." in name or ".lora_B." in name


def is_source_parameter(name: str) -> bool:
    return name.startswith(("source_word_embeds.", "source_text_gate."))


@dataclass(frozen=True)
class TrainableParameterStats:
    total: int
    lora: int
    source: int
    trainable: int
    trainable_tensors: int

    def as_dict(self) -> dict[str, int]:
        return {
            "total": self.total,
            "lora": self.lora,
            "source": self.source,
            "trainable": self.trainable,
            "trainable_tensors": self.trainable_tensors,
        }


def configure_stage_lora_trainable_parameters(
    model: torch.nn.Module,
) -> TrainableParameterStats:
    """Freeze the official backbone and enable only LoRA plus source parameters."""

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for name, parameter in model.named_parameters():
        if is_lora_parameter(name) or is_source_parameter(name):
            parameter.requires_grad_(True)

    named = list(model.named_parameters())
    total = sum(parameter.numel() for _, parameter in named)
    lora = sum(
        parameter.numel() for name, parameter in named if is_lora_parameter(name)
    )
    source = sum(
        parameter.numel() for name, parameter in named if is_source_parameter(name)
    )
    trainable_named = [
        (name, parameter) for name, parameter in named if parameter.requires_grad
    ]
    trainable = sum(parameter.numel() for _, parameter in trainable_named)
    return TrainableParameterStats(total, lora, source, trainable, len(trainable_named))


def _source_state_dict(
    model: torch.nn.Module,
    state_dict: Mapping[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    state = model.state_dict() if state_dict is None else state_dict
    selected = {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in state.items()
        if is_source_parameter(name)
    }
    if not selected:
        raise RuntimeError("model has no StageLoRA source branch state")
    return selected


def _canonical_compiled_state_dict(
    state_dict: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    keys = tuple(state_dict)
    has_compiled_prefix = [name.startswith("_orig_mod.") for name in keys]
    if any(has_compiled_prefix) and not all(has_compiled_prefix):
        raise RuntimeError(
            "FSDP full state mixes compiled and uncompiled parameter names"
        )
    if all(has_compiled_prefix):
        return {
            name[len("_orig_mod.") :]: tensor for name, tensor in state_dict.items()
        }
    return dict(state_dict)


def save_compact_stage_lora(
    output_dir: str | Path,
    model: torch.nn.Module,
    topology: StageLoRATopology,
    *,
    full_state_dict: Mapping[str, torch.Tensor] | None = None,
) -> None:
    """Save the unmerged adapter and source branch for publication/inference."""

    from safetensors.torch import save_file

    _, _, get_peft_model_state_dict, _ = _peft_api()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    state_dict = (
        _canonical_compiled_state_dict(full_state_dict)
        if full_state_dict is not None
        else None
    )
    adapter_state = {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in get_peft_model_state_dict(
            model,
            state_dict=state_dict,
            adapter_name=topology.adapter_name,
        ).items()
    }
    if not adapter_state:
        raise RuntimeError("PEFT returned an empty StageLoRA adapter state")
    save_file(adapter_state, str(output / "adapter_model.safetensors"))
    save_file(
        _source_state_dict(model, state_dict), str(output / "source_branch.safetensors")
    )
    # Compact exports are publication artifacts: keep the required base filename,
    # but do not publish the training machine's local directory layout.
    portable_topology = replace(
        topology, base_checkpoint=Path(topology.base_checkpoint).name
    )
    save_topology_json(portable_topology, output / "stage_lora_config.json")


def load_compact_stage_lora(
    output_dir: str | Path,
    model: torch.nn.Module,
    expected_topology: StageLoRATopology,
) -> None:
    """Strictly restore compact state after base load, source install, and LoRA injection."""

    from safetensors.torch import load_file

    _, _, get_peft_model_state_dict, set_peft_model_state_dict = _peft_api()
    output = Path(output_dir)
    config_path = output / "stage_lora_config.json"
    if not config_path.is_file():
        config_path = (
            output / "stage1_config.json"
        )  # Main-repository compact checkpoint.
    recorded = load_topology_json(config_path)
    expected_topology.assert_matches(recorded)

    adapter_state = load_file(str(output / "adapter_model.safetensors"), device="cpu")
    expected_adapter = get_peft_model_state_dict(
        model, adapter_name=expected_topology.adapter_name
    )
    _validate_state_mapping("adapter", adapter_state, expected_adapter)
    load_result = set_peft_model_state_dict(
        model,
        adapter_state,
        adapter_name=expected_topology.adapter_name,
    )
    if getattr(load_result, "unexpected_keys", None):
        raise RuntimeError(
            f"unexpected keys while loading StageLoRA adapter: {load_result.unexpected_keys}"
        )

    source_state = load_file(str(output / "source_branch.safetensors"), device="cpu")
    expected_source = _source_state_dict(model)
    _validate_state_mapping("source branch", source_state, expected_source)
    source_result = model.load_state_dict(source_state, strict=False)
    if source_result.unexpected_keys:
        raise RuntimeError(
            f"unexpected keys while loading StageLoRA source branch: {source_result.unexpected_keys}"
        )


def _validate_state_mapping(
    label: str,
    actual: Mapping[str, torch.Tensor],
    expected: Mapping[str, torch.Tensor],
) -> None:
    actual_keys, expected_keys = set(actual), set(expected)
    if actual_keys != expected_keys:
        raise RuntimeError(
            f"StageLoRA {label} keys do not match topology: "
            f"missing={sorted(expected_keys - actual_keys)[:8]}, unexpected={sorted(actual_keys - expected_keys)[:8]}"
        )
    mismatched = [
        name
        for name in expected_keys
        if tuple(actual[name].shape) != tuple(expected[name].shape)
    ]
    if mismatched:
        raise RuntimeError(
            f"StageLoRA {label} shape mismatch for keys: {mismatched[:8]}"
        )
