"""Inference-safe resource discovery and checkpoint fingerprint validation."""

from __future__ import annotations

import glob
import hashlib
import os.path as osp
from typing import Dict, Tuple


EXTERNAL_RESOURCE_FINGERPRINT_FIELDS = (
    "vae_sha256",
    "t5_encoder_sha256",
    "t5_tokenizer_sha256",
)


def resolve_t5_paths(t5_path: str) -> Tuple[str, str]:
    candidates = [
        (osp.join(t5_path, "models_t5_umt5-xxl-enc-bf16.pth"), osp.join(t5_path, "umt5-xxl")),
        (
            osp.join(t5_path, "umt5-xxl", "models_t5_umt5-xxl-enc-bf16.pth"),
            osp.join(t5_path, "umt5-xxl"),
        ),
    ]
    for checkpoint_path, tokenizer_path in candidates:
        if osp.isfile(checkpoint_path) and osp.isdir(tokenizer_path):
            return checkpoint_path, tokenizer_path
    expected = "; ".join(
        f"ckpt={checkpoint}, tokenizer={tokenizer}"
        for checkpoint, tokenizer in candidates
    )
    raise FileNotFoundError(
        f"Unable to locate the UMT5 checkpoint and tokenizer. Expected: {expected}"
    )


def _sha256_file(path: str) -> str:
    if not osp.isfile(path):
        raise FileNotFoundError(f"Required model resource does not exist: {path}")
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_directory(path: str) -> str:
    if not osp.isdir(path):
        raise FileNotFoundError(f"Required tokenizer directory does not exist: {path}")
    files = sorted(
        file_path
        for file_path in glob.glob(osp.join(path, "**", "*"), recursive=True)
        if osp.isfile(file_path)
    )
    if not files:
        raise FileNotFoundError(f"Tokenizer directory contains no files: {path}")
    digest = hashlib.sha256()
    for file_path in files:
        relative = osp.relpath(file_path, path).replace("\\", "/")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256_file(file_path).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def compute_external_resource_fingerprints(args) -> Dict[str, str]:
    t5_checkpoint, tokenizer_path = resolve_t5_paths(str(args.t5_path))
    return {
        "vae_sha256": _sha256_file(osp.abspath(str(args.vae_path))),
        "t5_encoder_sha256": _sha256_file(osp.abspath(t5_checkpoint)),
        "t5_tokenizer_sha256": _sha256_directory(osp.abspath(tokenizer_path)),
    }


def validate_external_resource_fingerprints(saved_args: dict, args, context: str) -> None:
    missing = [
        field for field in EXTERNAL_RESOURCE_FINGERPRINT_FIELDS
        if field not in saved_args
    ]
    if missing:
        if len(missing) == len(EXTERNAL_RESOURCE_FINGERPRINT_FIELDS):
            print(
                f"[GRNEdit compatibility] {context} has no external-resource fingerprints; "
                "VAE/T5 identity cannot be verified.",
                flush=True,
            )
            return
        raise RuntimeError(
            f"{context} has an incomplete external-resource fingerprint manifest: missing={missing}."
        )
    mismatches = [
        f"{field}: runtime={getattr(args, field)!r}, checkpoint={saved_args[field]!r}"
        for field in EXTERNAL_RESOURCE_FINGERPRINT_FIELDS
        if str(getattr(args, field)) != str(saved_args[field])
    ]
    if mismatches:
        raise RuntimeError(
            f"{context} uses different VAE/T5 resources:\n- " + "\n- ".join(mismatches)
        )
