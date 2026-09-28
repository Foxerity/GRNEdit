"""Resolve complete checkpoints or losslessly split .pth releases to mmap-able files."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
import shutil


def resolve_checkpoint_path(path: str | Path) -> Path:
    """Merge numbered byte parts once, under a process lock, without loading tensors.

    A split directory contains NAME.pth.000.part, ... and a SHA256SUMS entry
    for NAME.pth. Completed files are cached by checksum; inputs are never changed.
    This is transport support, not support for additional model architectures.
    """
    source = Path(path).expanduser().resolve()
    if source.is_file():
        if source.suffix == ".part":
            raise ValueError(
                "Pass the split checkpoint directory, not an individual .part file."
            )
        return source
    if not source.is_dir():
        raise FileNotFoundError(f"Checkpoint does not exist: {source}")

    checksum_file = source / "SHA256SUMS"
    if not checksum_file.is_file():
        raise FileNotFoundError(f"Split checkpoint directory needs {checksum_file}")
    records = [
        line.strip() for line in checksum_file.read_text().splitlines() if line.strip()
    ]
    match = (
        re.fullmatch(r"([0-9a-fA-F]{64})\s+\*?([^/\\]+\.pth)", records[0])
        if len(records) == 1
        else None
    )
    if match is None:
        raise ValueError(
            "SHA256SUMS must contain exactly one SHA-256 and a .pth basename."
        )
    digest, name = match.groups()
    digest = digest.lower()
    pattern = re.compile(re.escape(name) + r"\.(\d{3})\.part")
    parts = sorted(p for p in source.iterdir() if pattern.fullmatch(p.name))
    expected = [f"{name}.{index:03d}.part" for index in range(len(parts))]
    if not parts or [p.name for p in parts] != expected:
        raise ValueError("Checkpoint parts must be contiguous and start at .000.part.")
    sizes = [p.stat().st_size for p in parts]
    if not all(sizes):
        raise ValueError("Checkpoint contains an empty part.")
    total = sum(sizes)
    cache = Path(
        os.environ.get(
            "GRNEDIT_CHECKPOINT_CACHE", str(Path.home() / ".cache/grnedit/checkpoints")
        )
    ).expanduser()
    cache = cache.resolve() / digest
    cache.mkdir(parents=True, exist_ok=True)
    output = cache / name

    # filelock is also a PyTorch dependency. Lock before checking the cache so
    # torchrun workers sharing this directory cannot assemble the same file twice.
    from filelock import FileLock

    with FileLock(str(cache / ".merge.lock")):
        if output.is_file() and output.stat().st_size == total:
            return output
        temporary = cache / (name + ".incomplete")
        # A previous process may have been killed during assembly. Only our
        # incomplete cache file is discarded; source parts are never removed.
        temporary.unlink(missing_ok=True)
        if shutil.disk_usage(cache).free < total:
            raise OSError(
                f"Need {total / 2**30:.2f} GiB free to assemble checkpoint in {cache}"
            )
        actual = hashlib.sha256()
        try:
            with temporary.open("wb") as target:
                for index, part in enumerate(parts, 1):
                    print(
                        f"[checkpoint] assembling part {index}/{len(parts)}: {part.name}",
                        flush=True,
                    )
                    with part.open("rb") as handle:
                        while block := handle.read(8 * 1024 * 1024):
                            actual.update(block)
                            target.write(block)
            if actual.hexdigest() != digest:
                raise ValueError(
                    "Checkpoint checksum mismatch: a part is missing, corrupt, or from another release."
                )
            os.replace(temporary, output)
        finally:
            temporary.unlink(missing_ok=True)
    print(f"[checkpoint] verified and cached: {output}", flush=True)
    return output
