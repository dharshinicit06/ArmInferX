"""Model storage-footprint comparison for ArmInferX (STEP 11, Phase 3).

Compares the on-disk footprint of the FP16 GGUF shards against the Q4_K_M
GGUF file. This is a **storage/model-footprint comparison only** — it is
explicitly never described as a performance speedup: the FP16 baseline cannot
run inference on this 7.63 GiB RAM machine, so no latency/throughput numbers
exist for it (see ``optimization.report``).

The math is pure and deterministic: given file paths, sizes are read from
``stat()`` and the absolute + percentage reduction is computed. Tests exercise
it with small synthetic files.
"""

from __future__ import annotations

from pathlib import Path

#: Label for the FP16 model: explicitly marks it as a reference whose
#: inference is not feasible on the current hardware.
FP16_LABEL = "Reference FP16 model — inference infeasible on current 7.63 GiB RAM machine"

#: Label for the Q4_K_M model: validated optimized model with completed CPU
#: inference.
Q4_LABEL = "Validated optimized model — CPU inference successfully completed"

#: Explicit framing: footprint reduction is a storage fact, not a speedup.
FOOTPRINT_NOTE = (
    "Storage/footprint comparison only. It does not imply any inference "
    "speedup: the FP16 baseline completed no inference on this machine, so "
    "no latency/throughput comparison between FP16 and Q4_K_M is made."
)


def _size_of(path: Path) -> int:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"model file not found: {path}")
    return path.stat().st_size


def file_sizes(paths: list[str | Path]) -> list[dict]:
    """Return ``[{path, name, size_bytes}]`` for each existing file."""
    return [
        {"path": str(Path(p)), "name": Path(p).name, "size_bytes": _size_of(p)}
        for p in paths
    ]


def total_size(paths: list[str | Path]) -> int:
    """Sum of the on-disk sizes of the given files."""
    return sum(_size_of(p) for p in paths)


def compute_footprint(
    fp16_files: list[str | Path],
    q4_files: list[str | Path],
) -> dict:
    """Compare the storage footprint of the FP16 shards vs the Q4_K_M file.

    Args:
        fp16_files: The FP16 GGUF shard paths (e.g. both ``-00001-of-00002``
            and ``-00002-of-00002`` parts).
        q4_files: The Q4_K_M GGUF file path(s).

    Returns:
        A dict with per-model file listings, total bytes, the absolute and
        percentage reduction, and the explicit labels + framing note.

    Raises:
        FileNotFoundError: Any listed file does not exist.
        ValueError: No files provided for either side.
    """
    if not fp16_files:
        raise ValueError("fp16_files must contain at least one path")
    if not q4_files:
        raise ValueError("q4_files must contain at least one path")

    fp16_total = total_size(fp16_files)
    q4_total = total_size(q4_files)
    if fp16_total <= 0:
        raise ValueError("fp16 total size must be positive")

    reduction_bytes = fp16_total - q4_total
    reduction_percent = 100.0 * (1.0 - q4_total / fp16_total)

    return {
        "fp16": {
            "label": FP16_LABEL,
            "files": file_sizes(fp16_files),
            "shard_count": len(fp16_files),
            "total_bytes": fp16_total,
            "total_mb": round(fp16_total / (1024.0 * 1024.0), 2),
        },
        "q4_k_m": {
            "label": Q4_LABEL,
            "files": file_sizes(q4_files),
            "shard_count": len(q4_files),
            "total_bytes": q4_total,
            "total_mb": round(q4_total / (1024.0 * 1024.0), 2),
        },
        "reduction": {
            "bytes": reduction_bytes,
            "mb": round(reduction_bytes / (1024.0 * 1024.0), 2),
            "percent": round(reduction_percent, 2),
        },
        "note": FOOTPRINT_NOTE,
    }
