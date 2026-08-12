"""Optimization evidence layer for ArmInferX (STEP 11).

Provides deterministic, machine-readable evidence that:

- ``gguf_metadata`` — parses the actual GGUF file bytes (header, metadata KV,
  tensor table, SHA-256) without loading any model.
- ``model_footprint`` — computes the FP16 vs Q4_K_M on-disk footprint
  reduction (storage-only; never a speedup claim).
- ``report`` — assembles the machine-readable optimization report with an
  explicit measured / not-measured / hardware-limitation classification.

The reproducible Q4_K_M benchmark itself is produced by the existing
``benchmark.BenchmarkRunner`` / ``BenchmarkService`` — this layer only reads
its results and combines them with model facts. No benchmark methodology is
reimplemented here.
"""

from optimization.gguf_metadata import (
    GGUFMetadataError,
    analyze_gguf,
    file_type_name,
    sha256_file,
    tensor_type_name,
)
from optimization.model_footprint import (
    compute_footprint,
    file_sizes,
    total_size,
)
from optimization.report import (
    NOT_MEASURED_EXPLANATION,
    build_optimization_report,
    utc_now_iso,
)

__all__ = [
    "GGUFMetadataError",
    "NOT_MEASURED_EXPLANATION",
    "analyze_gguf",
    "build_optimization_report",
    "compute_footprint",
    "file_sizes",
    "file_type_name",
    "sha256_file",
    "tensor_type_name",
    "total_size",
    "utc_now_iso",
]
