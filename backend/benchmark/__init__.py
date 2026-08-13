"""Modular benchmarking subsystem for ArmInferX.

- ``benchmark.metrics``: the ``BenchmarkMetrics`` result object + system
  samplers (process memory, CPU).
- ``benchmark.logger``: dedicated logging for benchmark records.
- ``benchmark.config``: ``BenchmarkConfig`` — the deterministic, engine-agnostic
  benchmark procedure specification (prompt, limits, temperature, warmup,
  repeats, chat-template policy).
- ``benchmark.benchmark_service``: ``BenchmarkService`` orchestration that
  measures a call and returns a ``BenchmarkMetrics`` object.
- ``benchmark.runner``: ``BenchmarkRunner`` — runs the same ``BenchmarkConfig``
  against any ``InferenceEngine`` (llama.cpp today) with warmup, repeated
  measurement, engine identity, and aggregate statistics.
- ``benchmark.storage``: ``BaselineResultStore`` (unchanged) + engine-aware
  ``EngineResultStore`` (``results/benchmarks/<engine_id>/``) persisting
  records as unique JSON files.

Extensibility: new metrics (throughput by phase, ...) plug in later as new
typed fields on ``BenchmarkMetrics`` or via ``BenchmarkMetrics.extra`` without
changing the orchestration or persistence flow.
"""

from benchmark.benchmark_service import BenchmarkService
from benchmark.config import BenchmarkConfig, BenchmarkConfigError
from benchmark.metrics import BenchmarkMetrics, SystemSampler
from benchmark.runner import BenchmarkAggregates, BenchmarkRunner, BenchmarkRun
from benchmark.storage import (
    BaselineResultStore,
    EngineResultStore,
    ResultWriteError,
)

__all__ = [
    "BaselineResultStore",
    "BenchmarkAggregates",
    "BenchmarkConfig",
    "BenchmarkConfigError",
    "BenchmarkMetrics",
    "BenchmarkRun",
    "BenchmarkRunner",
    "BenchmarkService",
    "EngineResultStore",
    "ResultWriteError",
    "SystemSampler",
]
