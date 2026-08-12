"""Benchmark metrics data model and system samplers for ArmInferX.

Defines the benchmark result object (returned as a plain Python object by the
benchmark service) and the samplers that fill in process-level memory and CPU
readings.

Modularity: latency lives in ``api.utils.timing.Stopwatch``; memory/CPU
samplers live here; orchestration lives in ``benchmark.benchmark_service``.
New metrics (tokens/sec, throughput, TTFT) plug in as new typed fields on
``BenchmarkMetrics`` or via ``extra`` without touching the sampling flow.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import psutil


def utc_now_iso() -> str:
    """ISO-8601 UTC timestamp for benchmark records."""
    return datetime.now(timezone.utc).isoformat()


def compute_tokens_per_second(
    generated_tokens: int, latency_ms: float
) -> float | None:
    """Derive generation throughput: ``generated_tokens / inference_time_seconds``.

    ``latency_ms`` must be the same inference time that is persisted as the
    record's ``latency_ms`` (not the whole-request wall clock), so the stored
    record stays internally consistent:
    ``tokens_per_second * latency_ms / 1000 == generated_tokens``.

    Args:
        generated_tokens: Number of generated output tokens (tokenizer-native
            count from the model's output token IDs).
        latency_ms: Inference time in milliseconds.

    Returns:
        Throughput in tokens per second, rounded to 2 decimals, or ``None``
        when ``latency_ms`` is not positive (throughput is undefined for a
        zero or negative inference time).
    """
    if latency_ms <= 0:
        return None
    return round(generated_tokens / (latency_ms / 1000.0), 2)


@dataclass(frozen=True)
class BenchmarkMetrics:
    """Result of a single benchmark run, returned as a Python object.

    Fields:
        timestamp: ISO-8601 UTC wall-clock time of the run.
        latency_ms: wall-clock latency of the measured call in milliseconds.
        memory_mb: process RSS in MB; the peak of the pre/post samples taken
            around the run (not a continuous sample during inference).
        cpu_percent: process CPU usage (%) across the run. psutil reports
            per-process CPU as a fraction of a single core, so this can
            exceed 100 on multi-core machines.
        generated_tokens: number of output tokens the model generated. None
            when the measured callable's result does not report a token count.
        tokens_per_second: ``generated_tokens / inference_time_seconds``.
            The inference time is the result's own latency
            (``result.latency_ms``) when available, else this object's
            measured ``latency_ms`` — i.e. the same latency value persisted as
            the record's ``latency_ms``. None when tokens or a positive
            latency are unavailable.
        ttft_ms: time-to-first-token in milliseconds — the wall-clock time
            from the start of inference until the first generated token is
            produced. Kept separate from ``latency_ms`` (total inference
            time). None when the result does not report it.
        extra: optional dict for future metrics (throughput by phase...).
    """

    timestamp: str
    latency_ms: float
    memory_mb: float
    cpu_percent: float
    generated_tokens: int | None = None
    tokens_per_second: float | None = None
    ttft_ms: float | None = None
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """JSON-serializable representation (logs, future result storage)."""
        return {
            "timestamp": self.timestamp,
            "latency_ms": self.latency_ms,
            "memory_mb": self.memory_mb,
            "cpu_percent": self.cpu_percent,
            "generated_tokens": self.generated_tokens,
            "tokens_per_second": self.tokens_per_second,
            "ttft_ms": self.ttft_ms,
            **self.extra,
        }


class SystemSampler:
    """Process-level memory and CPU sampling backed by psutil.

    Usage:
        sampler = SystemSampler()          # establishes the CPU baseline
        sampler.memory_mb()                # current RSS in MB
        sampler.cpu_percent()              # % CPU since the previous call

    The first ``cpu_percent()`` call only establishes a baseline (psutil
    reports a delta since the last call), so the benchmark service calls it
    once before the timed run and once after.
    """

    def __init__(self) -> None:
        self._process = psutil.Process()
        # First call initializes psutil's internal CPU baseline (returns 0.0).
        self._process.cpu_percent(interval=None)

    def memory_mb(self) -> float:
        """Current resident set size (RSS) of this process in MB."""
        rss_bytes = self._process.memory_info().rss
        return rss_bytes / (1024.0 * 1024.0)

    def cpu_percent(self) -> float:
        """Process CPU usage (%) since the previous call (non-blocking)."""
        return self._process.cpu_percent(interval=None)
