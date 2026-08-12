"""Benchmark orchestration for ArmInferX.

Runs a single inference benchmark against any object exposing a
``generate(prompt, **kwargs)`` method (the existing InferenceService today,
ONNX/llama.cpp runtimes later) and returns a ``BenchmarkMetrics`` object.

Measurement split (modular by design):
- Latency: ``api.utils.timing.Stopwatch`` around the measured call only.
- Memory/CPU: ``metrics.SystemSampler`` (psutil).
- Timestamp: ``metrics.utc_now_iso``.
This service only orchestrates; new metrics plug into ``BenchmarkMetrics``
without changing the sampling flow. No optimization is performed.

``measure(call)`` is the general entry point (used by the HTTP layer to
benchmark the exact call it performs); ``run(service, prompt)`` is the
convenience wrapper for one-shot benchmark scripts.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

from api.utils.timing import Stopwatch

from benchmark.logger import get_benchmark_logger
from benchmark.metrics import (
    BenchmarkMetrics,
    SystemSampler,
    compute_tokens_per_second,
    utc_now_iso,
)

logger = get_benchmark_logger()


class Generatable(Protocol):
    """Minimal interface any benchmarkable service must satisfy."""

    def generate(self, prompt: str, **kwargs: Any) -> Any: ...


class BenchmarkService:
    """Orchestrates benchmark runs and returns metric objects."""

    def __init__(self, sampler: SystemSampler | None = None) -> None:
        self._sampler = sampler or SystemSampler()

    def measure(self, call: Callable[[], Any]) -> tuple[Any, BenchmarkMetrics]:
        """Measure latency/memory/CPU around an existing callable.

        Returns the callable's result together with the measured metrics so
        the caller can use both (e.g. the HTTP layer returns the generation
        result and persists the metrics).

        Latency times the whole ``call``; if the callable's result itself
        reports a finer-grained latency (e.g. ``GenerationResult.latency_ms``),
        it is surfaced under ``metrics.extra["inference_latency_ms"]``. When
        the result also reports ``generated_tokens``, throughput
        (``tokens_per_second``) is derived from the same latency value that
        gets persisted as the record's ``latency_ms``. A result-side
        ``ttft_ms`` (time-to-first-token) is surfaced as ``metrics.ttft_ms``,
        kept separate from the total ``latency_ms``.
        """
        logger.info("Benchmark start")
        timestamp = utc_now_iso()

        # --- System baselines (before the timed call) ----------------------
        # Memory: sample RSS before and after; report the peak of the two.
        # CPU: psutil reports a delta since the last call, so the call right
        # before the run establishes the interval we want to measure.
        memory_before = self._sampler.memory_mb()
        self._sampler.cpu_percent()

        # --- Latency: only the measured call is timed -----------------------
        with Stopwatch() as timer:
            result = call()
        latency_ms = timer.latency_ms

        # --- System readings after the call ---------------------------------
        memory_after = self._sampler.memory_mb()
        cpu_percent = self._sampler.cpu_percent()

        # If the result carries its own latency (e.g. the inference service
        # times only model.generate()), keep it instead of losing it.
        extra: dict = {}
        service_latency = getattr(result, "latency_ms", None)
        if service_latency is not None:
            extra["inference_latency_ms"] = round(service_latency, 2)

        # --- Token-level metrics -------------------------------------------
        # Token counting is tokenizer-native: the inference service reports
        # generated_tokens as the count of the output token IDs produced by
        # model.generate() (no character/word estimation). Throughput reuses
        # the same latency value that gets persisted as ``latency_ms`` (the
        # result's own inference latency when available, else the measured
        # wall-clock), so the saved record stays internally consistent:
        # tokens_per_second * latency_ms / 1000 == generated_tokens.
        generated_tokens = getattr(result, "generated_tokens", None)
        inference_time_ms = service_latency if service_latency is not None else latency_ms
        tokens_per_second = None
        if isinstance(generated_tokens, int):
            # The helper guards against non-positive latency (returns None).
            tokens_per_second = compute_tokens_per_second(
                generated_tokens, inference_time_ms
            )

        # --- Time-to-first-token -------------------------------------------
        # Total latency and TTFT stay separate metrics: latency_ms is the
        # whole generation, ttft_ms is only the time until the first output
        # token was produced (measured inside the inference service).
        ttft_ms = getattr(result, "ttft_ms", None)
        if isinstance(ttft_ms, (int, float)):
            ttft_ms = round(float(ttft_ms), 2)
        else:
            ttft_ms = None

        metrics = BenchmarkMetrics(
            timestamp=timestamp,
            latency_ms=round(latency_ms, 2),
            memory_mb=round(max(memory_before, memory_after), 2),
            cpu_percent=round(cpu_percent, 2),
            generated_tokens=generated_tokens,
            tokens_per_second=tokens_per_second,
            ttft_ms=ttft_ms,
            extra=extra,
        )
        logger.info(
            "Benchmark done: latency=%.2fms ttft=%.2fms memory=%.2fMB cpu=%.2f%% "
            "tokens=%s tokens_per_sec=%s",
            metrics.latency_ms,
            metrics.ttft_ms if metrics.ttft_ms is not None else 0.0,
            metrics.memory_mb,
            metrics.cpu_percent,
            metrics.generated_tokens,
            metrics.tokens_per_second,
        )
        return result, metrics

    def run(
        self,
        service: Generatable,
        prompt: str,
        *,
        max_new_tokens: int = 128,
        **generation_kwargs: Any,
    ) -> BenchmarkMetrics:
        """Benchmark one generation and return a ``BenchmarkMetrics`` object.

        Args:
            service: Any object with a ``generate(prompt, **kwargs)`` method.
            prompt: Non-empty prompt to generate from.
            max_new_tokens: Passed to ``service.generate`` to control output
                length (benchmark runs are length-controlled).
            **generation_kwargs: Extra kwargs forwarded to ``generate``.

        Raises:
            ValueError: If ``prompt`` is empty or whitespace-only.
        """
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")

        logger.info(
            "Benchmark start: prompt_len=%d max_new_tokens=%d",
            len(prompt),
            max_new_tokens,
        )
        _, metrics = self.measure(
            lambda: service.generate(
                prompt, max_new_tokens=max_new_tokens, **generation_kwargs
            )
        )
        return metrics
