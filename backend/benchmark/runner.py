"""Reusable benchmark runner for ArmInferX.

``BenchmarkRunner`` executes the **same** :class:`~benchmark.config.BenchmarkConfig`
against any :class:`~engines.base.InferenceEngine` (Transformers baseline or
llama.cpp optimized) so the benchmark procedure is identical for both engines
and future Arm64 measurements are scientifically comparable.

Procedure per engine:
1. Engine identity is read once via ``engine.get_model_info()``.
2. ``config.warmup`` untimed generation calls (same prompt + kwargs).
3. ``config.repeats`` timed calls via the existing ``BenchmarkService.measure``
   (wall-clock latency, memory, CPU, TTFT, token counts, tokens/sec).
4. Every result is tagged with engine identity (engine_id/runtime/model_id,
   plus model_path from ``EngineInfo.extra`` when present).
5. Records are persisted via an engine-aware ``EngineResultStore``
   (``results/benchmarks/<engine_id>/``), leaving the baseline store untouched.
6. Aggregate statistics (mean/median/p90 latency, mean TTFT, mean tokens,
   mean tokens/sec, peak memory, mean CPU) are computed over the runs.

Generation kwargs policy
------------------------
Only arguments supported by BOTH engines are passed: ``max_new_tokens`` and
(only when set) ``temperature``. ``do_sample`` is never passed. The
chat-template policy is applied explicitly for the Transformers baseline via
``use_chat_template=config.chat_template`` (default ``False`` = raw
completion); llama.cpp never receives ``use_chat_template`` or ``do_sample``.

No performance claims are made here. Actual Arm64 measurements are performed
later on an Arm64 cloud environment.
"""

from __future__ import annotations

import logging
import math
import statistics
from dataclasses import dataclass, field
from typing import Any

from benchmark.benchmark_service import BenchmarkService
from benchmark.config import BenchmarkConfig
from benchmark.metrics import BenchmarkMetrics
from benchmark.storage import EngineResultStore, ResultWriteError
from engines import InferenceEngine

logger = logging.getLogger(__name__)

#: Engine ids whose ``generate()`` accepts the ``use_chat_template`` kwarg.
#: Only the Transformers baseline supports chat templates today; llama.cpp
#: performs raw completion and must never receive template kwargs.
CHAT_TEMPLATE_ENGINE_IDS = ("transformers-baseline",)


def _p90(values: list[float]) -> float:
    """90th percentile using the nearest-rank method (sorted ascending)."""
    if not values:
        raise ValueError("cannot compute p90 of an empty list")
    ordered = sorted(values)
    rank = math.ceil(0.9 * len(ordered))
    return ordered[max(0, rank - 1)]


@dataclass(frozen=True)
class BenchmarkAggregates:
    """Aggregate statistics over one benchmark run's timed repetitions."""

    runs: int
    mean_latency_ms: float
    median_latency_ms: float
    p90_latency_ms: float
    mean_ttft_ms: float | None
    mean_generated_tokens: float
    mean_tokens_per_second: float | None
    peak_memory_mb: float
    mean_cpu_percent: float

    def to_dict(self) -> dict:
        return {
            "runs": self.runs,
            "mean_latency_ms": round(self.mean_latency_ms, 2),
            "median_latency_ms": round(self.median_latency_ms, 2),
            "p90_latency_ms": round(self.p90_latency_ms, 2),
            "mean_ttft_ms": (
                round(self.mean_ttft_ms, 2) if self.mean_ttft_ms is not None else None
            ),
            "mean_generated_tokens": round(self.mean_generated_tokens, 2),
            "mean_tokens_per_second": (
                round(self.mean_tokens_per_second, 2)
                if self.mean_tokens_per_second is not None
                else None
            ),
            "peak_memory_mb": round(self.peak_memory_mb, 2),
            "mean_cpu_percent": round(self.mean_cpu_percent, 2),
        }


@dataclass(frozen=True)
class BenchmarkRun:
    """Structured result of one engine benchmark run."""

    engine_id: str
    runtime: str
    model_id: str | None
    model_path: str | None
    config: BenchmarkConfig
    metrics: list[BenchmarkMetrics] = field(default_factory=list)
    records: list[dict] = field(default_factory=list)
    aggregates: BenchmarkAggregates | None = None

    def to_dict(self) -> dict:
        return {
            "engine_id": self.engine_id,
            "runtime": self.runtime,
            "model_id": self.model_id,
            "model_path": self.model_path,
            "config": {
                "prompt": self.config.prompt,
                "max_new_tokens": self.config.max_new_tokens,
                "temperature": self.config.temperature,
                "chat_template": self.config.chat_template,
                "repeats": self.config.repeats,
                "warmup": self.config.warmup,
            },
            "records": self.records,
            "aggregates": self.aggregates.to_dict() if self.aggregates else None,
        }


class BenchmarkRunner:
    """Runs the same benchmark procedure against any ``InferenceEngine``."""

    def __init__(
        self,
        benchmark: BenchmarkService | None = None,
        store: EngineResultStore | None = None,
    ) -> None:
        self._benchmark = benchmark or BenchmarkService()
        # Engine store is created per engine at run() time when not provided.
        self._store = store

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self, engine: InferenceEngine, config: BenchmarkConfig) -> BenchmarkRun:
        """Execute ``config`` against ``engine`` and return a ``BenchmarkRun``.

        Args:
            engine: Any engine implementing the ``InferenceEngine`` interface.
            config: The identical benchmark procedure to apply.
        """
        info = engine.get_model_info()
        engine_id = info.engine_id
        runtime = info.runtime

        gen_kwargs = self.generation_kwargs(engine_id, config)

        # --- Warmup (untimed, same prompt + kwargs) ------------------------
        for _ in range(config.warmup):
            engine.generate(config.prompt, **gen_kwargs)
        logger.info(
            "Benchmark warmup complete for %s (%d untimed call(s))",
            engine_id,
            config.warmup,
        )

        # --- Timed repetitions via the existing BenchmarkService ------------
        metrics_list: list[BenchmarkMetrics] = []
        records: list[dict] = []
        store = self._store or EngineResultStore(engine_id)

        for i in range(config.repeats):
            result, metrics = self._benchmark.measure(
                lambda: engine.generate(config.prompt, **gen_kwargs)
            )
            tagged = self._tag_metrics(metrics, info)
            record = self._build_record(config, result, tagged)
            metrics_list.append(tagged)
            records.append(record)
            try:
                store.save(record)
            except ResultWriteError:
                logger.exception("Failed to persist benchmark record to %s", store.root_dir)
            logger.info(
                "Benchmark repeat %d/%d done for %s", i + 1, config.repeats, engine_id
            )

        return BenchmarkRun(
            engine_id=engine_id,
            runtime=runtime,
            model_id=info.model_id,
            model_path=info.extra.get("model_path") if info.extra else None,
            config=config,
            metrics=metrics_list,
            records=records,
            aggregates=self._aggregate(records),
        )

    def generation_kwargs(self, engine_id: str, config: BenchmarkConfig) -> dict:
        """Build the generation kwargs for one engine from a shared config.

        Only kwargs supported by BOTH engines are produced (``max_new_tokens``
        and, when set, ``temperature``). ``do_sample`` is never included.
        ``use_chat_template`` is included ONLY for chat-template-capable
        engines (the Transformers baseline), controlled explicitly by
        ``config.chat_template``.
        """
        kwargs: dict[str, Any] = {"max_new_tokens": config.max_new_tokens}
        if config.temperature is not None:
            kwargs["temperature"] = config.temperature
        if engine_id in CHAT_TEMPLATE_ENGINE_IDS:
            kwargs["use_chat_template"] = config.chat_template
        return kwargs

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    @staticmethod
    def _tag_metrics(metrics: BenchmarkMetrics, info) -> BenchmarkMetrics:
        """Return a copy of ``metrics`` carrying engine identity in ``extra``."""
        extra = dict(metrics.extra)
        extra["engine_id"] = info.engine_id
        extra["runtime"] = info.runtime
        extra["model_id"] = info.model_id
        if info.extra and info.extra.get("model_path"):
            extra["model_path"] = info.extra["model_path"]
        return BenchmarkMetrics(
            timestamp=metrics.timestamp,
            latency_ms=metrics.latency_ms,
            memory_mb=metrics.memory_mb,
            cpu_percent=metrics.cpu_percent,
            generated_tokens=metrics.generated_tokens,
            tokens_per_second=metrics.tokens_per_second,
            ttft_ms=metrics.ttft_ms,
            extra=extra,
        )

    @staticmethod
    def _build_record(config: BenchmarkConfig, result, metrics: BenchmarkMetrics) -> dict:
        """Build the persisted record (mirrors the HTTP /generate record)."""
        return {
            "prompt": config.prompt,
            "model": result.model_id,
            "response": result.generated_text,
            "latency_ms": result.latency_ms,
            "ttft_ms": metrics.ttft_ms,
            "memory_mb": metrics.memory_mb,
            "cpu_percent": metrics.cpu_percent,
            "generated_tokens": metrics.generated_tokens,
            "tokens_per_second": metrics.tokens_per_second,
            "timestamp": metrics.timestamp,
            "engine_id": metrics.extra.get("engine_id"),
            "runtime": metrics.extra.get("runtime"),
            "model_path": metrics.extra.get("model_path"),
        }

    @staticmethod
    def _aggregate(records: list[dict]) -> BenchmarkAggregates:
        """Aggregate mean/median/p90 + token/memory/CPU stats over records.

        Latency statistics use each record's ``latency_ms`` (the engine's own
        inference latency, the same value the summary endpoint averages), so
        aggregates stay consistent with the persisted records.
        """
        latencies = [float(r["latency_ms"]) for r in records]
        ttfts = [float(r["ttft_ms"]) for r in records if r.get("ttft_ms") is not None]
        tokens = [float(r["generated_tokens"]) for r in records if r.get("generated_tokens") is not None]
        tps = [float(r["tokens_per_second"]) for r in records if r.get("tokens_per_second") is not None]
        memory = [float(r["memory_mb"]) for r in records]
        cpu = [float(r["cpu_percent"]) for r in records]

        return BenchmarkAggregates(
            runs=len(records),
            mean_latency_ms=statistics.mean(latencies),
            median_latency_ms=statistics.median(latencies),
            p90_latency_ms=_p90(latencies),
            mean_ttft_ms=statistics.mean(ttfts) if ttfts else None,
            mean_generated_tokens=statistics.mean(tokens) if tokens else 0.0,
            mean_tokens_per_second=statistics.mean(tps) if tps else None,
            peak_memory_mb=max(memory) if memory else 0.0,
            mean_cpu_percent=statistics.mean(cpu) if cpu else 0.0,
        )
