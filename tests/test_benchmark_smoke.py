"""Smoke test: benchmark service returns a metrics object with the four fields.

Uses a fake service so no model or GPU is needed.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from benchmark import BenchmarkMetrics, BenchmarkService  # noqa: E402
from benchmark.metrics import compute_tokens_per_second  # noqa: E402


class FakeResult:
    generated_text = "benchmarking works"
    model_id = "fake-model"
    prompt_tokens = 1
    generated_tokens = 2
    latency_ms = 12.34
    ttft_ms = 8.9


class FakeService:
    model_id = "fake-model"

    def generate(self, prompt, **kwargs):
        time.sleep(0.05)  # simulate inference work
        return FakeResult()


def test_benchmark_returns_metrics_object():
    service = BenchmarkService()
    metrics = service.run(FakeService(), "Explain Artificial Intelligence")

    assert isinstance(metrics, BenchmarkMetrics), type(metrics)
    assert isinstance(metrics.timestamp, str) and metrics.timestamp  # ISO-8601
    assert metrics.latency_ms >= 40, metrics.latency_ms  # real measurement, not the fake's attr
    assert metrics.memory_mb > 0, metrics.memory_mb
    assert metrics.cpu_percent >= 0, metrics.cpu_percent

    d = metrics.to_dict()
    assert set(d) == {
        "timestamp",
        "latency_ms",
        "memory_mb",
        "cpu_percent",
        "generated_tokens",
        "tokens_per_second",
        "ttft_ms",
        "inference_latency_ms",  # surfaced from the service via extra
    }, d
    assert d["inference_latency_ms"] == 12.34  # the fake's measured value
    # Token metrics come from the result object (2 generated tokens) and the
    # tokens/sec rate reuses the same latency the record persists (12.34 ms).
    assert d["generated_tokens"] == 2
    assert d["tokens_per_second"] == round(2 / (12.34 / 1000), 2), d
    # TTFT is surfaced separately from total latency and rounded to 2dp.
    assert d["ttft_ms"] == 8.9
    assert d["latency_ms"] >= 40, "TTFT and total latency stay separate"
    print("PASS:", d)


def test_compute_tokens_per_second_formula():
    # 100 tokens over 2000 ms == 2 s -> 50 tokens/sec.
    assert compute_tokens_per_second(100, 2000.0) == 50.0
    # Round-trip consistency: tokens_per_second * latency / 1000 == tokens.
    tps = compute_tokens_per_second(37, 812.4)
    assert round(tps * 812.4 / 1000, 2) == 37.0, tps
    # Throughput is undefined for a zero/negative inference time.
    assert compute_tokens_per_second(10, 0) is None
    assert compute_tokens_per_second(10, -5.0) is None
    print("PASS: compute_tokens_per_second ->", tps, "(zero-latency -> None)")


def test_benchmark_rejects_empty_prompt():
    service = BenchmarkService()
    try:
        service.run(FakeService(), "   ")
    except ValueError:
        print("PASS: empty prompt rejected")
        return
    raise AssertionError("expected ValueError for empty prompt")


if __name__ == "__main__":
    test_benchmark_returns_metrics_object()
    test_compute_tokens_per_second_formula()
    test_benchmark_rejects_empty_prompt()
