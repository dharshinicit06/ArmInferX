"""API tests for the benchmark results endpoints.

Seeds a temp results directory with known records, redirects the benchmarks
router's store to it, and verifies GET /benchmarks, /benchmarks/latest and
/benchmarks/summary — including empty-store behavior and OpenAPI presence.
No model or GPU needed.
"""

import importlib
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from fastapi.testclient import TestClient  # noqa: E402

import main  # noqa: E402
from benchmark.storage import BaselineResultStore  # noqa: E402

# importlib gives the real module so monkeypatching result_store works
# (a plain `import ... as x` would bind the package's re-export instead).
router_module = importlib.import_module("api.routes.benchmarks.router")

RECORDS = [
    {
        "prompt": "first",
        "model": "fake-model",
        "response": "one",
        "latency_ms": 100.0,
        "ttft_ms": 30.0,
        "memory_mb": 10.0,
        "cpu_percent": 20.0,
        "generated_tokens": 10,
        "tokens_per_second": 100.0,  # 10 tokens / 0.1 s
        "timestamp": "2026-08-07T04:00:00+00:00",
    },
    {
        "prompt": "second",
        "model": "fake-model",
        "response": "two",
        "latency_ms": 200.0,
        "ttft_ms": 60.0,
        "memory_mb": 30.0,
        "cpu_percent": 40.0,
        "generated_tokens": 20,
        "tokens_per_second": 100.0,  # 20 tokens / 0.2 s
        "timestamp": "2026-08-07T05:00:00+00:00",
    },
    {
        "prompt": "third",
        "model": "fake-model",
        "response": "three",
        "latency_ms": 300.0,
        "ttft_ms": 90.0,
        "memory_mb": 50.0,
        "cpu_percent": 60.0,
        "generated_tokens": 30,
        "tokens_per_second": 100.0,  # 30 tokens / 0.3 s
        "timestamp": "2026-08-07T06:00:00+00:00",
    },
]


def _client_with_records(tmp: str) -> TestClient:
    """Seed ``tmp`` with the three known records and bind the router to it."""
    store = BaselineResultStore(tmp)
    for record in RECORDS:
        store.save(record)
    router_module.result_store = store
    return TestClient(main.app)


def test_list_all_benchmark_records():
    with tempfile.TemporaryDirectory() as tmp:
        client = _client_with_records(tmp)
        resp = client.get("/benchmarks")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert len(body) == 3, body
        assert [r["prompt"] for r in body] == ["first", "second", "third"]  # oldest first
        assert set(body[0]) == set(RECORDS[0]), body[0]
        print("PASS: /benchmarks ->", body)


def test_latest_benchmark_record():
    with tempfile.TemporaryDirectory() as tmp:
        client = _client_with_records(tmp)
        resp = client.get("/benchmarks/latest")
        assert resp.status_code == 200, resp.text
        assert resp.json()["prompt"] == "third"
        assert resp.json()["latency_ms"] == 300.0
        assert resp.json()["ttft_ms"] == 90.0
        assert resp.json()["generated_tokens"] == 30
        assert resp.json()["tokens_per_second"] == 100.0
        print("PASS: /benchmarks/latest ->", resp.json()["prompt"])


def test_benchmark_summary():
    with tempfile.TemporaryDirectory() as tmp:
        client = _client_with_records(tmp)
        resp = client.get("/benchmarks/summary")
        assert resp.status_code == 200, resp.text
        assert resp.json() == {
            "avg_latency_ms": 200.0,  # (100+200+300)/3
            "avg_memory_mb": 30.0,  # (10+30+50)/3
            "avg_cpu_percent": 40.0,  # (20+40+60)/3
            "total_runs": 3,
        }, resp.json()
        print("PASS: /benchmarks/summary ->", resp.json())


def test_empty_store_behavior():
    with tempfile.TemporaryDirectory() as tmp:
        router_module.result_store = BaselineResultStore(tmp)
        client = TestClient(main.app)

        assert client.get("/benchmarks").json() == []

        resp = client.get("/benchmarks/latest")
        assert resp.status_code == 404, resp.text
        assert resp.json()["detail"] == "No benchmark records found"

        assert client.get("/benchmarks/summary").json() == {
            "avg_latency_ms": 0.0,
            "avg_memory_mb": 0.0,
            "avg_cpu_percent": 0.0,
            "total_runs": 0,
        }
        print("PASS: empty store -> [], 404, zeroed summary")


def test_schema_invalid_records_are_skipped_not_fatal():
    with tempfile.TemporaryDirectory() as tmp:
        store = BaselineResultStore(tmp)
        store.save(RECORDS[0])  # valid
        store.save({**RECORDS[1], "latency_ms": "not-a-number"})  # schema-invalid
        store.save({**RECORDS[2], "latency_ms": None})  # schema-invalid
        router_module.result_store = store
        client = TestClient(main.app)

        resp = client.get("/benchmarks")
        assert resp.status_code == 200, resp.text
        assert len(resp.json()) == 1 and resp.json()[0]["prompt"] == "first"

        resp = client.get("/benchmarks/latest")
        assert resp.status_code == 200 and resp.json()["prompt"] == "first"

        resp = client.get("/benchmarks/summary")
        assert resp.status_code == 200 and resp.json()["total_runs"] == 1
        print("PASS: schema-invalid records skipped ->", resp.json())


def test_legacy_records_without_token_metrics_remain_readable():
    """Records saved before the token metrics existed (no generated_tokens /
    tokens_per_second keys) must not be dropped by the read API."""
    with tempfile.TemporaryDirectory() as tmp:
        store = BaselineResultStore(tmp)
        legacy = {
            k: v for k, v in RECORDS[0].items()
            if k not in {"ttft_ms", "generated_tokens", "tokens_per_second"}
        }
        store.save(legacy)
        router_module.result_store = store
        client = TestClient(main.app)

        resp = client.get("/benchmarks")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert len(body) == 1 and body[0]["prompt"] == "first", body
        assert body[0]["generated_tokens"] is None
        assert body[0]["tokens_per_second"] is None
        assert body[0]["ttft_ms"] is None
        print("PASS: legacy records without token metrics still load ->", body)


def test_swagger_documents_benchmark_endpoints():
    with tempfile.TemporaryDirectory() as tmp:
        router_module.result_store = BaselineResultStore(tmp)
        client = TestClient(main.app)
        spec = client.get("/openapi.json").json()
        paths = spec["paths"]
        for path in ("/benchmarks", "/benchmarks/latest", "/benchmarks/summary"):
            assert path in paths, path
        # Response models are referenced in the OpenAPI components.
        assert "BenchmarkRecord" in spec["components"]["schemas"]
        assert "BenchmarkSummary" in spec["components"]["schemas"]
        print("PASS: OpenAPI documents /benchmarks paths + schemas")


if __name__ == "__main__":
    test_list_all_benchmark_records()
    test_latest_benchmark_record()
    test_benchmark_summary()
    test_empty_store_behavior()
    test_schema_invalid_records_are_skipped_not_fatal()
    test_legacy_records_without_token_metrics_remain_readable()
    test_swagger_documents_benchmark_endpoints()
    print(json.dumps({"result": "all benchmark API tests passed"}))
