"""STEP 13 tests: engine selection, lifecycle, schema, errors, streaming.

Uses fake engines (registered into the engine registry under test ids) — the
real ~470 MB Q4_K_M model is never loaded. Verifies:

1. engine_id request parsing
2. registry resolution
3. default behavior remains backward compatible (legacy app.state.inference)
4. generated response schema (with and without engine identity)
5. invalid engine_id -> clean 400
6. typed engine load errors -> clean 500 (no traceback)
7. SSE streaming for streaming-capable engines / 400 for others
8. benchmark records carry engine_id/runtime and support filtering
"""

import importlib
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from fastapi.testclient import TestClient  # noqa: E402

import main  # noqa: E402
import engines.registry as registry  # noqa: E402
from engines.base import EngineInfo, InferenceEngine, StreamChunk  # noqa: E402
from engines.llamacpp_optimized import LlamaCppOptimizedEngine  # noqa: E402
from engines.manager import EngineManager  # noqa: E402
from engines.registry import UnknownEngineError, get_engine_class  # noqa: E402
from engines.result import GenerationResult  # noqa: E402
from benchmark.storage import BaselineResultStore  # noqa: E402

inference_router = importlib.import_module("api.routes.inference.router")
benchmarks_router = importlib.import_module("api.routes.benchmarks.router")

FAKE_ENGINE_ID = "fake-engine"
FAILING_ENGINE_ID = "failing-engine"
NONSTREAM_ENGINE_ID = "fake-nonstream"


class FakeStreamingEngine(InferenceEngine):
    """Deterministic fake engine: generates and streams without any model."""

    engine_id = FAKE_ENGINE_ID
    runtime = "fake-runtime"
    supports_streaming = True

    @classmethod
    def load_model(cls, **load_kwargs):
        return cls()

    def generate(self, prompt, **kwargs):
        return GenerationResult(
            prompt=prompt,
            generated_text="hello world",
            model_id="fake-model",
            prompt_tokens=3,
            generated_tokens=2,
            latency_ms=12.34,
            ttft_ms=5.0,
        )

    def stream_generate(self, prompt, **kwargs):
        yield StreamChunk(text="hello ", is_first=True)
        yield StreamChunk(text="world")
        yield StreamChunk(text="", is_last=True)

    def get_model_info(self):
        return EngineInfo(
            engine_id=self.engine_id,
            runtime=self.runtime,
            supports_streaming=True,
            model_id="fake-model",
            loaded=True,
        )


class FakeNonStreamingEngine(InferenceEngine):
    """Fake engine that loads fine but does not support streaming."""

    engine_id = NONSTREAM_ENGINE_ID
    runtime = "fake-runtime"
    supports_streaming = False

    @classmethod
    def load_model(cls, **load_kwargs):
        return cls()

    def generate(self, prompt, **kwargs):
        return GenerationResult(
            prompt=prompt,
            generated_text="plain output",
            model_id="fake-model",
            prompt_tokens=3,
            generated_tokens=2,
            latency_ms=5.0,
            ttft_ms=None,
        )

    def get_model_info(self):
        return EngineInfo(
            engine_id=self.engine_id,
            runtime=self.runtime,
            supports_streaming=False,
            model_id="fake-model",
            loaded=True,
        )


class FakeFailingEngine(InferenceEngine):
    """Fake engine whose load always fails with a typed engine error."""

    engine_id = FAILING_ENGINE_ID
    runtime = "fake-runtime"
    supports_streaming = False

    @classmethod
    def load_model(cls, **load_kwargs):
        from engines.llamacpp_optimized import LlamaCppModelLoadError

        raise LlamaCppModelLoadError("GGUF model file not found: /fake/missing.gguf")

    def generate(self, prompt, **kwargs):
        raise AssertionError("generate() must never be reached for a load failure")

    def get_model_info(self):
        return EngineInfo(engine_id=self.engine_id, runtime=self.runtime)


def _register_fakes():
    registry.ENGINE_REGISTRY[FAKE_ENGINE_ID] = FakeStreamingEngine
    registry.ENGINE_REGISTRY[FAILING_ENGINE_ID] = FakeFailingEngine
    registry.ENGINE_REGISTRY[NONSTREAM_ENGINE_ID] = FakeNonStreamingEngine


def _unregister_fakes():
    registry.ENGINE_REGISTRY.pop(FAKE_ENGINE_ID, None)
    registry.ENGINE_REGISTRY.pop(FAILING_ENGINE_ID, None)
    registry.ENGINE_REGISTRY.pop(NONSTREAM_ENGINE_ID, None)


def _manager(*, default=FAKE_ENGINE_ID) -> EngineManager:
    return EngineManager(
        default_engine_id=default,
        engine_kwargs={
            FAKE_ENGINE_ID: {},
            FAILING_ENGINE_ID: {},
            NONSTREAM_ENGINE_ID: {},
        },
    )


def _client_with_manager(tmp: str, manager: EngineManager) -> TestClient:
    """Bind a fresh engine manager + isolated result store, return a client."""
    main.app.state.engine_manager = manager
    inference_router.result_store = BaselineResultStore(tmp)
    return TestClient(main.app)


# ---------------------------------------------------------------------------
# 1. engine_id request parsing
# ---------------------------------------------------------------------------

def test_engine_id_optional_in_request_schema():
    _register_fakes()
    try:
        client = TestClient(main.app)
        spec = client.get("/openapi.json").json()
        props = spec["components"]["schemas"]["GenerateRequest"]["properties"]
        assert "engine_id" in props, props.keys()
        # Nullable (optional) — type may be a bare "string" or an anyOf
        # union depending on the Pydantic version; default is always null.
        assert props["engine_id"].get("default") is None
        # Prompt remains required.
        assert "prompt" in props
        print("PASS: GenerateRequest has optional engine_id (OpenAPI)")
    finally:
        _unregister_fakes()


# ---------------------------------------------------------------------------
# 2. Registry resolution
# ---------------------------------------------------------------------------

def test_registry_resolves_llamacpp_optimized():
    assert get_engine_class("llamacpp-optimized") is LlamaCppOptimizedEngine
    assert "llamacpp-optimized" in registry.available_engines()
    print("PASS: registry resolves llamacpp-optimized -> LlamaCppOptimizedEngine")


def test_registry_rejects_unknown_engine():
    try:
        get_engine_class("no-such-engine")
    except UnknownEngineError as exc:
        assert "no-such-engine" in str(exc)
        print("PASS: unknown engine id rejected ->", exc)
        return
    raise AssertionError("expected UnknownEngineError")


# ---------------------------------------------------------------------------
# 3. Engine manager: default, selection, unknown
# ---------------------------------------------------------------------------

def test_manager_default_and_explicit_selection():
    manager = _manager(default="llamacpp-optimized")
    assert manager.resolve(None) == "llamacpp-optimized"
    assert manager.resolve("") == "llamacpp-optimized"
    assert manager.resolve("llamacpp-optimized") == "llamacpp-optimized"
    try:
        manager.resolve("nope")
    except UnknownEngineError:
        print("PASS: manager.resolve(None)->default, explicit, unknown rejected")
        return
    raise AssertionError("expected UnknownEngineError for unknown id")


def test_manager_loads_once_and_caches():
    _register_fakes()
    try:
        manager = _manager()
        first = manager.get(FAKE_ENGINE_ID)
        second = manager.get(FAKE_ENGINE_ID)
        assert first is second, "engine must be loaded once and reused"
        assert FAKE_ENGINE_ID in manager.snapshot()
        # Snapshot never loads: fresh manager reports nothing loaded.
        assert EngineManager().snapshot() == {}
        print("PASS: engine loaded once, cached, snapshot shows load state")
    finally:
        _unregister_fakes()


# ---------------------------------------------------------------------------
# 4. Backward compatibility: legacy app.state.inference, no manager
# ---------------------------------------------------------------------------

def test_default_behavior_backward_compatible():
    main.app.state.engine_manager = None  # force the legacy path

    class LegacyService:
        model_id = "legacy-model"

        def generate(self, prompt):
            return type(
                "R",
                (),
                {
                    "generated_text": "legacy output",
                    "model_id": "legacy-model",
                    "generated_tokens": 2,
                    "latency_ms": 1.0,
                    "ttft_ms": None,
                },
            )()

    main.app.state.inference = LegacyService()
    client = TestClient(main.app)
    resp = client.post("/generate", json={"prompt": "hi"})
    assert resp.status_code == 200, resp.text
    # Legacy shape preserved exactly: no engine/token metadata leaked.
    assert resp.json() == {
        "status": "success",
        "model": "legacy-model",
        "response": "legacy output",
        "latency_ms": 1.0,
    }, resp.json()
    print("PASS: no engine_id -> legacy app.state.inference, exact legacy body")


# ---------------------------------------------------------------------------
# 5. Generated response schema with engine identity
# ---------------------------------------------------------------------------

def test_generate_response_schema_with_engine():
    _register_fakes()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            client = _client_with_manager(tmp, _manager())
            resp = client.post(
                "/generate",
                json={"prompt": "hi", "engine_id": FAKE_ENGINE_ID},
            )
            assert resp.status_code == 200, resp.text
            body = resp.json()
            assert body["status"] == "success"
            assert body["model"] == "fake-model"
            assert body["response"] == "hello world"
            assert body["engine_id"] == FAKE_ENGINE_ID
            assert body["runtime"] == "fake-runtime"
            assert body["generated_tokens"] == 2
            assert body["tokens_per_second"] == round(2 / (12.34 / 1000), 2), body
            assert body["ttft_ms"] == 5.0

            # The auto-saved record carries the engine identity.
            files = sorted(Path(tmp).glob("baseline-*.json"))
            assert len(files) == 1, files
            record = json.loads(files[0].read_text(encoding="utf-8"))
            assert record["engine_id"] == FAKE_ENGINE_ID
            assert record["runtime"] == "fake-runtime"
            print("PASS: /generate response schema + engine-tagged record")
    finally:
        _unregister_fakes()


def test_generate_default_engine_when_omitted():
    _register_fakes()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            client = _client_with_manager(tmp, _manager(default=FAKE_ENGINE_ID))
            resp = client.post("/generate", json={"prompt": "hi"})
            assert resp.status_code == 200, resp.text
            assert resp.json()["engine_id"] == FAKE_ENGINE_ID
            print("PASS: omitted engine_id -> configured default engine")
    finally:
        _unregister_fakes()


# ---------------------------------------------------------------------------
# 6. Invalid engine_id -> clean 400
# ---------------------------------------------------------------------------

def test_invalid_engine_id_returns_clean_400():
    _register_fakes()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            client = _client_with_manager(tmp, _manager())
            resp = client.post(
                "/generate", json={"prompt": "hi", "engine_id": "not-real"}
            )
            assert resp.status_code == 400, resp.text
            detail = resp.json()["detail"]
            assert "not-real" in detail
            assert "Traceback" not in resp.text
            print("PASS: unknown engine_id -> 400 with clean detail")
    finally:
        _unregister_fakes()


# ---------------------------------------------------------------------------
# 7. Typed engine load errors -> clean 500 (no traceback)
# ---------------------------------------------------------------------------

def test_typed_engine_load_error_is_clean():
    _register_fakes()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            client = _client_with_manager(tmp, _manager())
            resp = client.post(
                "/generate",
                json={"prompt": "hi", "engine_id": FAILING_ENGINE_ID},
            )
            assert resp.status_code == 500, resp.text
            body = resp.json()
            assert "GGUF model file not found" in body["detail"]
            assert "Traceback" not in resp.text
            print("PASS: typed load error -> 500 with message, no traceback")
    finally:
        _unregister_fakes()


# ---------------------------------------------------------------------------
# 8. Streaming (SSE)
# ---------------------------------------------------------------------------

def _parse_sse(body: str) -> list[dict]:
    events = []
    for block in body.split("\n\n"):
        for line in block.splitlines():
            if line.startswith("data:"):
                events.append(json.loads(line[len("data:"):].strip()))
    return events


def test_streaming_sse_for_streaming_engine():
    _register_fakes()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            client = _client_with_manager(tmp, _manager())
            resp = client.post(
                "/generate/stream",
                json={"prompt": "hi", "engine_id": FAKE_ENGINE_ID},
            )
            assert resp.status_code == 200, resp.text
            assert resp.headers["content-type"].startswith("text/event-stream")
            events = _parse_sse(resp.text)
            texts = [e["text"] for e in events if "text" in e]
            assert texts == ["hello ", "world"], texts
            done = [e for e in events if e.get("done")]
            assert len(done) == 1, events
            meta = done[0]
            assert meta["engine_id"] == FAKE_ENGINE_ID
            assert meta["runtime"] == "fake-runtime"
            assert meta["generated_tokens"] == 2
            assert meta["latency_ms"] > 0
            assert meta["tokens_per_second"] is not None
            assert meta["ttft_ms"] is not None
            print("PASS: SSE stream -> text events + done metadata event")
    finally:
        _unregister_fakes()


def test_streaming_rejected_for_non_streaming_engine():
    _register_fakes()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            client = _client_with_manager(tmp, _manager())
            resp = client.post(
                "/generate/stream",
                json={"prompt": "hi", "engine_id": NONSTREAM_ENGINE_ID},
            )
            assert resp.status_code == 400, resp.text
            assert "does not support streaming" in resp.json()["detail"]
            print("PASS: non-streaming engine -> 400 with clean detail")
    finally:
        _unregister_fakes()


# ---------------------------------------------------------------------------
# 9. Benchmark engine identification + filter
# ---------------------------------------------------------------------------

def test_benchmark_records_carry_engine_and_filter():
    def _record(prompt, latency, engine_id=None, runtime=None):
        rec = {
            "prompt": prompt,
            "model": "fake-model",
            "response": "x",
            "latency_ms": latency,
            "memory_mb": 10.0,
            "cpu_percent": 20.0,
            "timestamp": f"2026-08-07T04:{len(prompt):02d}:00+00:00",
        }
        if engine_id:
            rec["engine_id"] = engine_id
            rec["runtime"] = runtime
        return rec

    with tempfile.TemporaryDirectory() as tmp:
        store = BaselineResultStore(tmp)
        store.save(_record("legacy", 100.0))
        store.save(_record("llama run", 200.0, "llamacpp-optimized", "llama.cpp"))
        store.save(_record("baseline run", 300.0, "transformers-baseline", "transformers"))
        benchmarks_router.result_store = store
        client = TestClient(main.app)

        all_records = client.get("/benchmarks").json()
        assert len(all_records) == 3, all_records
        llama_only = client.get("/benchmarks", params={"engine_id": "llamacpp-optimized"}).json()
        assert len(llama_only) == 1, llama_only
        assert llama_only[0]["engine_id"] == "llamacpp-optimized"
        assert llama_only[0]["runtime"] == "llama.cpp"
        # Filtered view never mixes in untagged legacy runs.
        assert client.get("/benchmarks", params={"engine_id": "nope"}).json() == []

        # Untagged legacy records keep their exact (engine-less) shape.
        legacy = [r for r in all_records if r["prompt"] == "legacy"][0]
        assert "engine_id" not in legacy
        print("PASS: /benchmarks identifies engines and filters without mixing")


def test_benchmark_summary_can_filter_by_engine():
    with tempfile.TemporaryDirectory() as tmp:
        store = BaselineResultStore(tmp)
        store.save({"prompt": "a", "model": "m", "response": "x", "latency_ms": 100.0,
                    "memory_mb": 10.0, "cpu_percent": 20.0,
                    "timestamp": "2026-08-07T04:01:00+00:00",
                    "engine_id": "llamacpp-optimized", "runtime": "llama.cpp"})
        store.save({"prompt": "b", "model": "m", "response": "x", "latency_ms": 300.0,
                    "memory_mb": 30.0, "cpu_percent": 40.0,
                    "timestamp": "2026-08-07T04:02:00+00:00",
                    "engine_id": "transformers-baseline", "runtime": "transformers"})
        benchmarks_router.result_store = store
        client = TestClient(main.app)

        summary = client.get("/benchmarks/summary", params={"engine_id": "llamacpp-optimized"}).json()
        assert summary == {"avg_latency_ms": 100.0, "avg_memory_mb": 10.0,
                           "avg_cpu_percent": 20.0, "total_runs": 1}, summary
        print("PASS: /benchmarks/summary filters by engine")


if __name__ == "__main__":
    test_engine_id_optional_in_request_schema()
    test_registry_resolves_llamacpp_optimized()
    test_registry_rejects_unknown_engine()
    test_manager_default_and_explicit_selection()
    test_manager_loads_once_and_caches()
    test_default_behavior_backward_compatible()
    test_generate_response_schema_with_engine()
    test_generate_default_engine_when_omitted()
    test_invalid_engine_id_returns_clean_400()
    test_typed_engine_load_error_is_clean()
    test_streaming_sse_for_streaming_engine()
    test_streaming_rejected_for_non_streaming_engine()
    test_benchmark_records_carry_engine_and_filter()
    test_benchmark_summary_can_filter_by_engine()
    print("\nALL ENGINE-SELECTION TESTS PASSED")
