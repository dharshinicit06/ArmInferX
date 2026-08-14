"""Tests for the engine-agnostic benchmark layer (Step 8B).

Covers ``BenchmarkConfig`` validation/defaults, ``BenchmarkRunner`` (warmup,
exact repeat count, identical kwargs, chat-template policy, engine identity,
aggregates), the engine registry, and backward compatibility of the result
stores. Uses fake engines only — no real llama.cpp model is loaded, and no
performance claims are made from fake numbers.
"""

import json
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from benchmark import (  # noqa: E402
    BenchmarkAggregates,
    BenchmarkConfig,
    BenchmarkConfigError,
    BenchmarkMetrics,
    BenchmarkRun,
    BenchmarkRunner,
    BaselineResultStore,
    EngineResultStore,
)
from engines import (  # noqa: E402
    EngineInfo,
    UnknownEngineError,
    available_engines,
    get_engine_class,
    load_engine,
)


# ---------------------------------------------------------------------------
# Deterministic fake engines
# ---------------------------------------------------------------------------

class _FakeEngineBase:
    """Records every generate() call and returns deterministic results."""

    engine_id: str = "fake"
    runtime: str = "fake"
    model_id: str = "fake-model"

    def __init__(self, latency_ms: float = 10.0) -> None:
        self.latency_ms = latency_ms
        self.calls: list[tuple[str, dict]] = []  # (prompt, kwargs)

    def generate(self, prompt: str, **kwargs):
        self.calls.append((prompt, kwargs))
        return _FakeResult(self.latency_ms, self.model_id)

    def get_model_info(self) -> EngineInfo:
        return EngineInfo(
            engine_id=self.engine_id,
            runtime=self.runtime,
            supports_streaming=False,
            model_id=self.model_id,
            max_context=4096,
            loaded=True,
            extra={},
        )


class FakeChatTemplateEngine(_FakeEngineBase):
    """Fake engine that opts into chat templates (registered for the test)."""

    engine_id = "chat-template-engine"
    runtime = "chat-template"


class FakeLlamaEngine(_FakeEngineBase):
    engine_id = "llamacpp-optimized"
    runtime = "llama.cpp"

    def get_model_info(self) -> EngineInfo:
        info = super().get_model_info()
        return EngineInfo(
            engine_id=info.engine_id,
            runtime=info.runtime,
            supports_streaming=True,
            model_id=info.model_id,
            max_context=info.max_context,
            loaded=True,
            extra={"model_path": "/models/fake.gguf"},
        )


class _FakeResult:
    generated_text = "fake response text"
    prompt_tokens = 5
    generated_tokens = 10
    ttft_ms = 2.5

    def __init__(self, latency_ms: float, model_id: str) -> None:
        self.latency_ms = latency_ms
        self.model_id = model_id


# ---------------------------------------------------------------------------
# 1. BenchmarkConfig defaults + validation
# ---------------------------------------------------------------------------

def test_config_defaults():
    cfg = BenchmarkConfig(prompt="Explain AI.")
    assert cfg.max_new_tokens == 128
    assert cfg.temperature is None  # greedy default
    assert cfg.chat_template is False  # raw-completion comparability policy
    assert cfg.repeats == 5
    assert cfg.warmup == 1
    print(
        "PASS: BenchmarkConfig defaults "
        f"(max_new_tokens={cfg.max_new_tokens}, temperature={cfg.temperature}, "
        f"chat_template={cfg.chat_template}, repeats={cfg.repeats}, warmup={cfg.warmup})"
    )


def test_config_validation():
    for bad in (
        lambda: BenchmarkConfig(prompt="  "),
        lambda: BenchmarkConfig(prompt="x", max_new_tokens=0),
        lambda: BenchmarkConfig(prompt="x", temperature=0.0),
        lambda: BenchmarkConfig(prompt="x", temperature=2.5),
        lambda: BenchmarkConfig(prompt="x", repeats=0),
        lambda: BenchmarkConfig(prompt="x", warmup=-1),
    ):
        try:
            bad()
        except BenchmarkConfigError:
            continue
        raise AssertionError("expected BenchmarkConfigError")
    # Valid boundary values.
    BenchmarkConfig(prompt="x", temperature=2.0, repeats=1, warmup=0)
    print("PASS: BenchmarkConfig validation (empty prompt / bad limits / bad temperature)")


# ---------------------------------------------------------------------------
# 2 + 3. Warmup + exact repetition count
# ---------------------------------------------------------------------------

def _runner_with_tmp_store(engine, tmp: str) -> BenchmarkRunner:
    """Build a runner whose engine store is isolated in ``tmp`` so the test
    suite never writes into the project's results/ tree."""
    return BenchmarkRunner(store=EngineResultStore(engine.engine_id, tmp))


def test_warmup_and_repeat_counts():
    cfg = BenchmarkConfig(prompt="hi", repeats=4, warmup=2)
    engine = FakeChatTemplateEngine()
    with tempfile.TemporaryDirectory() as tmp:
        runner = _runner_with_tmp_store(engine, tmp)
        run = runner.run(engine, cfg)

    # 2 warmup + 4 timed = 6 total generate calls.
    assert len(engine.calls) == 6, len(engine.calls)
    assert len(run.metrics) == 4, len(run.metrics)
    assert len(run.records) == 4, len(run.records)
    assert run.aggregates is not None and run.aggregates.runs == 4
    print(
        f"PASS: warmup={cfg.warmup} + repeats={cfg.repeats} -> "
        f"{len(engine.calls)} total calls, {len(run.metrics)} timed"
    )


def test_warmup_uses_same_prompt_and_kwargs():
    cfg = BenchmarkConfig(prompt="warm me", repeats=2, warmup=1)
    engine = FakeChatTemplateEngine()
    with tempfile.TemporaryDirectory() as tmp:
        _runner_with_tmp_store(engine, tmp).run(engine, cfg)

    prompt0, kwargs0 = engine.calls[0]  # warmup call
    for prompt, kwargs in engine.calls:
        assert prompt == "warm me"
        assert kwargs == kwargs0  # identical kwargs across warmup + timed runs
    print("PASS: warmup + timed calls use the identical prompt and kwargs")


# ---------------------------------------------------------------------------
# 4 + 5 + 6. Same prompt / max_new_tokens / temperature for both engines
# ---------------------------------------------------------------------------

def test_identical_config_applied_to_both_engines():
    cfg = BenchmarkConfig(
        prompt="same prompt for both", max_new_tokens=96, temperature=0.7, repeats=2
    )
    tf = FakeChatTemplateEngine()
    llm = FakeLlamaEngine()
    with tempfile.TemporaryDirectory() as tmp:
        _runner_with_tmp_store(tf, tmp).run(tf, cfg)
        _runner_with_tmp_store(llm, tmp).run(llm, cfg)

    for engine in (tf, llm):
        for prompt, kwargs in engine.calls:
            assert prompt == "same prompt for both", prompt
            assert kwargs["max_new_tokens"] == 96, kwargs
            assert kwargs["temperature"] == 0.7, kwargs
    print("PASS: both engines receive the identical prompt, max_new_tokens, temperature")


def test_temperature_none_is_not_passed():
    cfg = BenchmarkConfig(prompt="hi", repeats=1, warmup=0)  # temperature None
    engine = FakeLlamaEngine()
    with tempfile.TemporaryDirectory() as tmp:
        _runner_with_tmp_store(engine, tmp).run(engine, cfg)
    _, kwargs = engine.calls[0]
    assert "temperature" not in kwargs
    assert kwargs["max_new_tokens"] == 128
    print("PASS: temperature=None -> kwarg omitted (greedy defaults, deterministic)")


# ---------------------------------------------------------------------------
# 7 + 8. Chat-template policy + no unsupported kwargs for llama.cpp
# ---------------------------------------------------------------------------

@contextmanager
def _patch_chat_template_ids(*ids: str):
    """Temporarily make ``engine_id``s opt into chat templates."""
    import benchmark.runner as runner_module

    original = runner_module.CHAT_TEMPLATE_ENGINE_IDS
    runner_module.CHAT_TEMPLATE_ENGINE_IDS = tuple(ids)
    try:
        yield
    finally:
        runner_module.CHAT_TEMPLATE_ENGINE_IDS = original


def test_chat_template_policy():
    cfg = BenchmarkConfig(prompt="hi", repeats=1, warmup=0)  # chat_template=False
    tf = FakeChatTemplateEngine()
    llm = FakeLlamaEngine()
    with tempfile.TemporaryDirectory() as tmp:
        with _patch_chat_template_ids("chat-template-engine"):
            _runner_with_tmp_store(tf, tmp).run(tf, cfg)
            _runner_with_tmp_store(llm, tmp).run(llm, cfg)

    # Chat-template opt-in engine: use_chat_template explicitly controlled
    # (False by default).
    assert tf.calls[0][1]["use_chat_template"] is False
    # llama.cpp: must never receive use_chat_template or do_sample.
    for prompt, kwargs in llm.calls:
        assert "use_chat_template" not in kwargs
        assert "do_sample" not in kwargs
    assert set(tf.calls[0][1]) == {"max_new_tokens", "use_chat_template"}
    assert set(llm.calls[0][1]) == {"max_new_tokens"}
    print("PASS: chat-template policy (opt-in engine controlled; llama.cpp gets no template/do_sample)")


def test_chat_template_true_is_forwarded_to_optin_only():
    cfg = BenchmarkConfig(prompt="hi", repeats=1, warmup=0, chat_template=True)
    tf = FakeChatTemplateEngine()
    llm = FakeLlamaEngine()
    with tempfile.TemporaryDirectory() as tmp:
        with _patch_chat_template_ids("chat-template-engine"):
            _runner_with_tmp_store(tf, tmp).run(tf, cfg)
            _runner_with_tmp_store(llm, tmp).run(llm, cfg)

    assert tf.calls[0][1]["use_chat_template"] is True
    assert "use_chat_template" not in llm.calls[0][1]
    print("PASS: chat_template=True forwarded to opt-in engine only")


# ---------------------------------------------------------------------------
# 9. Engine identity present in results
# ---------------------------------------------------------------------------

def test_engine_identity_in_results():
    cfg = BenchmarkConfig(prompt="hi", repeats=2, warmup=1)
    llm = FakeLlamaEngine()
    with tempfile.TemporaryDirectory() as tmp:
        run = _runner_with_tmp_store(llm, tmp).run(llm, cfg)

    assert run.engine_id == "llamacpp-optimized"
    assert run.runtime == "llama.cpp"
    assert run.model_id == "fake-model"
    assert run.model_path == "/models/fake.gguf"  # from EngineInfo.extra

    for metrics in run.metrics:
        assert metrics.extra["engine_id"] == "llamacpp-optimized"
        assert metrics.extra["runtime"] == "llama.cpp"
        assert metrics.extra["model_id"] == "fake-model"
        assert metrics.extra["model_path"] == "/models/fake.gguf"
    for record in run.records:
        assert record["engine_id"] == "llamacpp-optimized"
        assert record["runtime"] == "llama.cpp"
        assert record["model_path"] == "/models/fake.gguf"

    d = run.to_dict()
    assert d["engine_id"] == "llamacpp-optimized" and d["runtime"] == "llama.cpp"
    assert d["config"]["prompt"] == "hi"
    print("PASS: engine identity (engine_id/runtime/model_id/model_path) on run, metrics, records")


# ---------------------------------------------------------------------------
# 10 + 11 + 12. Aggregate statistics
# ---------------------------------------------------------------------------

def test_aggregate_mean_median_p90():
    # 5 deterministic latencies -> mean 30, median 30, p90 50.
    latencies = [10.0, 50.0, 30.0, 20.0, 40.0]

    class SequencingEngine(FakeChatTemplateEngine):
        def __init__(self):
            super().__init__()
            self._i = 0

        def generate(self, prompt, **kwargs):
            self.calls.append((prompt, kwargs))
            lat = latencies[min(self._i, len(latencies) - 1)]
            self._i += 1
            return _FakeResult(lat, self.model_id)

    cfg = BenchmarkConfig(prompt="hi", repeats=5, warmup=0)
    with tempfile.TemporaryDirectory() as tmp:
        engine = SequencingEngine()
        run = _runner_with_tmp_store(engine, tmp).run(engine, cfg)

    agg = run.aggregates
    assert isinstance(agg, BenchmarkAggregates)
    assert agg.mean_latency_ms == 30.0
    assert agg.median_latency_ms == 30.0
    assert agg.p90_latency_ms == 50.0  # nearest-rank of sorted [10,20,30,40,50]
    assert agg.mean_generated_tokens == 10.0
    assert agg.mean_ttft_ms == 2.5
    assert agg.peak_memory_mb > 0
    assert agg.runs == 5
    print(
        "PASS: aggregates "
        f"(mean={agg.mean_latency_ms}, median={agg.median_latency_ms}, "
        f"p90={agg.p90_latency_ms}, mean_ttft={agg.mean_ttft_ms}ms, "
        f"mean_tokens={agg.mean_generated_tokens})"
    )


# ---------------------------------------------------------------------------
# 13. Backward compatibility of result stores
# ---------------------------------------------------------------------------

def test_legacy_baseline_records_still_readable():
    with tempfile.TemporaryDirectory() as tmp:
        store = BaselineResultStore(tmp)
        legacy = {
            "prompt": "old",
            "model": "qwen2.5-0.5b-instruct",
            "response": "old response",
            "latency_ms": 123.0,
            "memory_mb": 10.0,
            "cpu_percent": 5.0,
            "timestamp": "2026-08-01T00:00:00+00:00",
        }  # no engine_id/runtime/ttft/generated_tokens keys
        store.save(legacy)

        records = store.list_records()
        assert len(records) == 1
        assert "engine_id" not in records[0]  # untouched legacy record
        assert records[0]["latency_ms"] == 123.0
        print("PASS: legacy baseline records (no engine fields) still load untouched")


def test_engine_store_writes_distinct_dir_and_records():
    with tempfile.TemporaryDirectory() as tmp:
        store = EngineResultStore("llamacpp-optimized", tmp)
        store.save({"engine_id": "llamacpp-optimized", "latency_ms": 1.0, "timestamp": "t1"})
        store.save({"engine_id": "llamacpp-optimized", "latency_ms": 2.0, "timestamp": "t2"})

        assert len(store.list_records()) == 2
        assert store.latest()["latency_ms"] == 2.0
        assert all(
            p.name.startswith("benchmark-") for p in Path(tmp).glob("*.json")
        )
        # Baseline store in the same dir must not see engine records.
        baseline = BaselineResultStore(tmp)
        assert baseline.list_records() == []
        print("PASS: EngineResultStore isolates engine records from the baseline store")


def test_runner_persists_to_engine_store():
    with tempfile.TemporaryDirectory() as tmp:
        store = EngineResultStore("chat-template-engine", tmp)
        runner = BenchmarkRunner(store=store)
        cfg = BenchmarkConfig(prompt="hi", repeats=3, warmup=0)
        run = runner.run(FakeChatTemplateEngine(), cfg)

        files = sorted(Path(tmp).glob("benchmark-*.json"))
        assert len(files) == 3, files
        record = json.loads(files[0].read_text(encoding="utf-8"))
        assert record["engine_id"] == "chat-template-engine"
        assert record["runtime"] == "chat-template"
        assert run.aggregates is not None and run.aggregates.runs == 3
        print("PASS: runner persists 1 record per repeat, engine-tagged")


# ---------------------------------------------------------------------------
# 14 + 15. Engine registry
# ---------------------------------------------------------------------------

def test_registry_resolves_llamacpp_engine():
    assert available_engines() == ["llamacpp-optimized"]
    assert get_engine_class("llamacpp-optimized").engine_id == "llamacpp-optimized"
    print("PASS: registry resolves the llamacpp engine id:", available_engines())


def test_registry_unknown_engine_raises():
    try:
        load_engine("does-not-exist")
    except UnknownEngineError as exc:
        assert "does-not-exist" in str(exc)
        assert "llamacpp-optimized" in str(exc)
        print("PASS: unknown engine id raises UnknownEngineError")
        return
    raise AssertionError("expected UnknownEngineError")


if __name__ == "__main__":
    test_config_defaults()
    test_config_validation()
    test_warmup_and_repeat_counts()
    test_warmup_uses_same_prompt_and_kwargs()
    test_identical_config_applied_to_both_engines()
    test_temperature_none_is_not_passed()
    test_chat_template_policy()
    test_chat_template_true_is_forwarded_to_optin_only()
    test_engine_identity_in_results()
    test_aggregate_mean_median_p90()
    test_legacy_baseline_records_still_readable()
    test_engine_store_writes_distinct_dir_and_records()
    test_runner_persists_to_engine_store()
    test_registry_resolves_llamacpp_engine()
    test_registry_unknown_engine_raises()
    print(json.dumps({"result": "all benchmark runner tests passed"}))
