"""Unit tests for the inference engine abstraction.

Covers the ``InferenceEngine`` interface contract, the
``TransformersBaselineEngine`` adapter (which must preserve the existing
baseline behavior), and the ``LlamaCppOptimizedEngine`` (llama.cpp/GGUF). No
real model or GPU is needed: the baseline engine is exercised through a fake
model/tokenizer, ``load_model`` is tested with a monkeypatched loader, and
the llama.cpp engine uses a fake ``Llama`` class so nothing is loaded from
disk (the ~6.5 GB FP16 GGUF must not be loaded on this 8 GB laptop).
"""

import importlib
import json
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

import torch  # noqa: E402

from api.routes.inference.inference_service import InferenceService  # noqa: E402
from api.routes.inference.inference_service import (  # noqa: E402
    GenerationResult as ResultFromService,
)
from engines import (  # noqa: E402
    EngineInfo,
    EngineOperationUnsupportedError,
    GenerationResult,
    InferenceEngine,
    LlamaCppOptimizedEngine,
    TransformersBaselineEngine,
)
from engines.llamacpp_optimized import (  # noqa: E402
    DEFAULT_MODEL_PATH,
    LlamaCppConfigError,
    LlamaCppError,
    LlamaCppModelLoadError,
)


# ---------------------------------------------------------------------------
# Fakes (mirror tests/test_inference_service_ttft.py)
# ---------------------------------------------------------------------------

class FakeModel:
    config = SimpleNamespace(max_position_embeddings=4096)

    def generate(self, input_ids, max_new_tokens=None, do_sample=None, streamer=None, **kwargs):
        assert streamer is not None, "service must attach the TTFT streamer"
        time.sleep(0.01)  # prefill + first decode step
        streamer.put(torch.tensor([[5]]))
        streamer.put(torch.tensor([[6]]))
        return torch.cat([input_ids, torch.tensor([[5, 6]])], dim=1)


class FakeTokenizer:
    chat_template = "fake-template"  # present -> chat-template path is usable

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        # Naive template: just concatenate the user content.
        return "".join(m["content"] for m in messages)

    def __call__(self, text, return_tensors="pt"):
        ids = [ord(ch) % 50 + 1 for ch in text]
        return {"input_ids": torch.tensor([ids])}

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(int(i) % 26 + 97) for i in ids)


def _fake_inference_model() -> SimpleNamespace:
    return SimpleNamespace(
        model=FakeModel(), tokenizer=FakeTokenizer(), model_id="fake-model"
    )


def _make_baseline_engine() -> TransformersBaselineEngine:
    service = InferenceService(_fake_inference_model())
    return TransformersBaselineEngine(service)


# ---------------------------------------------------------------------------
# Interface contract
# ---------------------------------------------------------------------------

def test_interface_is_abstract():
    try:
        InferenceEngine()
    except TypeError:
        print("PASS: InferenceEngine cannot be instantiated directly")
        return
    raise AssertionError("expected TypeError when instantiating the abstract engine")


def test_interface_exposes_required_operations():
    for name in ("load_model", "generate", "stream_generate", "get_model_info"):
        assert hasattr(InferenceEngine, name), f"missing {name}"
    abstract = set(InferenceEngine.__abstractmethods__)
    assert {"load_model", "generate", "get_model_info"} <= abstract, abstract
    # stream_generate is concrete on the base (raises EngineOperationUnsupportedError).
    assert "stream_generate" not in abstract
    print("PASS: interface exposes load_model/generate/stream_generate/get_model_info")


def test_engine_info_serialization():
    info = EngineInfo(
        engine_id="x", runtime="r", supports_streaming=False, model_id="m", loaded=True
    )
    d = info.to_dict()
    assert d == {
        "engine_id": "x",
        "runtime": "r",
        "supports_streaming": False,
        "model_id": "m",
        "max_context": None,
        "loaded": True,
    }, d
    # Defaults: unloaded, no model.
    assert EngineInfo(engine_id="e", runtime="t", supports_streaming=True).loaded is False
    print("PASS: EngineInfo.to_dict shape + defaults")


def test_generation_result_is_shared_type():
    # The result contract lives in engines; the baseline service re-exports
    # the same class so existing imports keep working.
    assert GenerationResult is ResultFromService
    print("PASS: GenerationResult shared between engines and baseline service")


# ---------------------------------------------------------------------------
# TransformersBaselineEngine
# ---------------------------------------------------------------------------

def test_baseline_engine_generates_via_service():
    engine = _make_baseline_engine()
    result = engine.generate("hi", use_chat_template=False)

    assert isinstance(result, GenerationResult)
    assert result.generated_text == "fg"
    assert result.generated_tokens == 2
    assert result.ttft_ms is not None and 5 <= result.ttft_ms <= 300, result.ttft_ms
    assert result.latency_ms >= result.ttft_ms
    print(f"PASS: baseline engine generate -> {result.generated_text!r} (ttft={result.ttft_ms:.1f}ms)")


def test_baseline_engine_model_info_and_properties():
    engine = _make_baseline_engine()
    info = engine.get_model_info()

    assert isinstance(info, EngineInfo)
    assert info.engine_id == "transformers-baseline"
    assert info.runtime == "transformers"
    assert info.supports_streaming is False
    assert info.model_id == "fake-model"
    assert info.max_context == 4096
    assert info.loaded is True

    # Convenience passthroughs used by /health.
    assert engine.model_id == "fake-model"
    assert engine.max_context == 4096
    print("PASS: baseline engine model info + properties")


def test_baseline_engine_does_not_support_streaming():
    engine = _make_baseline_engine()
    try:
        engine.stream_generate("hi", use_chat_template=False)
    except EngineOperationUnsupportedError as exc:
        assert "transformers-baseline" in str(exc)
        print("PASS: baseline stream_generate raises EngineOperationUnsupportedError")
        return
    raise AssertionError("expected EngineOperationUnsupportedError")


def test_baseline_engine_load_model_wiring():
    # The engine imports the loader lazily inside load_model(), so patch the
    # loader module itself to verify the wiring without a real model on disk.
    from api.routes.inference import model_loader as loader_mod

    original = loader_mod.load_inference_model
    captured = {}

    def fake_load(model_dir, *, device, dtype, max_cpu_memory):
        captured["model_dir"] = model_dir
        captured["device"] = device
        captured["dtype"] = dtype
        captured["max_cpu_memory"] = max_cpu_memory
        return _fake_inference_model()

    loader_mod.load_inference_model = fake_load
    try:
        engine = TransformersBaselineEngine.load_model(
            "/models/qwen", device="cpu", dtype="float16", max_cpu_memory="3GiB"
        )
    finally:
        loader_mod.load_inference_model = original

    assert captured["model_dir"] == "/models/qwen"
    assert captured["device"] == "cpu"
    assert captured["dtype"] == "float16"
    assert engine.model_id == "fake-model"
    assert engine.get_model_info().loaded is True
    print("PASS: TransformersBaselineEngine.load_model builds a ready engine")


# ---------------------------------------------------------------------------
# LlamaCppOptimizedEngine (llama.cpp / GGUF)
#
# Validated WITHOUT loading the real ~6.5 GB FP16 GGUF (this laptop has only
# 7.6 GiB RAM): a fake ``Llama`` class records the load configuration and
# emits a deterministic token stream, so import / path / wiring / metadata /
# result-shape are all verified with zero model I/O.
# ---------------------------------------------------------------------------

class FakeLlama:
    """Minimal stand-in for ``llama_cpp.Llama`` used by the validation tests."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_calls = []

    def tokenize(self, text: bytes, add_bos: bool = True, special: bool = False):
        # Deterministic tokenizer-native stand-in: 1 token per 3 bytes.
        return [1] * (max(1, len(text) // 3))

    def create_completion(self, prompt, **kwargs):
        self.create_calls.append((prompt, kwargs))
        yield {"choices": [{"text": "hel"}]}
        yield {"choices": [{"text": "lo"}]}
        yield {"choices": [{"text": ""}]}  # final empty chunk


@contextmanager
def _patch_llama(fake_cls):
    """Patch ``engines.llamacpp_optimized.Llama`` for the duration of a test."""
    import engines.llamacpp_optimized as mod

    original = mod.Llama
    mod.Llama = fake_cls
    try:
        yield
    finally:
        mod.Llama = original

def test_engine_works_through_generate_endpoint():
    """POST /generate must keep working when the engine is wired into
    app.state (the baseline routing contract)."""
    from fastapi.testclient import TestClient  # noqa: PLC0415

    import main  # noqa: PLC0415
    from benchmark.storage import BaselineResultStore  # noqa: PLC0415

    router_module = importlib.import_module("api.routes.inference.router")

    engine = _make_baseline_engine()
    with tempfile.TemporaryDirectory() as tmp:
        router_module.result_store = BaselineResultStore(tmp)
        main.app.state.inference = engine
        client = TestClient(main.app)

        resp = client.post("/generate", json={"prompt": "hi"})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["status"] == "success"
        assert body["model"] == "fake-model"
        assert body["response"] == "fg"
        assert body["latency_ms"] >= 0

        # The benchmark record is still saved with all metrics.
        files = sorted(Path(tmp).glob("baseline-*.json"))
        assert len(files) == 1, files
        record = json.loads(files[0].read_text(encoding="utf-8"))
        assert record["generated_tokens"] == 2
        assert record["ttft_ms"] is not None
        print("PASS: TransformersBaselineEngine works through POST /generate")


def test_llamacpp_engine_identity_and_default_path():
    """The engine declares the llama.cpp runtime and its GGUF path resolves."""
    assert LlamaCppOptimizedEngine.engine_id == "llamacpp-optimized"
    assert LlamaCppOptimizedEngine.runtime == "llama.cpp"
    assert LlamaCppOptimizedEngine.supports_streaming is True

    # Default model path points at the validated Q4_K_M GGUF (the engine's
    # default was switched from the FP16 split to the single-file Q4_K_M in
    # STEP 9; the file must exist on disk).
    assert DEFAULT_MODEL_PATH.name == (
        "qwen2.5-3b-instruct-q4_k_m.gguf"
    ), DEFAULT_MODEL_PATH
    assert DEFAULT_MODEL_PATH.is_file(), f"GGUF not found at {DEFAULT_MODEL_PATH}"
    print(
        "PASS: llama.cpp engine identity + default GGUF path resolves "
        f"({DEFAULT_MODEL_PATH.name})"
    )


def test_llamacpp_engine_unloaded_reports_metadata_without_model():
    """An engine without a loaded model reports metadata but refuses inference."""
    engine = LlamaCppOptimizedEngine()
    info = engine.get_model_info()
    assert info.engine_id == "llamacpp-optimized"
    assert info.runtime == "llama.cpp"
    assert info.supports_streaming is True
    assert info.loaded is False
    assert info.model_id is None
    assert info.max_context == 2048  # declared default context

    for operation in (
        lambda: engine.generate("hi"),
        lambda: list(engine.stream_generate("hi")),
    ):
        try:
            operation()
        except LlamaCppError as exc:
            assert "no loaded model" in str(exc)
            continue
        raise AssertionError("expected LlamaCppError for unloaded engine")
    print("PASS: unloaded llama.cpp engine metadata + clean refusal to generate")


def test_llamacpp_engine_load_model_wiring():
    """load_model wires config into a (fake) Llama and derives a model id."""
    captured = {}

    class RecordingLlama(FakeLlama):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            captured.update(kwargs)

    with _patch_llama(RecordingLlama):
        engine = LlamaCppOptimizedEngine.load_model(
            DEFAULT_MODEL_PATH,
            n_ctx=1024,
            n_threads=4,
            temperature=0.0,
            max_new_tokens=32,
        )

    # CPU-only + fixed configuration flow through to the runtime.
    assert captured["model_path"] == str(DEFAULT_MODEL_PATH)
    assert captured["n_ctx"] == 1024
    assert captured["n_threads"] == 4
    assert captured["n_gpu_layers"] == 0  # CPU-only
    assert captured["n_threads_batch"] == 4

    info = engine.get_model_info()
    assert info.loaded is True
    # Single-file default (Q4_K_M) keeps its full stem as the model id.
    assert info.model_id == "qwen2.5-3b-instruct-q4_k_m"
    assert info.max_context == 1024
    assert info.extra["n_gpu_layers"] == 0
    print("PASS: LlamaCppOptimizedEngine.load_model wiring (CPU-only)")


def test_llamacpp_engine_generate_returns_shared_result():
    """generate() returns the shared GenerationResult with tokenizer counts."""
    fake = FakeLlama()
    engine = LlamaCppOptimizedEngine(fake, model_id="fake-gguf")
    result = engine.generate("hi", max_new_tokens=8, top_p=0.9, stop=["END"])

    assert isinstance(result, GenerationResult)
    assert result.generated_text == "hello"
    assert result.model_id == "fake-gguf"
    assert result.generated_tokens == 2  # two emitted stream chunks
    assert result.prompt_tokens > 0  # tokenizer-native, not char/word estimate
    assert result.latency_ms >= 0
    assert result.ttft_ms is not None and result.ttft_ms >= 0
    assert result.generation_kwargs["max_new_tokens"] == 8
    # Extra generation kwargs are forwarded to the runtime (baseline parity).
    assert fake.create_calls, "create_completion was never called"
    prompt, call_kwargs = fake.create_calls[-1]
    assert prompt == "hi"
    assert call_kwargs["stream"] is True
    assert call_kwargs["max_tokens"] == 8
    assert call_kwargs["temperature"] == 0.0
    assert call_kwargs["top_p"] == 0.9
    assert call_kwargs["stop"] == ["END"]
    print(
        "PASS: generate() -> shared GenerationResult "
        f"(text={result.generated_text!r}, tokens={result.prompt_tokens}+{result.generated_tokens}, "
        f"latency={result.latency_ms:.1f}ms, ttft={result.ttft_ms:.1f}ms; kwargs forwarded)"
    )


def test_llamacpp_engine_stream_generate_yields_chunks():
    """stream_generate() yields StreamChunk deltas with first/last markers."""
    engine = LlamaCppOptimizedEngine(FakeLlama(), model_id="fake-gguf")
    chunks = list(engine.stream_generate("hi", max_new_tokens=8))

    texts = [c.text for c in chunks]
    assert texts[:-1] == ["hel", "lo"]
    assert chunks[0].is_first is True
    assert chunks[-1].is_last is True
    assert chunks[-1].text == ""
    print("PASS: stream_generate yields StreamChunk deltas + boundaries")


def test_llamacpp_engine_error_handling():
    """Missing GGUF, invalid config and empty prompt raise typed errors."""
    # Missing GGUF file -> LlamaCppModelLoadError (path problem).
    with _patch_llama(FakeLlama):
        try:
            LlamaCppOptimizedEngine.load_model("models/gguf/does-not-exist.gguf")
        except LlamaCppModelLoadError as exc:
            assert "not found" in str(exc)
        else:
            raise AssertionError("expected LlamaCppModelLoadError for missing GGUF")

    # Invalid configuration -> LlamaCppConfigError.
    for bad_kwargs in ({"n_ctx": 0}, {"n_threads": 0}, {"temperature": 3.0}):
        try:
            LlamaCppOptimizedEngine.load_model(DEFAULT_MODEL_PATH, **bad_kwargs)
        except LlamaCppConfigError:
            continue
        raise AssertionError(f"expected LlamaCppConfigError for {bad_kwargs}")

    # Empty prompt -> LlamaCppConfigError.
    engine = LlamaCppOptimizedEngine(FakeLlama(), model_id="fake-gguf")
    for operation in (lambda: engine.generate("   "), lambda: list(engine.stream_generate(""))):
        try:
            operation()
        except LlamaCppConfigError:
            continue
        raise AssertionError("expected LlamaCppConfigError for empty prompt")
    print("PASS: llama.cpp engine typed error handling (missing file / config / prompt)")


if __name__ == "__main__":
    test_interface_is_abstract()
    test_interface_exposes_required_operations()
    test_engine_info_serialization()
    test_generation_result_is_shared_type()
    test_baseline_engine_generates_via_service()
    test_baseline_engine_model_info_and_properties()
    test_baseline_engine_does_not_support_streaming()
    test_baseline_engine_load_model_wiring()
    test_engine_works_through_generate_endpoint()
    test_llamacpp_engine_identity_and_default_path()
    test_llamacpp_engine_unloaded_reports_metadata_without_model()
    test_llamacpp_engine_load_model_wiring()
    test_llamacpp_engine_generate_returns_shared_result()
    test_llamacpp_engine_stream_generate_yields_chunks()
    test_llamacpp_engine_error_handling()
