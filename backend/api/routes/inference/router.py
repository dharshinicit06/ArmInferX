"""HTTP router for the inference domain.

Maps the inference domain engines to HTTP endpoints: validates the request
body with Pydantic, resolves the requested engine through the application's
:class:`~engines.manager.EngineManager` (which loads it once and reuses it),
and returns a typed response model.

Engine selection: ``POST /generate`` accepts an optional ``engine_id``
(``llamacpp-optimized``) resolved through the existing engine registry. When
omitted, the application's configured default engine is used. Backward
compatibility: when no engine manager exists on ``app.state`` (e.g. tests that
inject a bare service), the legacy single ``app.state.inference`` engine is
used unchanged.

Streaming: ``POST /generate/stream`` emits Server-Sent Events (``text/event-stream``)
from the engine's ``stream_generate()`` for engines that support streaming
(``llama.cpp``). The final event carries the same metadata fields as the
non-streaming response, so the UI can render progressive tokens and then fill
in engine/runtime/latency/token metadata from one stream.
"""

import json
import logging
import time
from collections.abc import Iterator
from types import SimpleNamespace

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from api.routes.inference.schemas import GenerateRequest, GenerateResponse
from benchmark.benchmark_service import BenchmarkService
from benchmark.metrics import BenchmarkMetrics, SystemSampler, compute_tokens_per_second, utc_now_iso
from benchmark.storage import BaselineResultStore, ResultWriteError
from engines import InferenceEngine
from engines.base import EngineError

logger = logging.getLogger(__name__)

router = APIRouter(tags=["inference"])

# Benchmark + persistence: every /generate request is measured and its record
# saved under results/baseline/. Both are stateless helpers, safe to share.
benchmark_service = BenchmarkService()
result_store = BaselineResultStore()


def get_inference_service(request: Request, engine_id: str | None = None) -> InferenceEngine:
    """Resolve the engine to use for a request.

    Preferred path: ``app.state.engine_manager`` (lazy, cached per engine).
    Legacy path (tests / pre-manager apps): ``app.state.inference`` — one
    pre-loaded engine, used exactly as before.

    Raises:
        HTTPException 400: unknown ``engine_id`` (manager path).
        HTTPException 503: no engine available at all.
    """
    manager = getattr(request.app.state, "engine_manager", None)
    if manager is not None:
        # Unknown ids raise UnknownEngineError, mapped to a clean 400 by the
        # registered exception handlers (never a traceback).
        return manager.get(engine_id)

    engine = getattr(request.app.state, "inference", None)
    if engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference service unavailable: model not loaded",
        )
    return engine


def _engine_identity(engine: InferenceEngine) -> dict:
    """Engine identity fields to surface, or {} for engines without them."""
    identity = {}
    engine_id = getattr(engine, "engine_id", None)
    runtime = getattr(engine, "runtime", None)
    if engine_id:
        identity["engine_id"] = engine_id
    if runtime:
        identity["runtime"] = runtime
    return identity


def _persist_baseline_result(prompt: str, result, metrics, engine: InferenceEngine) -> None:
    """Save the benchmark record to results/baseline/ as a unique JSON file.

    Records are tagged with the engine identity when the engine exposes it, so
    runs from different engines are never mixed without an identifier. Old
    baseline records (no engine fields) are unaffected. File-writing errors are
    handled here: the record is logged as lost and the request continues.
    """
    record = {
        "prompt": prompt,
        "model": result.model_id,
        "response": result.generated_text,
        "latency_ms": result.latency_ms,
        "ttft_ms": metrics.ttft_ms,
        "memory_mb": metrics.memory_mb,
        "cpu_percent": metrics.cpu_percent,
        "generated_tokens": metrics.generated_tokens,
        "tokens_per_second": metrics.tokens_per_second,
        "timestamp": metrics.timestamp,
        **_engine_identity(engine),
    }
    try:
        result_store.save(record)
    except ResultWriteError:
        logger.exception("Failed to persist benchmark result to %s", result_store.root_dir)


@router.post(
    "/generate",
    response_model=GenerateResponse,
    response_model_exclude_none=True,
    summary="Generate a completion",
    description=(
        "Runs the selected engine on the given prompt and returns the generated "
        "text plus engine metadata: inference latency (ms), generated token "
        "count, tokens/second and time-to-first-token (when the engine reports "
        "it). The optional engine_id selects the runtime (llamacpp-optimized); "
        "omitted means the configured default engine. Engines are loaded once "
        "and reused. Empty or whitespace-only prompts are rejected with a 400; "
        "unknown engine ids with a 400."
    ),
    responses={
        status.HTTP_400_BAD_REQUEST: {
            "description": "Empty/whitespace prompt or unknown engine_id",
        },
        status.HTTP_422_UNPROCESSABLE_CONTENT: {
            "description": "Invalid request body (e.g. missing prompt)",
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "description": "Generation or engine load failed at runtime",
        },
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "description": "Inference service unavailable",
        },
    },
)
def generate(payload: GenerateRequest, request: Request) -> GenerateResponse:
    # Resolve the engine (selection / lazy load / cache), then measure
    # latency/memory/CPU around the exact call we perform and persist the
    # benchmark record. Persistence failures are logged and never fail the
    # request.
    engine = get_inference_service(request, payload.engine_id)
    result, metrics = benchmark_service.measure(
        lambda: engine.generate(payload.prompt)
    )
    _persist_baseline_result(payload.prompt, result, metrics, engine)
    logger.info(
        "Generated %d tokens on %s (engine=%s)",
        result.generated_tokens,
        result.model_id,
        getattr(engine, "engine_id", "legacy"),
    )

    # Token/timing metadata is surfaced only for engines that identify
    # themselves (registered engines). The legacy path (bare service with no
    # engine_id) keeps its exact original response shape for compatibility.
    identity = _engine_identity(engine)
    extended = dict(identity)
    if identity:
        extended["generated_tokens"] = result.generated_tokens
        extended["tokens_per_second"] = compute_tokens_per_second(
            result.generated_tokens, result.latency_ms
        )
        extended["ttft_ms"] = result.ttft_ms
    return GenerateResponse(
        status="success",
        model=result.model_id,
        response=result.generated_text,
        latency_ms=result.latency_ms,
        **extended,
    )


def _sse(data: dict) -> str:
    """Serialize one Server-Sent Event (single data field, JSON payload)."""
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post(
    "/generate/stream",
    summary="Stream a completion (Server-Sent Events)",
    description=(
        "Like POST /generate but streams the response as text/event-stream: "
        "one 'data:' event per generated token ({\"text\": ...}) followed by a "
        "final metadata event ({\"done\": true, engine_id, runtime, model, "
        "latency_ms, generated_tokens, tokens_per_second, ttft_ms}). Only "
        "engines with streaming support (llamacpp-optimized) can be streamed; "
        "others return a 400. Errors are reported as an {\"error\": ...} event, "
        "never as a traceback."
    ),
    responses={
        status.HTTP_400_BAD_REQUEST: {
            "description": "Unknown engine_id or engine does not support streaming",
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "description": "Engine load failed at runtime",
        },
    },
)
def generate_stream(payload: GenerateRequest, request: Request) -> StreamingResponse:
    engine = get_inference_service(request, payload.engine_id)

    if not getattr(engine, "supports_streaming", False):
        engine_id = getattr(engine, "engine_id", "unknown")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Engine '{engine_id}' does not support streaming generation; "
                "use POST /generate"
            ),
        )

    identity = _engine_identity(engine)

    def event_source() -> Iterator[str]:
        started = time.perf_counter()
        sampler = SystemSampler()
        token_count = 0
        ttft_ms: float | None = None
        chunks: list[str] = []
        try:
            for chunk in engine.stream_generate(payload.prompt):
                if chunk.is_first and ttft_ms is None:
                    ttft_ms = (time.perf_counter() - started) * 1000.0
                if chunk.text:
                    token_count += 1
                    chunks.append(chunk.text)
                    yield _sse(
                        {
                            "text": chunk.text,
                            "is_first": chunk.is_first,
                            "is_last": chunk.is_last,
                        }
                    )
            latency_ms = (time.perf_counter() - started) * 1000.0
            # Mirror /generate: every inference is measured and auto-saved as a
            # benchmark record (tagged with the engine) so the latest-run panel
            # stays accurate after streamed chats.
            metrics = BenchmarkMetrics(
                timestamp=utc_now_iso(),
                latency_ms=latency_ms,
                memory_mb=sampler.memory_mb(),
                cpu_percent=sampler.cpu_percent(),
                generated_tokens=token_count,
                tokens_per_second=compute_tokens_per_second(token_count, latency_ms),
                ttft_ms=ttft_ms,
            )
            _persist_stream_result(
                payload.prompt,
                "".join(chunks),
                _model_label(engine, payload),
                metrics,
                engine,
            )
            yield _sse(
                {
                    "done": True,
                    **identity,
                    "model": _model_label(engine, payload),
                    "latency_ms": round(latency_ms, 2),
                    "generated_tokens": token_count,
                    "tokens_per_second": compute_tokens_per_second(
                        token_count, latency_ms
                    ),
                    "ttft_ms": round(ttft_ms, 2) if ttft_ms is not None else None,
                }
            )
        except EngineError as exc:
            # Typed engine failures (e.g. generation failed mid-stream) are
            # reported as a clean event — never a raw traceback to the client.
            logger.exception("Streaming generation failed")
            yield _sse({"error": str(exc)})

    return StreamingResponse(event_source(), media_type="text/event-stream")


def _persist_stream_result(
    prompt: str,
    text: str,
    model_id: str,
    metrics: BenchmarkMetrics,
    engine: InferenceEngine,
) -> None:
    """Auto-save a benchmark record for a streamed generation (mirrors /generate)."""
    result = SimpleNamespace(
        model_id=model_id,
        generated_text=text,
        latency_ms=metrics.latency_ms,
    )
    _persist_baseline_result(prompt, result, metrics, engine)


def _model_label(engine: InferenceEngine, payload: GenerateRequest) -> str:
    """Best-effort model label for the streaming final event."""
    info = getattr(engine, "get_model_info", None)
    if info is not None:
        try:
            return info().model_id or ""
        except Exception:  # noqa: BLE001 - a metadata hiccup must not break streaming
            return ""
    return ""
