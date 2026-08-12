"""Optimized engine: llama.cpp (GGUF, CPU-only).

``LlamaCppOptimizedEngine`` adapts llama.cpp (via ``llama-cpp-python``) to the
shared :class:`~engines.base.InferenceEngine` contract so the benchmark
pipeline can later compare the Transformers baseline and the llama.cpp runtime
through one uniform interface:

- ``load_model()``      — load a GGUF file with llama.cpp (CPU-only).
- ``generate()``        — complete a prompt, returning the shared
                          :class:`~engines.result.GenerationResult`.
- ``stream_generate()`` — emit tokens incrementally as ``StreamChunk`` objects.
- ``get_model_info()``  — engine/model metadata (``EngineInfo``).

Split GGUF handling
-------------------
The default model is the single-file Q4_K_M GGUF. Split GGUFs (such as the
FP16 two-shard variant) are still supported: llama.cpp auto-discovers sibling
shards from the file naming convention (``-00001-of-00002``) plus the embedded
``split.*`` GGUF metadata, so this engine passes only the primary shard path
and relies on that normal behavior. The files are never concatenated manually.

CPU-only configuration
----------------------
``n_gpu_layers=0`` — no CUDA/Metal/Vulkan/ROCm offload is used. This matches
the verified development environment (llama-cpp-python 0.3.34, CPU build).

No quantization is introduced: the GGUF file is used exactly as provided.
No performance claims are made here — measured comparisons against the
Transformers baseline only become meaningful once the Arm64 benchmark runs.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from llama_cpp import Llama

from engines.base import EngineError, EngineInfo, InferenceEngine, StreamChunk
from engines.result import GenerationResult

logger = logging.getLogger(__name__)

# Project root: backend/engines/llamacpp_optimized.py -> parents[2]
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Default GGUF: the verified single-file Q4_K_M quantized model. Split GGUFs
# remain supported through llama.cpp's normal shard discovery when a different
# path is passed.
DEFAULT_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "gguf"
    / "qwen2.5-3b-instruct-q4_k_m.gguf"
)

# Safe defaults for the current 8 GB development laptop (CPU-only). These are
# deliberately fixed so future benchmark runs are comparable; do not tune yet.
DEFAULT_N_CTX = 2048
DEFAULT_N_THREADS = 8
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_NEW_TOKENS = 64
N_GPU_LAYERS = 0  # CPU-only: never offload to a GPU


class LlamaCppError(EngineError):
    """Base error for the llama.cpp optimized engine."""


class LlamaCppModelLoadError(LlamaCppError):
    """GGUF file is missing or llama.cpp failed to load it."""


class LlamaCppConfigError(LlamaCppError):
    """Invalid engine configuration."""


class LlamaCppInferenceError(LlamaCppError):
    """Generation failed at runtime."""


def _validate_config(
    n_ctx: int,
    n_threads: int,
    temperature: float,
    max_new_tokens: int,
    n_gpu_layers: int,
) -> None:
    """Validate engine configuration, raising :class:`LlamaCppConfigError`."""
    if not isinstance(n_ctx, int) or n_ctx <= 0:
        raise LlamaCppConfigError(f"n_ctx must be a positive integer, got {n_ctx!r}")
    if not isinstance(n_threads, int) or n_threads <= 0:
        raise LlamaCppConfigError(
            f"n_threads must be a positive integer, got {n_threads!r}"
        )
    if not isinstance(max_new_tokens, int) or max_new_tokens <= 0:
        raise LlamaCppConfigError(
            f"max_new_tokens must be a positive integer, got {max_new_tokens!r}"
        )
    if not isinstance(temperature, (int, float)) or not 0.0 <= temperature <= 2.0:
        raise LlamaCppConfigError(
            f"temperature must be between 0 and 2, got {temperature!r}"
        )
    if not isinstance(n_gpu_layers, int) or n_gpu_layers < 0:
        raise LlamaCppConfigError(
            f"n_gpu_layers must be a non-negative integer, got {n_gpu_layers!r}"
        )


class LlamaCppOptimizedEngine(InferenceEngine):
    """Optimized engine backed by llama.cpp (GGUF models, CPU-only)."""

    engine_id = "llamacpp-optimized"
    runtime = "llama.cpp"
    supports_streaming = True  # llama.cpp natively streams tokens

    def __init__(
        self,
        llm: Llama | None = None,
        *,
        model_path: str | Path | None = None,
        model_id: str | None = None,
        n_ctx: int = DEFAULT_N_CTX,
        n_threads: int = DEFAULT_N_THREADS,
        temperature: float = DEFAULT_TEMPERATURE,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    ) -> None:
        self._llm = llm
        self._model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        self._model_id = model_id
        self._n_ctx = n_ctx
        self._n_threads = n_threads
        self._temperature = temperature
        self._max_new_tokens = max_new_tokens

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    @classmethod
    def load_model(
        cls,
        model_path: str | Path | None = None,
        *,
        n_ctx: int = DEFAULT_N_CTX,
        n_threads: int = DEFAULT_N_THREADS,
        temperature: float = DEFAULT_TEMPERATURE,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        n_gpu_layers: int = N_GPU_LAYERS,
        **kwargs: Any,
    ) -> "LlamaCppOptimizedEngine":
        """Load a GGUF model with llama.cpp and return a ready engine.

        Split GGUFs are handled by llama.cpp's normal shard discovery; only the
        primary shard path is required. Extra keyword arguments are accepted
        for interface uniformity but are not used.

        Raises:
            LlamaCppConfigError: invalid configuration.
            LlamaCppModelLoadError: model file missing or llama.cpp load failed
                (path / GGUF / memory / runtime problem).
        """
        _validate_config(n_ctx, n_threads, temperature, max_new_tokens, n_gpu_layers)

        path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        if not path.is_file():
            raise LlamaCppModelLoadError(
                f"GGUF model file not found: {path}. Place the primary shard "
                "(and its sibling shards) under models/gguf/ before loading."
            )

        logger.info(
            "Loading GGUF %s (n_ctx=%d, n_threads=%d, n_gpu_layers=%d) ...",
            path.name,
            n_ctx,
            n_threads,
            n_gpu_layers,
        )
        try:
            llm = Llama(
                model_path=str(path),
                n_ctx=n_ctx,
                n_threads=n_threads,
                n_threads_batch=n_threads,
                n_gpu_layers=n_gpu_layers,
                verbose=False,
            )
        except MemoryError as exc:
            raise LlamaCppModelLoadError(
                f"Not enough memory to load {path.name} "
                "(this machine has 7.6 GiB total)."
            ) from exc
        except Exception as exc:  # noqa: BLE001 - normalize any load failure
            raise LlamaCppModelLoadError(
                f"llama.cpp failed to load {path.name}: {exc}"
            ) from exc

        model_id = cls._derive_model_id(path)
        logger.info("GGUF model loaded: %s (%s)", model_id, path.name)
        return cls(
            llm=llm,
            model_path=path,
            model_id=model_id,
            n_ctx=n_ctx,
            n_threads=n_threads,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )

    @staticmethod
    def _derive_model_id(path: Path) -> str:
        """Derive a stable model id from the GGUF filename.

        Strips a split-shard suffix when present, so both shards of
        ``qwen2.5-3b-instruct-fp16-00001-of-00002.gguf`` identify as
        ``qwen2.5-3b-instruct-fp16``; single-file models such as the default
        ``qwen2.5-3b-instruct-q4_k_m.gguf`` keep their full stem.
        """
        return re.sub(r"-\d{5}-of-\d{5}$", "", path.stem)

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    def _completion_stream(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        **generation_kwargs: Any,
    ) -> Iterator[tuple[str, float]]:
        """Yield ``(text_delta, seconds_since_start)`` per emitted token.

        llama.cpp streams exactly one token per chunk (plus a final empty
        chunk), so each non-empty delta counts as one generated token — a
        tokenizer-native count, never a char/word estimate. Extra keyword
        arguments are forwarded to ``create_completion`` (e.g. ``top_p``,
        ``repeat_penalty``, ``stop``), mirroring the baseline which forwards
        generation kwargs to the underlying runtime.
        """
        start = time.perf_counter()
        for chunk in self._llm.create_completion(  # type: ignore[union-attr]
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
            **generation_kwargs,
        ):
            try:
                delta = chunk["choices"][0]["text"]
            except (KeyError, IndexError, TypeError) as exc:
                raise LlamaCppInferenceError(
                    f"Unexpected streaming chunk from llama.cpp: {chunk!r}"
                ) from exc
            if delta:
                yield delta, time.perf_counter() - start

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        **generation_kwargs: Any,
    ) -> GenerationResult:
        """Generate a complete response for ``prompt``.

        Uses llama.cpp streaming internally so time-to-first-token is measured
        (mirroring the baseline's TTFT semantics). Token counts are
        tokenizer-native: prompt tokens via ``Llama.tokenize``, completion
        tokens as the number of emitted stream chunks.
        """
        if self._llm is None:
            raise LlamaCppError(
                f"Engine '{self.engine_id}' has no loaded model; "
                "call load_model() before generate()."
            )
        if not prompt or not prompt.strip():
            raise LlamaCppConfigError("prompt must be a non-empty string")

        n_max = self._max_new_tokens if max_new_tokens is None else max_new_tokens
        temp = self._temperature if temperature is None else temperature
        _validate_config(self._n_ctx, self._n_threads, temp, n_max, N_GPU_LAYERS)

        deltas: list[str] = []
        ttft_ms: float | None = None
        started = time.perf_counter()
        try:
            for delta, elapsed in self._completion_stream(
                prompt, n_max, temp, **generation_kwargs
            ):
                if ttft_ms is None:
                    ttft_ms = elapsed * 1000.0
                deltas.append(delta)
        except LlamaCppError:
            raise
        except Exception as exc:  # noqa: BLE001 - normalize any generation failure
            raise LlamaCppInferenceError(
                f"llama.cpp generation failed: {exc}"
            ) from exc
        latency_ms = (time.perf_counter() - started) * 1000.0

        text = "".join(deltas)
        prompt_tokens = len(self._llm.tokenize(prompt.encode("utf-8")))  # type: ignore[union-attr]
        completion_tokens = len(deltas)

        logger.debug(
            "generate(): %d prompt tokens, %d generated tokens, %.1f ms, TTFT %.1f ms",
            prompt_tokens,
            completion_tokens,
            latency_ms,
            ttft_ms if ttft_ms is not None else -1.0,
        )
        return GenerationResult(
            prompt=prompt,
            generated_text=text,
            model_id=self._model_id or self._derive_model_id(self._model_path),
            prompt_tokens=prompt_tokens,
            generated_tokens=completion_tokens,
            latency_ms=latency_ms,
            ttft_ms=ttft_ms,
            generation_kwargs={
                "max_new_tokens": n_max,
                "temperature": temp,
                **generation_kwargs,
            },
        )

    def stream_generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        **generation_kwargs: Any,
    ) -> Iterator[StreamChunk]:
        """Generate incrementally, yielding :class:`StreamChunk` objects."""
        if self._llm is None:
            raise LlamaCppError(
                f"Engine '{self.engine_id}' has no loaded model; "
                "call load_model() before stream_generate()."
            )
        if not prompt or not prompt.strip():
            raise LlamaCppConfigError("prompt must be a non-empty string")

        n_max = self._max_new_tokens if max_new_tokens is None else max_new_tokens
        temp = self._temperature if temperature is None else temperature
        _validate_config(self._n_ctx, self._n_threads, temp, n_max, N_GPU_LAYERS)

        is_first = True
        try:
            for delta, _elapsed in self._completion_stream(
                prompt, n_max, temp, **generation_kwargs
            ):
                yield StreamChunk(text=delta, is_first=is_first)
                is_first = False
        except LlamaCppError:
            raise
        except Exception as exc:  # noqa: BLE001 - normalize any streaming failure
            raise LlamaCppInferenceError(
                f"llama.cpp streaming failed: {exc}"
            ) from exc
        yield StreamChunk(text="", is_last=True)

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------
    def get_model_info(self) -> EngineInfo:
        """Metadata about this engine instance and its loaded model."""
        return EngineInfo(
            engine_id=self.engine_id,
            runtime=self.runtime,
            supports_streaming=self.supports_streaming,
            model_id=self._model_id,
            max_context=self._n_ctx,
            loaded=self._llm is not None,
            extra={
                "model_path": str(self._model_path),
                "n_threads": self._n_threads,
                "n_gpu_layers": N_GPU_LAYERS,
                "temperature": self._temperature,
                "max_new_tokens": self._max_new_tokens,
            },
        )
