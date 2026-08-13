"""Inference engine interface for ArmInferX.

Every inference runtime (llama.cpp optimized today) is exposed through the
same ``InferenceEngine`` contract so the HTTP layer and the benchmark service
can treat all engines uniformly:

- ``load_model()``       — load the engine's model and return a ready engine.
- ``generate()``         — produce a complete ``GenerationResult`` for a prompt.
- ``stream_generate()``  — stream output chunks (only when supported).
- ``get_model_info()``   — metadata about this engine and its loaded model.

Engines that cannot stream inherit the default ``stream_generate()`` which
raises :class:`EngineOperationUnsupportedError`, so callers can probe
``supports_streaming`` first.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, ClassVar

from engines.result import GenerationResult


class EngineError(RuntimeError):
    """Base error for inference engine operations."""


class EngineOperationUnsupportedError(EngineError):
    """Raised when an engine does not support a requested operation."""


class EngineNotImplementedError(EngineError, NotImplementedError):
    """Raised by engines whose runtime is not implemented yet."""


@dataclass(frozen=True)
class StreamChunk:
    """One unit of streamed generation output.

    ``text`` is the text delta for this chunk; ``token_id`` is the raw token
    when the runtime exposes it. ``is_first``/``is_last`` mark the boundaries
    of a stream (used for TTFT and client-side stream finalization).
    """

    text: str
    token_id: int | None = None
    is_first: bool = False
    is_last: bool = False


@dataclass(frozen=True)
class EngineInfo:
    """Metadata about an inference engine instance (from ``get_model_info``).

    ``loaded`` distinguishes a ready engine from a declared-but-unloaded one
    (e.g. the optimized engine when no GGUF model is present). ``extra`` is a
    free-form slot for runtime-specific details.
    """

    engine_id: str
    runtime: str
    supports_streaming: bool
    model_id: str | None = None
    max_context: int | None = None
    loaded: bool = False
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """JSON-serializable representation."""
        return {
            "engine_id": self.engine_id,
            "runtime": self.runtime,
            "supports_streaming": self.supports_streaming,
            "model_id": self.model_id,
            "max_context": self.max_context,
            "loaded": self.loaded,
            **self.extra,
        }


class InferenceEngine(ABC):
    """Uniform contract implemented by every ArmInferX inference runtime."""

    #: Stable identifier used across the API and engine metadata.
    engine_id: ClassVar[str]
    #: Underlying runtime name (e.g. "transformers", "llama.cpp").
    runtime: ClassVar[str]
    #: Whether this engine can emit tokens incrementally.
    supports_streaming: ClassVar[bool] = False

    @classmethod
    @abstractmethod
    def load_model(cls, **kwargs: Any) -> "InferenceEngine":
        """Load the engine's model and return a ready-to-use engine instance.

        Concrete engines define their own parameters (model path, device,
        dtype, thread counts, ...). Raises a typed load error on failure.
        """

    @abstractmethod
    def generate(self, prompt: str, **kwargs: Any) -> GenerationResult:
        """Generate a complete response for ``prompt``.

        The returned :class:`~engines.result.GenerationResult` carries the
        tokenizer-native token counts, total latency and (when supported)
        time-to-first-token, so the benchmark pipeline stays engine-agnostic.
        """

    def stream_generate(self, prompt: str, **kwargs: Any) -> Iterator[StreamChunk]:
        """Generate incrementally, yielding :class:`StreamChunk` objects.

        Only supported by engines with ``supports_streaming``; the default
        implementation rejects the operation with a clear error.
        """
        raise EngineOperationUnsupportedError(
            f"Engine '{self.engine_id}' ({self.runtime}) does not support "
            "streaming generation"
        )

    @abstractmethod
    def get_model_info(self) -> EngineInfo:
        """Return metadata about this engine instance and its loaded model."""
