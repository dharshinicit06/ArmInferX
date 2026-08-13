"""Inference engine abstraction for ArmInferX.

Every runtime implements the :class:`~engines.base.InferenceEngine` interface.
Today that is a single runtime:

- ``LlamaCppOptimizedEngine`` — the llama.cpp/GGUF runtime (CPU-only),
  exposed through the interface so optimized runs can be benchmarked on
  identical contracts.

Shared types: ``GenerationResult`` (result contract), ``EngineInfo`` and
``StreamChunk`` (metadata/streaming contracts).

Orchestration: :func:`~engines.registry.load_engine` resolves an ``engine_id``
(``llamacpp-optimized``) to a loaded engine for benchmark use; the HTTP
application resolves engines through ``EngineManager``.
"""

from engines.base import (
    EngineError,
    EngineInfo,
    EngineNotImplementedError,
    EngineOperationUnsupportedError,
    InferenceEngine,
    StreamChunk,
)
from engines.llamacpp_optimized import LlamaCppOptimizedEngine
from engines.registry import (
    UnknownEngineError,
    available_engines,
    get_engine_class,
    load_engine,
)
from engines.result import GenerationResult

__all__ = [
    "EngineError",
    "EngineInfo",
    "EngineNotImplementedError",
    "EngineOperationUnsupportedError",
    "GenerationResult",
    "InferenceEngine",
    "LlamaCppOptimizedEngine",
    "StreamChunk",
    "UnknownEngineError",
    "available_engines",
    "get_engine_class",
    "load_engine",
]
