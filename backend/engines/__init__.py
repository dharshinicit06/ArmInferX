"""Inference engine abstraction for ArmInferX.

Every runtime (Transformers baseline today, llama.cpp optimized) implements
the :class:`~engines.base.InferenceEngine` interface:

- ``TransformersBaselineEngine`` — the existing Hugging Face baseline,
  wrapped to conform to the interface (the baseline pipeline itself is
  unchanged).
- ``LlamaCppOptimizedEngine`` — the llama.cpp/GGUF runtime (CPU-only),
  sharing the same interface so baseline vs. optimized runs can be compared
  on identical contracts.

Shared types: ``GenerationResult`` (result contract), ``EngineInfo`` and
``StreamChunk`` (metadata/streaming contracts).

Orchestration: :func:`~engines.registry.load_engine` resolves an ``engine_id``
(``transformers-baseline`` | ``llamacpp-optimized``) to a loaded engine for
benchmark use; the HTTP application is unaffected and keeps using the
Transformers baseline.
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
from engines.transformers_baseline import TransformersBaselineEngine

__all__ = [
    "EngineError",
    "EngineInfo",
    "EngineNotImplementedError",
    "EngineOperationUnsupportedError",
    "GenerationResult",
    "InferenceEngine",
    "LlamaCppOptimizedEngine",
    "StreamChunk",
    "TransformersBaselineEngine",
    "UnknownEngineError",
    "available_engines",
    "get_engine_class",
    "load_engine",
]
