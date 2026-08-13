"""Engine registry for benchmark orchestration.

Maps a stable ``engine_id`` to the concrete :class:`~engines.base.InferenceEngine`
class and exposes a uniform ``load_engine(engine_id, ...)`` helper so the
benchmark layer can load a runtime through the same entry point:

    llamacpp-optimized     -> LlamaCppOptimizedEngine

Add new runtimes here (engine_id -> engine class) to make them benchmarkable.
"""

from __future__ import annotations

from typing import Any

from engines.base import EngineError, InferenceEngine
from engines.llamacpp_optimized import LlamaCppOptimizedEngine


class UnknownEngineError(EngineError):
    """Raised when a requested ``engine_id`` is not registered."""


#: engine_id -> engine class. Add new runtimes here to make them benchmarkable.
ENGINE_REGISTRY: dict[str, type[InferenceEngine]] = {
    "llamacpp-optimized": LlamaCppOptimizedEngine,
}


def available_engines() -> list[str]:
    """Return the sorted list of registered engine ids."""
    return sorted(ENGINE_REGISTRY)


def get_engine_class(engine_id: str) -> type[InferenceEngine]:
    """Resolve an ``engine_id`` to its engine class.

    Raises:
        UnknownEngineError: If the engine id is not registered.
    """
    try:
        return ENGINE_REGISTRY[engine_id]
    except KeyError:
        raise UnknownEngineError(
            f"Unknown engine id {engine_id!r}. "
            f"Available engines: {', '.join(available_engines())}"
        ) from None


def load_engine(engine_id: str, **load_kwargs: Any) -> InferenceEngine:
    """Load an engine by id, forwarding ``load_kwargs`` to its ``load_model``.

    Raises:
        UnknownEngineError: If the engine id is not registered.
    """
    return get_engine_class(engine_id).load_model(**load_kwargs)
