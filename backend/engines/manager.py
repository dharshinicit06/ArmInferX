"""Engine lifecycle manager for the HTTP application.

The application may run different runtimes (``llamacpp-optimized`` today,
``transformers-baseline`` on machines where it is feasible). Models are heavy
(an LLM weights file), so engines are loaded **lazily on first use** and then
**reused for every subsequent request** — never re-loaded per request, and
never eagerly at startup (which would force-load a model the machine may not
have room for).

``EngineManager`` is the single holder of loaded engine instances. It resolves
``engine_id`` values through the existing registry (:mod:`engines.registry`)
and applies per-engine load kwargs supplied at construction time (e.g. the
baseline's model directory / device / dtype). Thread-safe so concurrent first
requests cannot double-load a model.
"""

from __future__ import annotations

import threading
from typing import Any

from engines.base import EngineInfo, InferenceEngine
from engines.registry import UnknownEngineError, available_engines, get_engine_class, load_engine


class EngineManager:
    """Lazy, cached access to inference engines by ``engine_id``."""

    def __init__(
        self,
        default_engine_id: str = "llamacpp-optimized",
        engine_kwargs: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        if not isinstance(default_engine_id, str) or not default_engine_id.strip():
            raise ValueError("default_engine_id must be a non-empty string")
        self._default_engine_id = default_engine_id
        self._engine_kwargs = engine_kwargs or {}
        self._loaded: dict[str, InferenceEngine] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def default_engine_id(self) -> str:
        """Engine used when a request does not specify one."""
        return self._default_engine_id

    def resolve(self, engine_id: str | None) -> str:
        """Normalize a request's engine choice to a concrete engine id.

        ``None`` (or empty) falls back to the configured default. Unknown ids
        are rejected here with :class:`UnknownEngineError` so the HTTP layer
        can map them to a clean client error.
        """
        chosen = engine_id if engine_id else self._default_engine_id
        if chosen not in available_engines():
            raise UnknownEngineError(
                f"Unknown engine id {chosen!r}. "
                f"Available engines: {', '.join(available_engines())}"
            )
        return chosen

    def get(self, engine_id: str | None) -> InferenceEngine:
        """Return a ready engine for ``engine_id``, loading it once on first use.

        Loaded instances are cached; repeated calls reuse the same engine, so
        the model is loaded exactly once per process.

        Raises:
            UnknownEngineError: Unregistered engine id.
            EngineError: The engine's ``load_model`` failed (typed load error).
        """
        chosen = self.resolve(engine_id)
        with self._lock:
            engine = self._loaded.get(chosen)
            if engine is None:
                engine = load_engine(chosen, **self._engine_kwargs.get(chosen, {}))
                self._loaded[chosen] = engine
            return engine

    def snapshot(self) -> dict[str, dict]:
        """Metadata of the engines loaded so far (never triggers a load).

        Returns ``engine_id -> EngineInfo.to_dict()`` for every engine that has
        actually been loaded. Engines that were never requested are absent, so
        callers can show the honest startup state (nothing loaded yet).
        """
        return {
            engine_id: engine.get_model_info().to_dict()
            for engine_id, engine in self._loaded.items()
        }

    def info(self, engine_id: str | None = None) -> EngineInfo | None:
        """Metadata for one engine without forcing a load, or ``None``."""
        if engine_id is None:
            engine_id = self._default_engine_id
        engine = self._loaded.get(engine_id)
        return engine.get_model_info() if engine is not None else None

    def available_ids(self) -> list[str]:
        """All registered engine ids (whether or not they are loaded)."""
        return available_engines()
