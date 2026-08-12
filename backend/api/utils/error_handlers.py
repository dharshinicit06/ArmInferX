"""HTTP error mapping for the ArmInferX API.

Translates the domain exception hierarchy (:mod:`api.utils.exceptions`) and the
engine error family (:mod:`engines.base`) into clean JSON error responses, and
provides a catch-all handler for unexpected failures so clients always receive
a structured body — engine failures (e.g. a GGUF that cannot be loaded) are
reported as messages, never as raw Python tracebacks.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from engines.base import EngineError
from engines.registry import UnknownEngineError

from api.utils.exceptions import (
    InferenceServiceError,
    InvalidGenerationParamsError,
    InvalidPromptError,
    ModelLoadError,
)

logger = logging.getLogger(__name__)

# Domain error -> HTTP status code.
_ERROR_STATUS = {
    InvalidPromptError: 400,
    InvalidGenerationParamsError: 422,
    InferenceServiceError: 500,
    # Engine selection/loading/generation failures (typed, client-safe).
    UnknownEngineError: 400,
    EngineError: 500,
}


def register_exception_handlers(app: FastAPI) -> None:
    """Register all API exception handlers on ``app`` (idempotent)."""
    for exc_type, status_code in _ERROR_STATUS.items():
        app.add_exception_handler(exc_type, _make_domain_handler(status_code))
    # ModelLoadError is often an *expected* refusal (e.g. the FP16 baseline on
    # a < 16 GiB machine), not a server bug — respond 503 without the
    # error-level traceback that _make_domain_handler logs for status >= 500.
    app.add_exception_handler(ModelLoadError, _model_load_handler)
    app.add_exception_handler(Exception, _unexpected_handler)


async def _model_load_handler(request: Request, exc: ModelLoadError) -> JSONResponse:
    """Clean 503 for refused/known-failure model loads (no traceback log)."""
    logger.warning("Model load refused or failed: %s", exc)
    return JSONResponse(status_code=503, content={"detail": str(exc)})


def _make_domain_handler(status_code: int):
    """Build a handler for a domain exception mapped to ``status_code``."""

    async def _handler(request: Request, exc: Exception) -> JSONResponse:
        if status_code >= 500:
            logger.exception("Unhandled domain error: %s", exc)
        return JSONResponse(status_code=status_code, content={"detail": str(exc)})

    return _handler


async def _unexpected_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled exception: %s", exc)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})
