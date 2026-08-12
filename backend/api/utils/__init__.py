"""Shared cross-cutting utilities for the ArmInferX API."""

from api.utils.error_handlers import register_exception_handlers
from api.utils.exceptions import (
    ArmInferXError,
    GenerationError,
    InferenceServiceError,
    InvalidGenerationParamsError,
    InvalidPromptError,
    ModelLoadError,
)
from api.utils.logging_config import configure_logging
from api.utils.timing import Stopwatch

__all__ = [
    "ArmInferXError",
    "GenerationError",
    "InferenceServiceError",
    "InvalidGenerationParamsError",
    "InvalidPromptError",
    "ModelLoadError",
    "Stopwatch",
    "configure_logging",
    "register_exception_handlers",
]
