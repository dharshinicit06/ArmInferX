"""Shared, typed exception hierarchy for the ArmInferX API.

Every domain error derives from ``ArmInferXError`` so the HTTP layer and
callers can catch the whole family in one place, while subclasses stay
specific enough for fine-grained handling and clean HTTP mappings.
"""


class ArmInferXError(Exception):
    """Base class for all ArmInferX domain errors."""


class ModelLoadError(ArmInferXError, RuntimeError):
    """Raised when the model or tokenizer cannot be loaded."""


class InferenceServiceError(ArmInferXError):
    """Base class for all inference service errors."""


class InvalidPromptError(InferenceServiceError, ValueError):
    """The prompt is missing, empty, or would exceed the model context."""


class InvalidGenerationParamsError(InferenceServiceError, ValueError):
    """One or more generation parameters are invalid."""


class GenerationError(InferenceServiceError):
    """Model inference failed at runtime."""
