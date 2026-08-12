"""Inference domain: model loading, text generation, and HTTP routing.

Convenience re-exports so callers can import the public surface from the
package instead of individual modules.
"""

from api.routes.inference.inference_service import GenerationResult, InferenceService
from api.routes.inference.model_loader import (
    DEFAULT_MODEL_DIR,
    InferenceModel,
    load_inference_model,
)
from api.routes.inference.router import router
from api.routes.inference.schemas import GenerateRequest, GenerateResponse

__all__ = [
    "DEFAULT_MODEL_DIR",
    "GenerateRequest",
    "GenerateResponse",
    "GenerationResult",
    "InferenceModel",
    "InferenceService",
    "load_inference_model",
    "router",
]
