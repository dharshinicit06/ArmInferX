"""Inference domain: HTTP routing and request/response schemas.

Convenience re-exports so callers can import the public surface from the
package instead of individual modules.
"""

from api.routes.inference.router import router
from api.routes.inference.schemas import GenerateRequest, GenerateResponse

__all__ = [
    "GenerateRequest",
    "GenerateResponse",
    "router",
]
