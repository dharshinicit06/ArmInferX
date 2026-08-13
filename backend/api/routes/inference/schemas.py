"""Pydantic request/response models for the inference API.

These schemas define the public HTTP contract of the inference endpoints and
carry the OpenAPI documentation (descriptions, examples) shown in Swagger UI.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

PROMPT_MAX_LENGTH = 8192


class GenerateRequest(BaseModel):
    """Request body for ``POST /generate``."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [{"prompt": "Explain Artificial Intelligence"}]
        }
    )

    prompt: str = Field(
        ...,
        min_length=1,
        max_length=PROMPT_MAX_LENGTH,
        description=(
            "User prompt to generate from. Must be a non-empty string of at "
            f"most {PROMPT_MAX_LENGTH} characters."
        ),
    )
    engine_id: str | None = Field(
        None,
        description=(
            "Optional engine to run inference with (resolved through the engine "
            "registry). When omitted, the application's configured default "
            "engine is used. Unknown ids are rejected with a 400."
        ),
    )


class GenerateResponse(BaseModel):
    """Response body for ``POST /generate``."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "status": "success",
                    "model": "qwen2.5-3b-instruct",
                    "response": "Artificial Intelligence (AI) is the simulation "
                    "of human intelligence in machines that are programmed to "
                    "think and learn...",
                    "latency_ms": 1234,
                }
            ]
        }
    )

    status: Literal["success"] = Field(..., description="Outcome of the request.")
    model: str = Field(
        ..., description="Identifier of the model that produced the response."
    )
    response: str = Field(..., description="Generated text.")
    latency_ms: float = Field(
        ...,
        ge=0,
        description=(
            "Engine inference latency in milliseconds. Wall-clock time "
            "(Python time module) measured around the engine's generate() call "
            "only - not the whole request."
        ),
    )
    engine_id: str | None = Field(
        None,
        description=(
            "Engine that produced the response (e.g. 'llamacpp-optimized'). "
            "Null on legacy responses before engine selection existed."
        ),
    )
    runtime: str | None = Field(
        None,
        description=(
            "Underlying runtime name (e.g. 'llama.cpp'). "
            "Null when the engine does not expose it."
        ),
    )
    generated_tokens: int | None = Field(
        None,
        ge=0,
        description=(
            "Number of output tokens the engine generated (tokenizer-native). "
            "Null when unavailable."
        ),
    )
    tokens_per_second: float | None = Field(
        None,
        ge=0,
        description=(
            "Generation throughput: generated_tokens / (latency_ms / 1000). "
            "Null when unavailable."
        ),
    )
    ttft_ms: float | None = Field(
        None,
        ge=0,
        description=(
            "Time-to-first-token in milliseconds. Null when the engine does "
            "not measure it."
        ),
    )
