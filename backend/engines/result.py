"""Shared result contract for inference engines.

``GenerationResult`` is the canonical outcome object every engine's
``generate()`` returns. It was originally defined in the baseline inference
service; it lives here now so the engine interface (and future runtimes such
as llama.cpp) share one type without importing from the HTTP route package.
The baseline service re-exports it, so existing imports keep working.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class GenerationResult:
    """Outcome of a successful generation call.

    Fields:
        prompt: The exact prompt that was generated from.
        generated_text: The model's output text.
        model_id: Identifier of the model that produced the response.
        prompt_tokens: Number of tokens in the prompt (tokenizer-native).
        generated_tokens: Number of output tokens the model generated
            (tokenizer-native count of the emitted token IDs).
        latency_ms: Total inference time in milliseconds (model generation
            only, not tokenization/decoding).
        ttft_ms: Time-to-first-token in milliseconds, or ``None`` when the
            engine does not measure it.
        generation_kwargs: The validated generation parameters used.
    """

    prompt: str
    generated_text: str
    model_id: str
    prompt_tokens: int
    generated_tokens: int
    latency_ms: float
    ttft_ms: float | None = None
    generation_kwargs: dict = field(default_factory=dict)
