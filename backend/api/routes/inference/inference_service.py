"""Inference service for ArmInferX.

Responsibility: expose a reusable, validated interface for text generation on
top of a loaded model, isolating callers (HTTP layer, future benchmark engine)
from torch/transformers internals. The service holds the already-loaded model
and is safe to reuse across requests and threads.
"""

from __future__ import annotations

import logging
import threading

import torch

from api.routes.inference.model_loader import InferenceModel
from api.routes.inference.streamer import TTFTStreamer
from engines.result import GenerationResult  # shared engine result contract
from api.utils.exceptions import (
    GenerationError,
    InferenceServiceError,
    InvalidGenerationParamsError,
    InvalidPromptError,
)
from api.utils.timing import Stopwatch

logger = logging.getLogger(__name__)

DEFAULT_MAX_NEW_TOKENS = 128


class InferenceService:
    """Reusable text-generation service bound to one loaded model."""

    def __init__(self, inference_model: InferenceModel) -> None:
        self._model = inference_model.model
        self._tokenizer = inference_model.tokenizer
        self._model_id = inference_model.model_id
        self._max_context = getattr(
            self._model.config, "max_position_embeddings", 32768
        )
        # Serialize generation: the CPU/disk-offload pipeline is not safe for
        # concurrent generate() calls. Concurrency tuning is a later concern.
        self._lock = threading.Lock()
        logger.info(
            "InferenceService ready for %s (context=%d tokens)",
            self._model_id,
            self._max_context,
        )

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def max_context(self) -> int:
        return self._max_context

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        temperature: float | None = None,
        do_sample: bool | None = None,
        use_chat_template: bool = True,
        **generation_kwargs,
    ) -> GenerationResult:
        """Generate text for ``prompt``.

        The returned result also reports ``ttft_ms`` — the wall-clock time
        from the start of ``model.generate()`` until the first output token
        is produced (measured via an internal streamer; the generated text
        is identical to a non-streamed call).

        Args:
            prompt: Non-empty user prompt.
            max_new_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature (implies sampling when > 0).
            do_sample: Explicitly enable/disable sampling. Defaults to True when
                ``temperature`` is set, else False (greedy).
            use_chat_template: Wrap the prompt in the model's chat template.
            **generation_kwargs: Extra kwargs forwarded to ``model.generate()``
                (e.g. ``top_p``, ``repetition_penalty``).

        Raises:
            InvalidPromptError: Bad or over-length prompt.
            InvalidGenerationParamsError: Bad generation parameters.
            GenerationError: The model failed during generation.
        """
        if not isinstance(prompt, str) or not prompt.strip():
            raise InvalidPromptError("prompt must be a non-empty string")

        do_sample = self._validate_params(
            max_new_tokens, temperature, do_sample, generation_kwargs
        )

        try:
            if use_chat_template:
                if not getattr(self._tokenizer, "chat_template", None):
                    raise InvalidGenerationParamsError(
                        f"Model '{self._model_id}' has no chat template; "
                        "pass use_chat_template=False to generate from a raw prompt"
                    )
                text = self._tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                text = prompt

            inputs = self._tokenizer(text, return_tensors="pt")
            prompt_len = inputs["input_ids"].shape[1]
            if prompt_len + max_new_tokens > self._max_context:
                raise InvalidPromptError(
                    f"Prompt ({prompt_len} tokens) + max_new_tokens ({max_new_tokens}) "
                    f"exceeds the model context of {self._max_context} tokens"
                )

            gen_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": do_sample}
            if temperature is not None:
                gen_kwargs["temperature"] = temperature
            gen_kwargs.update(generation_kwargs)

            logger.info(
                "Generating up to %d tokens on %s (do_sample=%s)",
                max_new_tokens,
                self._model_id,
                do_sample,
            )
            # --- Latency + time-to-first-token measurement ------------------
            # Only the raw model.generate() call is timed. The start/end times
            # are captured with Python's time module (perf_counter) via the
            # Stopwatch utility. Tokenization, chat-template wrapping and
            # decoding are intentionally excluded so latency_ms reflects pure
            # inference time. The TTFTStreamer observes the tokens
            # model.generate() produces and records when the first one is
            # emitted (prefill + first decode step); it never buffers or
            # changes them, so the output is identical to a non-streamed call.
            # CPU/memory sampling and the tokens-per-second throughput metric
            # are added around this call by the benchmark service, using the
            # tokenizer-native token count below.
            with self._lock:
                with torch.no_grad():
                    with Stopwatch() as timer:
                        streamer = TTFTStreamer()
                        output = self._model.generate(
                            **inputs, **gen_kwargs, streamer=streamer
                        )
            latency_ms = timer.latency_ms
            ttft_ms = streamer.ttft_ms
            logger.info(
                "Inference took %.1f ms on %s (ttft=%.1f ms)",
                latency_ms,
                self._model_id,
                ttft_ms if ttft_ms is not None else -1.0,
            )

            generated_ids = output[0][prompt_len:]
            # Token counting is tokenizer-native: ``generated_ids`` are the
            # actual token IDs the model emitted, so ``len(generated_ids)`` is
            # the exact number of generated output tokens (no character/word
            # estimation). This value feeds the tokens-per-second metric.
            generated_text = self._tokenizer.decode(generated_ids, skip_special_tokens=True)
        except (InvalidPromptError, InvalidGenerationParamsError):
            raise
        except Exception as exc:  # noqa: BLE001 - normalize runtime failures
            logger.exception("Generation failed")
            raise GenerationError(f"Generation failed: {exc}") from exc

        return GenerationResult(
            prompt=prompt,
            generated_text=generated_text.strip(),
            model_id=self._model_id,
            prompt_tokens=prompt_len,
            generated_tokens=len(generated_ids),
            latency_ms=latency_ms,
            ttft_ms=ttft_ms,
            generation_kwargs=gen_kwargs,
        )

    @staticmethod
    def _validate_params(
        max_new_tokens: int,
        temperature: float | None,
        do_sample: bool | None,
        extra: dict,
    ) -> bool:
        if (
            isinstance(max_new_tokens, bool)
            or not isinstance(max_new_tokens, int)
            or max_new_tokens < 1
        ):
            raise InvalidGenerationParamsError(
                "max_new_tokens must be a positive integer"
            )
        if temperature is not None and not (
            isinstance(temperature, (int, float)) and temperature > 0
        ):
            raise InvalidGenerationParamsError(
                "temperature must be a number > 0 when provided"
            )
        if do_sample is None:
            do_sample = temperature is not None
        if temperature is not None and not do_sample:
            raise InvalidGenerationParamsError(
                "temperature requires do_sample=True (or leave do_sample unset)"
            )
        for key, value in extra.items():
            if not isinstance(key, str):
                raise InvalidGenerationParamsError(
                    f"generation kwarg names must be strings, got {key!r}"
                )
            if key in {"max_new_tokens", "temperature", "do_sample"}:
                raise InvalidGenerationParamsError(
                    f"'{key}' is a reserved parameter; pass it as a named argument "
                    "so it can be validated"
                )
            if key == "streamer":
                raise InvalidGenerationParamsError(
                    "'streamer' is reserved for internal TTFT measurement"
                )
            if key in {"num_beams", "num_beam_groups"} and isinstance(
                value, (int, float)
            ):
                # TTFT is measured with a token streamer attached to
                # model.generate(); transformers only supports streaming for
                # greedy/sampling decoding, so beam search is rejected with a
                # clear error instead of failing inside generate().
                if value > 1:
                    raise InvalidGenerationParamsError(
                        f"'{key}={value}' is not supported: TTFT measurement uses "
                        "streaming generation, which requires greedy or sampling "
                        "decoding (beam search cannot stream)"
                    )
        return do_sample
