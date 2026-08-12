"""Benchmark configuration for the engine-agnostic benchmark layer.

``BenchmarkConfig`` is the single, reusable specification of *how* a benchmark
run is executed. It is deliberately engine-agnostic: the same config drives
both ``TransformersBaselineEngine`` and ``LlamaCppOptimizedEngine`` so the
benchmark procedure is identical and future Arm64 measurements are
scientifically comparable.

Determinism policy
------------------
The config contains no random parameters. ``temperature=None`` means "do not
pass a temperature to the engines" — both engines then use their deterministic
greedy decoding by default (Transformers: ``do_sample=False``; llama.cpp:
``temperature=0.0``). When ``temperature`` is set it must be in ``(0, 2]``
(the range accepted by both runtimes) and is passed verbatim to both engines.
"""

from __future__ import annotations

from dataclasses import dataclass


class BenchmarkConfigError(ValueError):
    """Raised when a :class:`BenchmarkConfig` is invalid."""


@dataclass(frozen=True)
class BenchmarkConfig:
    """Deterministic, engine-agnostic benchmark procedure specification.

    Attributes:
        prompt: The exact prompt string used for every warmup and timed run.
        max_new_tokens: Output length cap passed to both engines.
        temperature: Sampling temperature in ``(0, 2]``, or ``None`` for each
            engine's deterministic greedy default (do not pass temperature).
        chat_template: Whether the Transformers baseline should wrap the prompt
            in the model's chat template. Defaults to ``False`` (raw
            completion) so both engines run on the identical raw prompt — the
            llama.cpp engine does not apply a chat template.
        repeats: Number of timed repetitions. Must be >= 1.
        warmup: Number of untimed warmup calls before the timed runs. Must be
            >= 0. Warmup runs use the exact same prompt and generation kwargs.
    """

    prompt: str
    max_new_tokens: int = 128
    temperature: float | None = None
    chat_template: bool = False
    repeats: int = 5
    warmup: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise BenchmarkConfigError("prompt must be a non-empty string")
        if (
            isinstance(self.max_new_tokens, bool)
            or not isinstance(self.max_new_tokens, int)
            or self.max_new_tokens < 1
        ):
            raise BenchmarkConfigError(
                f"max_new_tokens must be a positive integer, got {self.max_new_tokens!r}"
            )
        if self.temperature is not None and not (
            isinstance(self.temperature, (int, float))
            and 0 < self.temperature <= 2
        ):
            raise BenchmarkConfigError(
                "temperature must be in (0, 2] when set (or None for greedy "
                f"defaults), got {self.temperature!r}"
            )
        if not isinstance(self.chat_template, bool):
            raise BenchmarkConfigError(
                f"chat_template must be a bool, got {self.chat_template!r}"
            )
        if (
            isinstance(self.repeats, bool)
            or not isinstance(self.repeats, int)
            or self.repeats < 1
        ):
            raise BenchmarkConfigError(
                f"repeats must be a positive integer, got {self.repeats!r}"
            )
        if (
            isinstance(self.warmup, bool)
            or not isinstance(self.warmup, int)
            or self.warmup < 0
        ):
            raise BenchmarkConfigError(
                f"warmup must be a non-negative integer, got {self.warmup!r}"
            )
