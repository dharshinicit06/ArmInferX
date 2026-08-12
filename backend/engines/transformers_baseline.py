"""Baseline engine: Hugging Face Transformers.

``TransformersBaselineEngine`` adapts the existing baseline pipeline to the
:class:`~engines.base.InferenceEngine` interface. It does **not** reimplement
any generation logic: it wraps the existing
:class:`~api.routes.inference.inference_service.InferenceService` (validation,
chat templates, latency/TTFT measurement, tokenizer-native token counts) and
the existing model loader, so the baseline behavior is unchanged — the engine
only adds the uniform ``load_model`` / ``generate`` / ``get_model_info``
contract on top.

Note: the baseline service/loader are imported lazily (inside the methods) so
the ``engines`` package can initialize without depending on the HTTP route
package — this avoids a circular import (the baseline service imports the
shared ``GenerationResult`` from this package).
"""

from __future__ import annotations

from engines.base import EngineInfo, InferenceEngine
from engines.result import GenerationResult


class TransformersBaselineEngine(InferenceEngine):
    """Baseline engine backed by Hugging Face Transformers."""

    engine_id = "transformers-baseline"
    runtime = "transformers"
    supports_streaming = False

    def __init__(self, service) -> None:
        self._service = service

    @classmethod
    def load_model(
        cls,
        model_dir: str | None = None,
        *,
        device: str = "cpu",
        dtype: str = "float16",
        max_cpu_memory: str | None = "3GiB",
        **kwargs,
    ) -> "TransformersBaselineEngine":
        """Load the baseline model and build a ready engine.

        Parameters match ``load_inference_model``; extra keyword arguments are
        accepted for interface uniformity but are not used by the baseline.
        """
        from api.routes.inference.inference_service import InferenceService
        from api.routes.inference.model_loader import load_inference_model

        model = load_inference_model(
            model_dir,
            device=device,
            dtype=dtype,
            max_cpu_memory=max_cpu_memory,
        )
        return cls(InferenceService(model))

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        do_sample: bool | None = None,
        use_chat_template: bool = True,
        **generation_kwargs,
    ) -> GenerationResult:
        """Delegate to the baseline inference service.

        ``max_new_tokens=None`` means "use the service's default" so the
        engine never drifts from the baseline behavior.
        """
        kwargs = {}
        if max_new_tokens is not None:
            kwargs["max_new_tokens"] = max_new_tokens
        if temperature is not None:
            kwargs["temperature"] = temperature
        if do_sample is not None:
            kwargs["do_sample"] = do_sample
        return self._service.generate(
            prompt,
            use_chat_template=use_chat_template,
            **kwargs,
            **generation_kwargs,
        )

    def get_model_info(self) -> EngineInfo:
        return EngineInfo(
            engine_id=self.engine_id,
            runtime=self.runtime,
            supports_streaming=self.supports_streaming,
            model_id=self.model_id,
            max_context=self.max_context,
            loaded=True,
        )

    @property
    def model_id(self) -> str:
        """Identifier of the loaded model (convenience passthrough)."""
        return self._service.model_id

    @property
    def max_context(self) -> int:
        """Maximum context length in tokens (convenience passthrough)."""
        return self._service.max_context
