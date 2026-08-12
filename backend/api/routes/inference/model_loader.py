"""Model loading for ArmInferX.

Responsibility: load a Hugging Face tokenizer + causal LM from local disk,
exactly once, with configurable device / dtype / memory limits, and raise
clean, typed errors on failure. This module knows nothing about HTTP or
benchmarking — it is purely a loading concern.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging as hf_logging

from api.utils.exceptions import ModelLoadError

logger = logging.getLogger(__name__)

# tqdm progress bars crash on Windows when stdout is redirected (e.g. uvicorn
# started from a non-interactive shell), so disable them at import time.
hf_logging.disable_progress_bar()

# Default baseline model downloaded by scripts/download_model.py. The repo
# root is 5 levels up from this module (backend/api/routes/inference/).
DEFAULT_MODEL_DIR = (
    Path(__file__).resolve().parents[4]
    / "models"
    / "downloaded"
    / "qwen2.5-3b-instruct"
)

_VALID_DTYPES = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


@dataclass(frozen=True)
class InferenceModel:
    """Immutable bundle of the loaded tokenizer + model handed to the service.

    Note: ``device`` is the *requested* device. When disk offloading is active
    (``max_cpu_memory`` set), parameters live on a mix of cpu + disk.
    """

    tokenizer: AutoTokenizer
    model: AutoModelForCausalLM
    model_id: str
    device: str
    dtype: torch.dtype


def _resolve_dtype(name: str) -> torch.dtype:
    key = name.strip().lower()
    if key not in _VALID_DTYPES:
        raise ModelLoadError(
            f"Unsupported dtype '{name}'. Valid options: {', '.join(sorted(_VALID_DTYPES))}."
        )
    return _VALID_DTYPES[key]


def load_inference_model(
    model_dir: str | Path | None = None,
    *,
    device: str = "cpu",
    dtype: str = "float16",
    max_cpu_memory: str | None = "3GiB",
) -> InferenceModel:
    """Load the tokenizer and model from disk into memory.

    Args:
        model_dir: Path to the Hugging Face model directory. Defaults to
            ``models/downloaded/qwen2.5-3b-instruct`` in the project root.
        device: Target device, e.g. ``"cpu"`` or ``"cuda"``.
        dtype: Weight dtype — ``"float16"``, ``"float32"`` or ``"bfloat16"``.
        max_cpu_memory: Optional cap on CPU RAM used for weights, e.g. ``"3GiB"``.
            Layers beyond the cap are disk-offloaded by accelerate, which keeps
            peak RAM low on small machines (an 8 GB laptop cannot hold the
            fp16 3B weights in RAM). Pass ``None`` or ``""`` to load fully into
            RAM (requires ~6 GB for the 3B model).

    Raises:
        ModelLoadError: If the directory is missing, the dtype is invalid, or
            the tokenizer/model cannot be loaded.
    """
    path = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR
    if not path.is_dir():
        raise ModelLoadError(
            f"Model directory not found: {path}. "
            "Run `scripts/download_model.py` first."
        )

    try:
        torch_dtype = _resolve_dtype(dtype)

        logger.info("Loading tokenizer from %s", path)
        tokenizer = AutoTokenizer.from_pretrained(path)

        logger.info("Loading model from %s (device=%s, dtype=%s)", path, device, dtype)
        kwargs: dict = {"dtype": torch_dtype}
        if device == "cpu" and max_cpu_memory:
            kwargs["device_map"] = "auto"
            kwargs["max_memory"] = {"cpu": max_cpu_memory}
            logger.info(
                "CPU memory capped at %s - excess layers will be disk-offloaded", max_cpu_memory
            )
        elif device != "cpu":
            kwargs["device_map"] = device

        model = AutoModelForCausalLM.from_pretrained(path, **kwargs)
        model.eval()
    except ModelLoadError:
        raise
    except Exception as exc:  # noqa: BLE001 - normalize any load failure
        raise ModelLoadError(f"Failed to load model from {path}: {exc}") from exc

    logger.info("Model loaded: %s (params=%s)", path.name, _format_param_count(model))
    return InferenceModel(
        tokenizer=tokenizer,
        model=model,
        model_id=path.name,
        device=device,
        dtype=torch_dtype,
    )


def _format_param_count(model: AutoModelForCausalLM) -> str:
    total = sum(p.numel() for p in model.parameters())
    return f"{total / 1e9:.2f}B"
