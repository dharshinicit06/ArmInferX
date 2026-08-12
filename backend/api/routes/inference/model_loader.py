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

import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging as hf_logging

from api.utils.exceptions import ModelLoadError

# Minimum total RAM the FP16 baseline needs. The 3B FP16 weights are ~6.5 GB
# and the framework needs headroom on top; the project documents the baseline
# as requiring >= 16 GiB (see docs/optimization.md). Attempting the load below
# this on a 7.63 GiB machine exhausted memory and killed the whole backend
# process (OOM), so the load is refused up front with a clean, honest error.
MIN_RAM_GIB_FOR_FP16_BASELINE = 16.0

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

    # Hardware feasibility guard: the FP16 baseline must not OOM-kill the
    # backend process. Check BEFORE touching the weights so the API returns a
    # clean, friendly error instead of the process dying mid-load.
    if _is_fp16_baseline_load(dtype):
        _assert_fp16_baseline_feasible()

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


def _is_fp16_baseline_load(dtype: str) -> bool:
    """True when this load is the FP16 baseline (the documented heavy case)."""
    return dtype.strip().lower() == "float16"


def _assert_fp16_baseline_feasible() -> None:
    """Refuse the FP16 baseline load on machines that cannot hold it safely.

    Raises:
        ModelLoadError: when total physical RAM is below the documented
            threshold, with a message the API surfaces to the client.
    """
    total_gib = psutil.virtual_memory().total / (1024**3)
    if total_gib >= MIN_RAM_GIB_FOR_FP16_BASELINE:
        return
    raise ModelLoadError(
        "The Transformers FP16 baseline requires at least 16 GiB of RAM to "
        f"load reliably, but this machine has {total_gib:.1f} GiB. Loading it "
        "here would exhaust memory and kill the backend process, so the load "
        "is refused for safety. Use the 'llamacpp-optimized' engine instead, "
        "or run the baseline on a machine with >= 16 GiB RAM."
    )


def _format_param_count(model: AutoModelForCausalLM) -> str:
    total = sum(p.numel() for p in model.parameters())
    return f"{total / 1e9:.2f}B"
