"""Verify the downloaded Qwen2.5-3B-Instruct model and tokenizer load correctly.

Usage (from the repo root):
    backend/.venv/Scripts/python scripts/verify_model.py
    backend/.venv/Scripts/python scripts/verify_model.py --model-dir models/downloaded/qwen2.5-1.5b-instruct

Loads the model from local disk (no network) using low_cpu_mem_usage streaming
so peak RAM stays near the fp16 weight size (~6.2 GB), then runs a short CPU
generation. Prints stage markers so progress can be monitored from a log file.
"""

import argparse
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging as hf_logging

DEFAULT_MODEL_DIR = "models/downloaded/qwen2.5-3b-instruct"

# tqdm progress bars crash on Windows when stdout is redirected (non-interactive shell).
hf_logging.disable_progress_bar()


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        default=DEFAULT_MODEL_DIR,
        help="Path to the downloaded model, relative to the repo root "
        "(default: %(default)s)",
    )
    args = parser.parse_args()
    model_dir = args.model_dir

    log(f"Python {sys.version.split()[0]} | torch {torch.__version__}")
    log(f"Loading tokenizer from {model_dir} ...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    log(f"  tokenizer loaded in {time.time() - t0:.1f}s | vocab_size={tokenizer.vocab_size} "
        f"| pad_token={tokenizer.pad_token!r} (id={tokenizer.pad_token_id})")

    log(f"Loading model (fp16, CPU, disk-offloaded to fit RAM) from {model_dir} ...")
    t1 = time.time()
    # The full fp16 model needs ~6.2 GB RAM; on machines with 8 GB the native
    # allocation segfaults. Capping RAM at 3 GiB makes accelerate disk-offload
    # the rest and stream each layer into RAM during forward passes (slow, but
    # proves correctness without quantizing/optimizing the model).
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        dtype=torch.float16,
        device_map="auto",
        max_memory={"cpu": "3GiB"},
    )
    log(f"  model loaded in {time.time() - t1:.1f}s")
    n_params = sum(p.numel() for p in model.parameters())
    log(f"  parameters={n_params / 1e9:.2f}B | dtype={model.dtype} | device={model.device}")

    prompt = "Write a short introduction to the city of Tokyo."
    t2 = time.time()
    inputs = tokenizer(prompt, return_tensors="pt")
    log(f"Generating up to 8 tokens (CPU, disk-offloaded - expect it to be slow) ...")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=8, do_sample=False)
    log(f"  generated {out.shape[1] - inputs['input_ids'].shape[1]} tokens in {time.time() - t2:.1f}s")
    print("--- generated text ---", flush=True)
    print(tokenizer.decode(out[0], skip_special_tokens=True), flush=True)
    print("----------------------", flush=True)

    print("\nVERIFICATION PASSED - model and tokenizer load and run correctly.", flush=True)


if __name__ == "__main__":
    main()
