"""ArmInferX - standalone llama.cpp smoke test for Qwen2.5-3B-Instruct GGUF FP16.

Usage (from the repo root):
    backend/.venv/Scripts/python scripts/test_llamacpp.py

Proves that the split GGUF model (00001-of-00002 / 00002-of-00002) loads and
runs on CPU via llama.cpp before any engine integration. CPU-only, fixed
benchmark configuration, deterministic output.

Exit codes:
    0  all checks passed
    1  any failure (model load, generation, or streaming/TTFT)

Note on split GGUF: llama.cpp auto-discovers the second shard from the
filename convention (-00001-of-00002) plus the embedded split.* metadata.
If the installed runtime cannot load the split files, this script reports the
error and exits non-zero instead of working around it.
"""

import sys
import time

from llama_cpp import Llama

# ---------------------------------------------------------------------------
# Fixed benchmark configuration - keep values stable for comparability.
# Do not tune these yet.
# ---------------------------------------------------------------------------
MODEL_PATH = r"D:\ArmInferX\models\gguf\qwen2.5-3b-instruct-fp16-00001-of-00002.gguf"
MODEL_NAME = "Qwen2.5-3B-Instruct"
MODEL_FORMAT = "GGUF FP16"
N_CTX = 2048                 # fixed context window
N_THREADS = 8                # fixed CPU thread count (change here, nothing else)
MAX_NEW_TOKENS = 64          # non-streaming completion length
MAX_NEW_TOKENS_STREAM = 16   # streaming run length (TTFT measurement)
TEMPERATURE = 0.0            # deterministic sampling
SEED = 42                    # fixed seed for reproducibility
PROMPT = "Hello! Explain what an AI inference engine is."
STREAM_PROMPT = "Hello! Explain what an AI inference engine is."

# Windows consoles often default to cp1252, which cannot encode tokens this
# model emits (e.g. U+0120). Force UTF-8-safe stdout/stderr.
for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def fail(category: str, message: str, exc: BaseException | None = None) -> None:
    """Print a classified error and exit non-zero."""
    print("\nERROR [%s]: %s" % (category, message), file=sys.stderr)
    if exc is not None:
        print("Detail: %s: %s" % (type(exc).__name__, exc), file=sys.stderr)
    print("# Status: FAIL", file=sys.stderr)
    sys.exit(1)


def load_model() -> Llama:
    """Load the GGUF (CPU-only). Classify failures: path / GGUF / memory / runtime."""
    try:
        llm = Llama(
            model_path=MODEL_PATH,
            n_ctx=N_CTX,
            n_threads=N_THREADS,
            n_threads_batch=N_THREADS,
            n_gpu_layers=0,   # explicit CPU-only: no CUDA/Metal/Vulkan/ROCm
            seed=SEED,
            verbose=False,
        )
    except FileNotFoundError as exc:
        fail("path problem", "model file not found: %s" % MODEL_PATH, exc)
    except MemoryError as exc:
        fail("memory problem", "not enough memory to allocate the model", exc)
    except Exception as exc:
        msg = str(exc).lower()
        if "gguf" in msg or "magic" in msg or "invalid" in msg or "corrupt" in msg:
            fail("GGUF problem", "model file is not a readable GGUF: %s" % exc, exc)
        fail("runtime problem", "failed to load model: %s" % exc, exc)
    return llm


def main() -> None:
    print("=" * 40)
    print("ArmInferX - llama.cpp Standalone Test")
    print("=" * 40)
    print()
    print("Runtime: llama.cpp")
    print("Model: %s" % MODEL_NAME)
    print("Format: %s" % MODEL_FORMAT)
    print("Device: CPU")
    print("Context: %d" % N_CTX)
    print("Threads: %d" % N_THREADS)
    print()

    # ---- 1. Load the model (includes split-shard discovery) ----------------
    t0 = time.perf_counter()
    llm = load_model()
    load_time = time.perf_counter() - t0
    print("Model load time: %.2f s" % load_time)
    print()

    # ---- 2. Deterministic non-streaming completion -------------------------
    t0 = time.perf_counter()
    try:
        result = llm.create_completion(
            PROMPT,
            max_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            seed=SEED,
            stream=False,
        )
    except Exception as exc:
        fail("runtime problem", "generation failed: %s" % exc, exc)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    try:
        text = result["choices"][0]["text"]
        usage = result.get("usage") or {}
    except (KeyError, IndexError) as exc:
        fail("runtime problem", "unexpected completion response shape: %s" % exc, exc)

    # Tokenizer-native counts from llama.cpp usage info (no char/word estimates).
    prompt_tokens = usage.get("prompt_tokens")
    if prompt_tokens is None:
        prompt_tokens = len(llm.tokenize(PROMPT.encode("utf-8")))
    completion_tokens = usage.get("completion_tokens")
    if completion_tokens is None:
        completion_tokens = len(llm.tokenize(text.encode("utf-8")))
    total_tokens = usage.get("total_tokens") or (prompt_tokens + completion_tokens)

    tokens_per_sec = (completion_tokens / (latency_ms / 1000.0)) if latency_ms > 0 else 0.0

    # ---- 3. Streaming run: measure TTFT (first token/chunk) ----------------
    t0 = time.perf_counter()
    first_token_at = None
    try:
        for chunk in llm.create_completion(
            STREAM_PROMPT,
            max_tokens=MAX_NEW_TOKENS_STREAM,
            temperature=TEMPERATURE,
            seed=SEED,
            stream=True,
        ):
            delta = chunk["choices"][0]["text"]
            if first_token_at is None and delta:
                first_token_at = time.perf_counter() - t0
    except Exception as exc:
        fail("runtime problem", "streaming generation failed: %s" % exc, exc)
    ttft_ms = (first_token_at * 1000.0) if first_token_at is not None else None
    if ttft_ms is None:
        fail("runtime problem", "streaming returned no generated tokens; TTFT undefined")

    # ---- 4. Report ----------------------------------------------------------
    print("Prompt:")
    print(PROMPT)
    print()
    print("Prompt tokens: %d" % prompt_tokens)
    print("Completion tokens: %d" % completion_tokens)
    print("Total tokens: %d" % total_tokens)
    print()
    print("Generated text:")
    print(text)
    print()
    print("Total latency: %.2f ms" % latency_ms)
    print("Tokens/sec: %.2f" % tokens_per_sec)
    print("TTFT: %.2f ms" % ttft_ms)
    print()
    print("# Status: PASS")


if __name__ == "__main__":
    main()
