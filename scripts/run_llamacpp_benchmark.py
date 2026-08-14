"""ArmInferX - STEP 9: run the real Q4_K_M benchmark through the existing
benchmark infrastructure.

This is a *driver* only: it composes the existing architecture
(``engines.registry.load_engine`` + ``benchmark.BenchmarkConfig`` +
``benchmark.BenchmarkRunner`` + ``benchmark.EngineResultStore``) and makes no
modifications to it.

Usage (from the repo root):
    backend/.venv/Scripts/python scripts/run_llamacpp_benchmark.py

Procedure (per the STEP 9 spec):
    engine_id      = llamacpp-optimized   (llama.cpp runtime, CPU-only)
    model          = models/gguf/qwen2.5-0.5b-instruct-q4_k_m.gguf (default)
    max_new_tokens = 64
    temperature    = None  -> llama.cpp greedy default (0.0)
    chat_template  = False
    repeats        = 5 timed runs
    warmup         = 1 untimed call
    n_ctx / n_threads / n_gpu_layers / engine defaults: untouched

``prompt_tokens`` is not part of the runner's persisted record schema (it
lives on ``GenerationResult``). To record it without modifying the
architecture, this driver derives the count once via the loaded tokenizer
(the exact computation the engine performs in ``generate()``; deterministic
because the prompt is identical for every run) and injects it into each
record at the storage boundary through a driver-side ``EngineResultStore``
subclass.

Exit codes:
    0  benchmark completed and persisted records verified
    1  any failure (load, generation, persistence, or verification)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Project root = one level above this script (scripts/run_llamacpp_benchmark.py).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"
# The benchmark/engines packages import as top-level packages; put backend on
# the path (mirrors running with cwd=backend).
sys.path.insert(0, str(BACKEND_DIR))

from benchmark import (  # noqa: E402
    BenchmarkConfig,
    BenchmarkRunner,
    EngineResultStore,
)
from engines import load_engine  # noqa: E402

# Windows consoles often default to cp1252, which cannot encode tokens this
# model emits. Force UTF-8-safe stdout/stderr.
for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ENGINE_ID = "llamacpp-optimized"
EXPECTED_RUNTIME = "llama.cpp"
DEFAULT_MODEL = PROJECT_ROOT / "models" / "gguf" / "qwen2.5-0.5b-instruct-q4_k_m.gguf"

# One deterministic benchmark prompt, suitable for comparing inference
# performance (identical for every warmup/timed call).
PROMPT = "Explain what an AI inference engine is."


def build_config() -> BenchmarkConfig:
    """The exact STEP 9/11 benchmark procedure (single source of truth).

    Module-level so deployment tests can verify the procedure without
    loading the ~470 MB model or running the benchmark.
    """
    return BenchmarkConfig(
        prompt=PROMPT,
        max_new_tokens=64,
        temperature=None,      # greedy: llama.cpp default 0.0
        chat_template=False,   # raw completion
        repeats=5,
        warmup=1,
    )


class PromptTokenStore(EngineResultStore):
    """EngineResultStore that injects the fixed prompt_tokens count into every
    record before it is persisted (driver-side composition, no architecture
    change). The count is tokenizer-exact and prompt-identical per run."""

    def __init__(self, engine_id: str, prompt_tokens: int) -> None:
        super().__init__(engine_id)
        self._prompt_tokens = prompt_tokens

    def save(self, record: dict) -> Path:
        # Inject on a copy so the storage layer never mutates the caller's
        # record dict (the runner also keeps that dict in memory).
        return super().save({**record, "prompt_tokens": self._prompt_tokens})


def fail(category: str, message: str, exc: BaseException | None = None) -> None:
    print(f"\nERROR [{category}]: {message}", file=sys.stderr)
    if exc is not None:
        print(f"Detail: {type(exc).__name__}: {exc}", file=sys.stderr)
    print("# Status: FAIL", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    print("=" * 72)
    print("ArmInferX STEP 9 - Real Q4_K_M Benchmark (llamacpp-optimized)")
    print("=" * 72)

    # ------------------------------------------------------------------ 1. Load
    print("\n[1] Loading engine through the registry ...")
    try:
        engine = load_engine(ENGINE_ID)
    except Exception as exc:
        fail("engine load", f"load_engine({ENGINE_ID!r}) failed", exc)

    info = engine.get_model_info()
    print(f"    engine_id      = {info.engine_id}")
    print(f"    runtime        = {info.runtime}")
    print(f"    supports_stream= {info.supports_streaming}")
    print(f"    model_id       = {info.model_id}")
    print(f"    n_ctx          = {info.max_context}")
    print(f"    model_path     = {info.extra.get('model_path')}")
    print(f"    n_threads      = {info.extra.get('n_threads')}")
    print(f"    n_gpu_layers   = {info.extra.get('n_gpu_layers')}")

    if info.engine_id != ENGINE_ID or info.runtime != EXPECTED_RUNTIME:
        fail(
            "engine identity",
            f"expected {ENGINE_ID}/{EXPECTED_RUNTIME}, got "
            f"{info.engine_id}/{info.runtime}",
        )
    model_path = Path(info.extra.get("model_path") or DEFAULT_MODEL)
    if model_path.name != DEFAULT_MODEL.name:
        fail("engine identity", f"expected default Q4_K_M model, got {model_path.name}")

    # ------------------------------------------------------------------ 2. Config
    config = build_config()
    print("\n[2] Benchmark config")
    print(f"    prompt          = {PROMPT!r}")
    print(f"    max_new_tokens  = {config.max_new_tokens}")
    print(f"    temperature     = {config.temperature} (greedy)")
    print(f"    chat_template   = {config.chat_template}")
    print(f"    repeats         = {config.repeats}")
    print(f"    warmup          = {config.warmup}")

    # prompt_tokens: exact same computation the engine performs in generate().
    llm = getattr(engine, "_llm", None)
    if llm is None:
        fail("prompt tokenization", "engine has no loaded model handle")
    try:
        prompt_tokens = len(llm.tokenize(PROMPT.encode("utf-8")))
    except Exception as exc:
        fail("prompt tokenization", "failed to tokenize prompt", exc)
    print(f"    prompt_tokens   = {prompt_tokens} (tokenizer-native)")

    # ------------------------------------------------------------------ 3. Run
    store = PromptTokenStore(ENGINE_ID, prompt_tokens)
    runner = BenchmarkRunner(store=store)
    print(f"\n[3] Running benchmark ({config.warmup} warmup + {config.repeats} timed) ...")
    try:
        run = runner.run(engine, config)
    except Exception as exc:
        fail("benchmark run", f"BenchmarkRunner.run() failed", exc)

    # ------------------------------------------------------------------ 4. Per-run
    print("\n[4] Per-run metrics (engine-reported inference latency is persisted)")
    header = (
        f"{'run':>3} | {'wall_ms':>9} | {'lat_ms':>9} | {'ttft_ms':>9} | "
        f"{'gen_tok':>7} | {'tok/s':>7} | {'mem_MB':>8} | {'cpu_%':>6}"
    )
    print(header)
    print("-" * len(header))
    def fmt(value: float | None, digits: int = 2, width: int = 9) -> str:
        if value is None:
            return "N/A".rjust(width)
        return f"{value:.{digits}f}".rjust(width)

    for i, (record, metrics) in enumerate(zip(run.records, run.metrics), start=1):
        print(
            f"{i:>3} | {fmt(metrics.latency_ms)} | {fmt(record['latency_ms'])} | "
            f"{fmt(record['ttft_ms'])} | "
            f"{record['generated_tokens']:>7} | "
            f"{fmt(record['tokens_per_second'], width=7)} | "
            f"{fmt(record['memory_mb'], width=8)} | {fmt(record['cpu_percent'], width=6)}"
        )

    # ------------------------------------------------------------------ 5. Aggregate
    agg = run.aggregates
    print("\n[5] BenchmarkAggregates")
    for key, value in agg.to_dict().items():
        print(f"    {key:24s} = {value}")

    # ------------------------------------------------------------------ 6. Verify
    print("\n[6] Persisted records (results/benchmarks/llamacpp-optimized/)")
    result_files = sorted(store.root_dir.glob(f"{store.filename_prefix}*.json"))
    if len(result_files) != config.repeats:
        fail(
            "persistence",
            f"expected {config.repeats} result files under {store.root_dir}, "
            f"found {len(result_files)}",
        )
    ok = True
    for path in result_files:
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            fail("persistence", f"unreadable record {path}: {exc}", exc)
        checks = {
            "engine_id": record.get("engine_id") == ENGINE_ID,
            "runtime": record.get("runtime") == EXPECTED_RUNTIME,
            "model": record.get("model") == info.model_id,
            "model_path": (record.get("model_path") or "").endswith(
                DEFAULT_MODEL.name
            ),
            "prompt_tokens": record.get("prompt_tokens") == prompt_tokens,
            "latency_ms": isinstance(record.get("latency_ms"), (int, float)),
            "ttft_ms": isinstance(record.get("ttft_ms"), (int, float)),
            "generated_tokens": isinstance(record.get("generated_tokens"), int),
            "tokens_per_second": isinstance(
                record.get("tokens_per_second"), (int, float)
            ),
            "memory_mb": isinstance(record.get("memory_mb"), (int, float)),
            "cpu_percent": isinstance(record.get("cpu_percent"), (int, float)),
        }
        failed_checks = [k for k, v in checks.items() if not v]
        status = "OK " if not failed_checks else f"FAIL({','.join(failed_checks)})"
        if failed_checks:
            ok = False
        print(f"    {path.name}  ->  {status}")

    if not ok:
        fail("verification", "one or more persisted records failed verification")
    print(f"\n    store root      = {store.root_dir}")

    print("\n# Status: PASS")


if __name__ == "__main__":
    main()
