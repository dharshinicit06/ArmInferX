"""ArmInferX - STEP 11: generate the machine-readable optimization report.

This driver composes ONLY existing pieces:

- ``engines.registry.load_engine("llamacpp-optimized")`` — the real Q4_K_M
  engine (defaults untouched: n_ctx=2048, n_threads=8, n_gpu_layers=0).
- ``benchmark.BenchmarkConfig`` + ``benchmark.BenchmarkRunner`` — the exact
  STEP 9 procedure (prompt, max_new_tokens=64, temperature=None/greedy,
  chat_template=False, warmup=1, repeats=5) is RE-RUN every time so the
  report's benchmark section is fresh, reproducible evidence.
- ``benchmark.EngineResultStore`` — persists each repeat under
  ``results/benchmarks/llamacpp-optimized/``.
- ``optimization.gguf_metadata`` / ``optimization.model_footprint`` /
  ``optimization.report`` — the new evidence layer (parses the real GGUF
  file bytes, computes the storage-footprint comparison, and classifies
  measured vs not-measured metrics).

Output: ``results/optimization_report.json`` (gitignored, frontend-ready).

Usage (from the repo root):
    backend/.venv/Scripts/python scripts/run_optimization_report.py
    backend/.venv/Scripts/python scripts/run_optimization_report.py --analyze-only

Exit codes:
    0  report generated successfully
    1  any failure (analysis, footprint, benchmark run, or write)
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

# Project root = one level above this script (scripts/run_optimization_report.py).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from benchmark import (  # noqa: E402
    BenchmarkConfig,
    BenchmarkRunner,
    EngineResultStore,
)
from engines import load_engine  # noqa: E402
from optimization import (  # noqa: E402
    analyze_gguf,
    build_optimization_report,
    compute_footprint,
)

for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ENGINE_ID = "llamacpp-optimized"
PROMPT = "Explain what an AI inference engine is."
MAX_NEW_TOKENS = 64
WARMUP = 1
REPEATS = 5

GGUF_DIR = PROJECT_ROOT / "models" / "gguf"
Q4_GGUF = GGUF_DIR / "qwen2.5-3b-instruct-q4_k_m.gguf"
FP16_SHARDS = [
    GGUF_DIR / "qwen2.5-3b-instruct-fp16-00001-of-00002.gguf",
    GGUF_DIR / "qwen2.5-3b-instruct-fp16-00002-of-00002.gguf",
]
REPORT_OUT = PROJECT_ROOT / "results" / "optimization_report.json"


class RecordingStore(EngineResultStore):
    """EngineResultStore that also records the exact paths it wrote."""

    def __init__(self, engine_id: str) -> None:
        super().__init__(engine_id)
        self.saved_paths: list[Path] = []

    def save(self, record: dict) -> Path:
        path = super().save(record)
        self.saved_paths.append(path)
        return path


def fail(category: str, message: str, exc: BaseException | None = None) -> None:
    print(f"\nERROR [{category}]: {message}", file=sys.stderr)
    if exc is not None:
        print(f"Detail: {type(exc).__name__}: {exc}", file=sys.stderr)
    print("# Status: FAIL", file=sys.stderr)
    sys.exit(1)


def tool_versions() -> dict:
    """Versions of the runtime pieces relevant to the report."""
    import llama_cpp
    import psutil
    import torch
    import transformers

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "llama_cpp_python": getattr(llama_cpp, "__version__", "unknown"),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "psutil": psutil.__version__,
    }


def hardware_info() -> dict:
    import psutil

    vm = psutil.virtual_memory()
    return {
        "total_ram_gb": round(vm.total / 1024**3, 2),
        "cpu_logical_count": psutil.cpu_count(logical=True),
        "cpu_physical_count": psutil.cpu_count(logical=False),
        "platform": platform.platform(),
        "machine": platform.machine(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="analyze model + footprint only; skip the benchmark re-run",
    )
    args = parser.parse_args()

    print("=" * 72)
    print("ArmInferX STEP 11 - Optimization Evidence Report")
    print("=" * 72)

    # ---------------------------------------------------------------- [1] model
    print("\n[1] Analyzing GGUF files (pure parser, no model load) ...")
    for path in [Q4_GGUF, *FP16_SHARDS]:
        if not path.is_file():
            fail("model analysis", f"GGUF not found: {path}")
    q4_meta = analyze_gguf(Q4_GGUF)
    fp16_meta = [analyze_gguf(p, include_sha256=False) for p in FP16_SHARDS]
    print(f"    Q4_K_M: {q4_meta['file_name']} "
          f"({q4_meta['file_size_bytes'] / 1024**2:.1f} MB, "
          f"gguf v{q4_meta['gguf_version']}, "
          f"file_type={q4_meta['file_type']}, "
          f"params={q4_meta['parameter_count'] / 1e9:.2f}B, "
          f"sha256={q4_meta['sha256'][:16]}...)")
    print(f"    FP16  : {len(fp16_meta)} shard(s) analyzed")

    # ------------------------------------------------------------- [2] footprint
    print("\n[2] FP16 vs Q4_K_M storage footprint ...")
    footprint = compute_footprint(FP16_SHARDS, [Q4_GGUF])
    print(f"    FP16 total  : {footprint['fp16']['total_mb']:.1f} MB "
          f"({footprint['fp16']['shard_count']} shards)")
    print(f"    Q4_K_M total: {footprint['q4_k_m']['total_mb']:.1f} MB")
    print(f"    reduction   : {footprint['reduction']['mb']:.1f} MB "
          f"({footprint['reduction']['percent']:.2f}%)")

    # ------------------------------------------------------------ [3] benchmark
    benchmark_block: dict | None = None
    if not args.analyze_only:
        print("\n[3] Re-running the STEP 9 benchmark "
              f"({WARMUP} warmup + {REPEATS} timed) ...")
        try:
            engine = load_engine(ENGINE_ID)
        except Exception as exc:
            fail("engine load", f"load_engine({ENGINE_ID!r}) failed", exc)
        info = engine.get_model_info()
        print(f"    engine_id={info.engine_id} runtime={info.runtime} "
              f"model={info.model_id} n_ctx={info.max_context}")

        llm = getattr(engine, "_llm", None)
        if llm is None:
            fail("prompt tokenization", "engine has no loaded model handle")
        try:
            prompt_tokens = len(llm.tokenize(PROMPT.encode("utf-8")))
        except Exception as exc:
            fail("prompt tokenization", "failed to tokenize prompt", exc)

        config = BenchmarkConfig(
            prompt=PROMPT,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=None,
            chat_template=False,
            repeats=REPEATS,
            warmup=WARMUP,
        )
        store = RecordingStore(ENGINE_ID)
        runner = BenchmarkRunner(store=store)
        try:
            run = runner.run(engine, config)
        except Exception as exc:
            fail("benchmark run", "BenchmarkRunner.run() failed", exc)

        per_run = []
        for i, (record, metrics) in enumerate(
            zip(run.records, run.metrics), start=1
        ):
            per_run.append(
                {
                    "repeat": i,
                    "wall_clock_latency_ms": metrics.latency_ms,
                    "engine_latency_ms": record["latency_ms"],
                    "ttft_ms": record["ttft_ms"],
                    "generated_tokens": record["generated_tokens"],
                    "tokens_per_second": record["tokens_per_second"],
                    "memory_mb": record["memory_mb"],
                    "cpu_percent": record["cpu_percent"],
                }
            )
        benchmark_block = {
            "engine_id": run.engine_id,
            "runtime": run.runtime,
            "model_id": run.model_id,
            "model_path": run.model_path,
            "prompt": config.prompt,
            "prompt_tokens": prompt_tokens,
            "warmup": config.warmup,
            # Measured Q4_K_M file size (reported under ``measured``).
            "model_footprint_bytes": q4_meta["file_size_bytes"],
            "aggregates": run.aggregates.to_dict(),
            "per_run": per_run,
            "records": [str(p) for p in store.saved_paths],
            "results_dir": str(store.root_dir),
            "generation_kwargs": {"max_new_tokens": config.max_new_tokens},
        }
        print(f"    mean latency={run.aggregates.mean_latency_ms:.2f} ms | "
              f"mean TTFT={run.aggregates.mean_ttft_ms:.2f} ms | "
              f"mean tok/s={run.aggregates.mean_tokens_per_second:.2f}")
        print(f"    records saved: {len(store.saved_paths)}")
    else:
        print("\n[3] --analyze-only: benchmark re-run skipped")

    # ---------------------------------------------------------------- [4] report
    print("\n[4] Assembling optimization report ...")
    q4_label = "Validated optimized model — CPU inference successfully completed"

    def _compact(meta: dict) -> dict:
        """Report-ready summary: all evidence facts, minus the huge tokenizer
        arrays and per-tensor detail (the tensor-type breakdown is kept)."""
        skip_keys = {
            "tokenizer.ggml.tokens",
            "tokenizer.ggml.merges",
            "tokenizer.ggml.token_type",
        }
        metadata = {
            k: v
            for k, v in meta["metadata"].items()
            if not k.startswith("tokenizer.ggml.tokens") and k not in skip_keys
        }
        return {
            "path": meta["path"],
            "file_name": meta["file_name"],
            "file_size_bytes": meta["file_size_bytes"],
            "gguf_version": meta["gguf_version"],
            "tensor_count": meta["tensor_count"],
            "parameter_count": meta["parameter_count"],
            "file_type": meta["file_type"],
            "quantization": meta["quantization"],
            "architecture": meta["architecture"],
            "model_name": meta["model_name"],
            "quantization_version": meta["quantization_version"],
            "context_length": meta["context_length"],
            "tensor_types": meta["tensor_types"],
            "sha256": meta.get("sha256"),
            "metadata": metadata,
        }

    model_block = {
        "q4_k_m": _compact(q4_meta),
        "fp16_shards": [_compact(m) for m in fp16_meta],
    }

    configuration_block = {
        "engine_id": ENGINE_ID,
        "runtime": "llama.cpp",
        "quantization": "Q4_K_M",
        "model_path": str(Q4_GGUF),
        "n_ctx": 2048,
        "n_threads": 8,
        "n_gpu_layers": 0,
        "temperature": None,
        "decoding": "greedy (deterministic)",
        "chat_template": False,
        "max_new_tokens": MAX_NEW_TOKENS,
        "warmup": WARMUP,
        "repeats": REPEATS,
    }

    report = build_optimization_report(
        project={
            "name": "ArmInferX",
            "description": "AI Inference Optimization Studio for Arm64 Cloud",
            "phase": "STEP 11 - Optimization Evidence, Validation & Benchmark Reporting",
            "tool_versions": tool_versions(),
        },
        hardware=hardware_info(),
        optimized={
            "engine_id": ENGINE_ID,
            "runtime": "llama.cpp",
            "model_id": q4_meta.get("model_name") or "qwen2.5-3b-instruct",
            "quantization": "Q4_K_M",
            "status": "validated",
            "label": q4_label,
        },
        model=model_block,
        configuration=configuration_block,
        model_footprint=footprint,
        benchmark=benchmark_block,
        limitations=[
            "FP16 baseline inference is not feasible on this machine "
            "(7.63 GiB total RAM): the STEP 10A feasibility watchdog aborted "
            "during model loading when available RAM dropped to 159-263 MB. "
            "No FP16 latency/throughput/TTFT exists, so no speedup over FP16 "
            "is claimed.",
            "The GGUF general.file_type enum reports 'Q4_K' (code 11); the "
            "'M' (medium) variant suffix is not part of the GGUF enum. The "
            "Q4_K_M label is taken from the filename; the per-tensor "
            "quantization breakdown is reported from the file itself.",
            "Benchmark measured on a Windows development laptop (CPU), not on "
            "Arm64 cloud hardware. Numbers are reproducible on this machine "
            "only and make no claim about Arm64 performance.",
            "results/optimization_report.json is a gitignored generated "
            "artifact; regenerate it with the command in 'reproducibility'.",
        ],
        reproducibility={
            "command": "backend/.venv/Scripts/python scripts/run_optimization_report.py",
            "engine_load": 'load_engine("llamacpp-optimized")',
            "benchmark": "BenchmarkRunner.run(engine, BenchmarkConfig(...)) "
                         "with warmup=1, repeats=5, max_new_tokens=64, "
                         "temperature=None, chat_template=False",
            "prompt": PROMPT,
            "model": str(Q4_GGUF),
            "result_records_dir": str(PROJECT_ROOT / "results" / "benchmarks" / ENGINE_ID),
            "note": "Re-running regenerates the benchmark records; each "
                    "record is timestamped so runs accumulate as evidence.",
        },
    )

    # ---------------------------------------------------------------- [5] write
    try:
        REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
        REPORT_OUT.write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    except OSError as exc:
        fail("report write", f"failed to write {REPORT_OUT}: {exc}", exc)

    print(f"\nReport written: {REPORT_OUT}")
    print("\n" + "=" * 72)
    print("STEP 11 REPORT SUMMARY")
    print("=" * 72)
    summary = {
        "model": report["model"]["q4_k_m"]["file_name"],
        "file_size_mb": round(report["model"]["q4_k_m"]["file_size_bytes"] / 1024**2, 2),
        "fp16_total_mb": report["model_footprint"]["fp16"]["total_mb"],
        "q4_total_mb": report["model_footprint"]["q4_k_m"]["total_mb"],
        "footprint_reduction_percent": report["model_footprint"]["reduction"]["percent"],
        "benchmark": report["benchmark"]["aggregates"] if report["benchmark"] else None,
        "fp16_feasibility": report["feasibility"]["fp16"]["status"],
        "measured": sorted(report["measured"]),
        "not_measured": sorted(report["not_measured"]),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\n# Status: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
