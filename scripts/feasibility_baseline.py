"""ArmInferX - STEP 10A: Transformers baseline feasibility check.

> **Historical note (2026-08-13):** the FP16 Transformers baseline was removed
> from the engine registry before the hackathon submission (STEP 10A proved it
> infeasible on this machine, and the demo ships llama.cpp only). Running this
> script now ends with an ``UnknownEngineError`` listing the available engine
> (``llamacpp-optimized``) — which itself documents the feasibility outcome.
> The script is kept as historical evidence of the STEP 10A check.

Purpose
-------
Verify whether the existing ``transformers-baseline`` engine can perform ONE
real inference on this machine (model load + a single generation), using the
existing downloaded Qwen2.5-3B-Instruct model and the existing baseline
configuration. This is a feasibility test only - the 5-repeat benchmark is
NOT run here.

No project file is modified: the engine, model loader, dtype, device, memory
limits, BenchmarkRunner and BenchmarkService are all untouched. This script
only *composes* the existing ``engines.registry.load_engine`` entry point.

Safety design
-------------
Model.load() with the existing config caps CPU RAM at 3GiB and disk-offloads
the rest of the ~6.2 GB fp16 weights, so generation can be very slow and can
page. To stop safely, the actual load+inference runs in a CHILD process and
this driver acts as a watchdog:

- samples system available memory and the child's RSS every second;
- kills the child and reports the exact condition if:
    * system available memory drops below 300 MB (severe paging risk),
    * child RSS exceeds ~6.4 GiB (machine has ~7.6 GiB total), or
    * the whole run exceeds 15 minutes (excessively slow).

The child writes a JSON report (to a temp file, not the repo) which the
watchdog reads back and prints.

Usage (from the repo root):
    backend/.venv/Scripts/python scripts/feasibility_baseline.py

Exit codes:
    0  feasibility PASS (model loaded, one inference completed)
    1  feasibility FAIL (error, memory alarm, or timeout) - exact reason reported
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# Project root = one level above this script (scripts/feasibility_baseline.py).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"
# The benchmark/engines packages import as top-level packages; put backend on
# the path (mirrors running with cwd=backend).
sys.path.insert(0, str(BACKEND_DIR))

for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ENGINE_ID = "transformers-baseline"
PROMPT = "Explain what an AI inference engine is."
MAX_NEW_TOKENS = 64

# --- Watchdog limits ---------------------------------------------------------
TIMEOUT_S = 15 * 60          # 15 minutes: "excessively slow" cap
MEMORY_AVAILABLE_ALARM_MB = 300.0   # system available RAM (paging risk)
CHILD_RSS_ALARM_GB = 6.4            # child RSS (machine has ~7.6 GiB total)

REPORT_PATH = Path(tempfile.gettempdir()) / "armiferx_step10a_report.json"


# ---------------------------------------------------------------------------
# Child: does the real work and writes a JSON report
# ---------------------------------------------------------------------------
def child(report_path: Path) -> int:
    import psutil  # noqa: PLC0415 - child-only dependency

    from engines import load_engine  # noqa: PLC0415 - child-only dependency

    proc = psutil.Process()
    report: dict = {"engine_id": ENGINE_ID, "status": "unknown"}
    heartbeats: list[str] = []

    def heartbeat(msg: str) -> None:
        line = f"[child +{time.perf_counter() - t0:.1f}s] {msg}"
        heartbeats.append(line)
        print(line, file=sys.stderr, flush=True)

    t0 = time.perf_counter()
    try:
        # ------------------------------------------------------------ 1. Load
        heartbeat(f"loading engine via registry load_engine({ENGINE_ID!r}) "
                  "(device=cpu, dtype=float16, max_cpu_memory=3GiB, defaults)...")
        engine = load_engine(ENGINE_ID)
        load_time_s = time.perf_counter() - t0
        rss_after_load_mb = proc.memory_info().rss / (1024.0 * 1024.0)
        info = engine.get_model_info()
        heartbeat(f"model loaded in {load_time_s:.1f}s - model_id={info.model_id} "
                  f"context={info.max_context} RSS={rss_after_load_mb:.0f}MB")

        # -------------------------------------------------------- 2. Inference
        heartbeat(f"running ONE inference: max_new_tokens={MAX_NEW_TOKENS}, "
                  "temperature=None (greedy), use_chat_template=False ...")
        proc.cpu_percent(None)                       # establish psutil baseline
        memory_before_mb = proc.memory_info().rss / (1024.0 * 1024.0)
        t_inf = time.perf_counter()
        result = engine.generate(
            PROMPT,
            max_new_tokens=MAX_NEW_TOKENS,
            use_chat_template=False,
        )
        inference_wall_s = time.perf_counter() - t_inf
        memory_after_mb = proc.memory_info().rss / (1024.0 * 1024.0)
        cpu_percent = proc.cpu_percent(None)

        tokens_per_second = None
        if result.latency_ms > 0 and result.generated_tokens:
            tokens_per_second = round(
                result.generated_tokens / (result.latency_ms / 1000.0), 2
            )

        report = {
            "status": "success",
            "engine_id": ENGINE_ID,
            "runtime": info.runtime,
            "model_id": info.model_id,
            "model_dir": str(PROJECT_ROOT / "models" / "downloaded" / "qwen2.5-3b-instruct"),
            "prompt": PROMPT,
            "load_time_s": round(load_time_s, 2),
            "rss_after_load_mb": round(rss_after_load_mb, 2),
            "inference_wall_s": round(inference_wall_s, 2),
            "latency_ms": round(result.latency_ms, 2),
            "ttft_ms": round(result.ttft_ms, 2) if result.ttft_ms is not None else None,
            "generated_tokens": result.generated_tokens,
            "prompt_tokens": result.prompt_tokens,
            "tokens_per_second": tokens_per_second,
            "memory_before_mb": round(memory_before_mb, 2),
            "memory_after_mb": round(memory_after_mb, 2),
            "memory_peak_mb": round(max(memory_before_mb, memory_after_mb), 2),
            "cpu_percent": round(cpu_percent, 2),
            "generated_text_preview": result.generated_text[:120],
        }
        heartbeat(f"inference done in {inference_wall_s:.1f}s wall "
                  f"(latency={result.latency_ms:.0f}ms, "
                  f"tokens={result.generated_tokens})")
    except Exception as exc:  # noqa: BLE001 - report the exact failure
        report = {
            "status": "error",
            "engine_id": ENGINE_ID,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        heartbeat(f"FAILED: {type(exc).__name__}: {exc}")
    finally:
        report["heartbeats"] = heartbeats
        try:
            report_path.write_text(
                json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except OSError as exc:
            print(f"ERROR writing report {report_path}: {exc}", file=sys.stderr)
            return 1
    return 0 if report.get("status") == "success" else 1


# ---------------------------------------------------------------------------
# Parent: watchdog over the child process
# ---------------------------------------------------------------------------
def parent(report_path: Path) -> int:
    import psutil  # noqa: PLC0415 - parent-only dependency

    if report_path.exists():
        report_path.unlink()

    vm = psutil.virtual_memory()
    print("=" * 72)
    print("ArmInferX STEP 10A - Transformers Baseline Feasibility Check")
    print("=" * 72)
    print(f"    machine total RAM : {vm.total / 1024**3:.2f} GiB")
    print(f"    RAM available     : {vm.available / 1024**3:.2f} GiB")
    print(f"    child timeout     : {TIMEOUT_S}s   (excessively-slow cap)")
    print(f"    RAM alarms        : available < {MEMORY_AVAILABLE_ALARM_MB:.0f} MB | "
          f"child RSS > {CHILD_RSS_ALARM_GB:.1f} GiB")
    print()

    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child",
        "--report",
        str(report_path),
    ]
    child_proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    start = time.time()
    killed_reason: str | None = None
    while child_proc.poll() is None:
        time.sleep(1)
        elapsed = time.time() - start

        vm = psutil.virtual_memory()
        if vm.available < MEMORY_AVAILABLE_ALARM_MB * 1024 * 1024:
            killed_reason = (
                f"severe memory pressure: system available RAM dropped to "
                f"{vm.available / 1024**2:.0f} MB (severe paging risk) after "
                f"{elapsed:.0f}s"
            )
            child_proc.kill()
            break

        try:
            cproc = psutil.Process(child_proc.pid)
            c_rss_gb = cproc.memory_info().rss / 1024**3
            if c_rss_gb > CHILD_RSS_ALARM_GB:
                killed_reason = (
                    f"child RSS reached {c_rss_gb:.2f} GiB "
                    f"(alarm threshold {CHILD_RSS_ALARM_GB:.1f} GiB) after "
                    f"{elapsed:.0f}s"
                )
                child_proc.kill()
                break
        except psutil.NoSuchProcess:
            pass

        if elapsed > TIMEOUT_S:
            killed_reason = (
                f"excessively slow: no completion within {TIMEOUT_S}s "
                f"({elapsed:.0f}s elapsed)"
            )
            child_proc.kill()
            break

    out, err = child_proc.communicate(timeout=30)
    total_s = time.time() - start
    exit_code = child_proc.returncode

    print(f"[watchdog] child exited after {total_s:.1f}s (exit code {exit_code})")
    if err.strip():
        print("\n[child progress / stderr]")
        for line in err.splitlines()[-40:]:
            print(f"    {line}")

    # --- Report --------------------------------------------------------------
    if report_path.exists():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            report = {}
    else:
        report = {}

    print("\n" + "=" * 72)
    print("STEP 10A RESULT")
    print("=" * 72)

    if killed_reason:
        print(f"Model load : UNKNOWN (aborted: {killed_reason})")
        print(f"Inference  : NOT COMPLETED (aborted)")
        print(f"Abort      : {killed_reason}")
        print("\n# Status: ABORTED - SAFE STOP")
        return 1

    if exit_code != 0 or report.get("status") == "error":
        print("Model load : FAIL")
        print("Inference  : FAIL")
        print(f"Reason     : {report.get('error_type', 'unknown')}: "
              f"{report.get('error', 'child exited with code %d' % exit_code)}")
        print("\n# Status: FAIL")
        return 1

    print("Model load : PASS")
    print("Inference  : PASS")
    print(f"    load_time_s        = {report['load_time_s']} s")
    print(f"    RSS after load     = {report['rss_after_load_mb']} MB")
    print(f"    model_id           = {report['model_id']}")
    print(f"    prompt_tokens      = {report['prompt_tokens']}")
    print(f"    latency_ms         = {report['latency_ms']} ms")
    print(f"    ttft_ms            = {report['ttft_ms']} ms")
    print(f"    generated_tokens   = {report['generated_tokens']}")
    print(f"    tokens_per_second  = {report['tokens_per_second']}")
    print(f"    memory_before_mb   = {report['memory_before_mb']} MB")
    print(f"    memory_after_mb    = {report['memory_after_mb']} MB")
    print(f"    memory_peak_mb     = {report['memory_peak_mb']} MB")
    print(f"    cpu_percent        = {report['cpu_percent']} %")
    print(f"    generated preview  = {report.get('generated_text_preview', '')!r}")
    print("\n# Status: PASS")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="STEP 10A feasibility check")
    parser.add_argument("--child", action="store_true", help="run the child role")
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    args = parser.parse_args()

    if args.child:
        return child(args.report)
    return parent(args.report)


if __name__ == "__main__":
    sys.exit(main())
