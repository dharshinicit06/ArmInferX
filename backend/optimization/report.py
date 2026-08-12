"""STEP 11 optimization report builder for ArmInferX (Phase 5).

Assembles the machine-readable evidence report (JSON) from the pieces
produced by the other modules:

- ``optimization.gguf_metadata``  -> model facts (validated from file bytes)
- ``optimization.model_footprint`` -> FP16 vs Q4_K_M storage footprint
- the existing benchmark layer    -> reproducible Q4_K_M benchmark results
- documented STEP 10A facts       -> FP16 feasibility hardware limitation

The report makes an explicit, auditable distinction between **measured**
facts (Q4_K_M file size, latency, TTFT, throughput, memory, CPU, run count)
and **not measured** facts (FP16 latency/throughput/TTFT, any comparative
speedup). No FP16 inference was ever completed on this machine, so no
performance comparison between the two models is made anywhere in the report.
"""

from __future__ import annotations

from datetime import datetime, timezone

#: FP16 baseline facts recorded during STEP 10A (measured at the time of the
#: watchdog abort; see scripts/feasibility_baseline.py).
FP16_FEASIBILITY = {
    "status": "not_feasible",
    "classification": "hardware memory constraint",
    "engine_id": "transformers-baseline",
    "runtime": "transformers",
    "model": "Qwen2.5-3B-Instruct",
    "dtype": "float16",
    "inference_completed": False,
    "reason": (
        "Model loading started but could not complete: the feasibility "
        "watchdog detected severe memory pressure (system available RAM "
        "dropped to 159-263 MB on a machine with 7.63 GiB total) and aborted "
        "the run safely. No FP16 inference was performed."
    ),
    "watchdog_abort": {
        "available_ram_at_abort_mb": 159,
        "reported_available_ram_range_mb": [159, 263],
        "trigger": "system available RAM below the 300 MB severe-paging alarm",
        "total_ram_gb": 7.63,
    },
    "label": "Reference FP16 model — inference infeasible on current 7.63 GiB RAM machine",
}

#: Metrics that must never be fabricated: they stay null unless an FP16
#: inference is actually completed under identical conditions.
NOT_MEASURED_METRICS = {
    "fp16_inference_latency_ms": None,
    "fp16_ttft_ms": None,
    "fp16_tokens_per_second": None,
    "fp16_throughput": None,
    "comparative_speedup_vs_fp16_percent": None,
}

NOT_MEASURED_EXPLANATION = (
    "The FP16 baseline completed no inference on this machine (7.63 GiB RAM; "
    "the STEP 10A feasibility watchdog aborted during model loading when "
    "available RAM dropped to 159-263 MB). Therefore no FP16 latency, TTFT, "
    "throughput, or speedup-vs-FP16 value was measured, and no percentage "
    "performance improvement over FP16 is claimed."
)

REQUIRED_SECTIONS = [
    "project",
    "hardware",
    "baseline",
    "optimized",
    "model",
    "configuration",
    "model_footprint",
    "benchmark",
    "feasibility",
    "limitations",
    "reproducibility",
    "measured",
    "not_measured",
]


def _measured_from_benchmark(benchmark: dict | None) -> dict:
    """Extract the measured metrics block from a benchmark run dict."""
    if not benchmark:
        return {}
    aggregates = benchmark.get("aggregates") or {}
    measured = {
        "q4_k_m_file_size_bytes": benchmark.get("model_footprint_bytes"),
        "q4_k_m_latency_mean_ms": aggregates.get("mean_latency_ms"),
        "q4_k_m_latency_median_ms": aggregates.get("median_latency_ms"),
        "q4_k_m_latency_p90_ms": aggregates.get("p90_latency_ms"),
        "q4_k_m_ttft_mean_ms": aggregates.get("mean_ttft_ms"),
        "q4_k_m_generated_tokens_mean": aggregates.get("mean_generated_tokens"),
        "q4_k_m_tokens_per_second_mean": aggregates.get("mean_tokens_per_second"),
        "q4_k_m_peak_memory_mb": aggregates.get("peak_memory_mb"),
        "q4_k_m_mean_cpu_percent": aggregates.get("mean_cpu_percent"),
        "q4_k_m_prompt_tokens": benchmark.get("prompt_tokens"),
        "benchmark_repetitions": aggregates.get("runs"),
        "warmup_runs": benchmark.get("warmup"),
    }
    return {k: v for k, v in measured.items() if v is not None}


def build_optimization_report(
    *,
    project: dict,
    hardware: dict,
    optimized: dict,
    model: dict,
    configuration: dict,
    model_footprint: dict,
    benchmark: dict | None,
    limitations: list[str],
    reproducibility: dict,
    generated_at: str | None = None,
) -> dict:
    """Assemble the full STEP 11 optimization report.

    Args:
        project: Project identity (name, repo, phase, tool versions, ...).
        hardware: Machine facts (total RAM, CPU, platform).
        optimized: Optimized-engine identity + status block.
        model: Model analysis output (Q4_K_M + FP16 metadata from
            ``gguf_metadata.analyze_gguf``).
        configuration: The benchmark configuration used (prompt, limits, ...).
        model_footprint: FP16 vs Q4_K_M footprint comparison from
            ``model_footprint.compute_footprint``.
        benchmark: Fresh benchmark run dict (aggregates, per-run rows,
            records, engine identity), or ``None`` when no run was executed.
        limitations: List of honest limitation statements.
        reproducibility: Exact commands/config needed to reproduce.

    Returns:
        The report dict. JSON-serializable by construction.
    """
    baseline = dict(FP16_FEASIBILITY)

    measured = {}
    if benchmark:
        measured = _measured_from_benchmark(benchmark)

    report = {
        "project": project,
        "hardware": hardware,
        "baseline": baseline,
        "optimized": optimized,
        "model": model,
        "configuration": configuration,
        "model_footprint": model_footprint,
        "benchmark": benchmark,
        "feasibility": {
            "fp16": baseline,
            "explanation": NOT_MEASURED_EXPLANATION,
        },
        "limitations": limitations,
        "reproducibility": reproducibility,
        "measured": measured,
        "not_measured": dict(NOT_MEASURED_METRICS),
        "not_measured_explanation": NOT_MEASURED_EXPLANATION,
        "generated_at": generated_at
        or datetime.now(timezone.utc).isoformat(),
    }
    if missing := [s for s in REQUIRED_SECTIONS if s not in report]:
        raise ValueError(f"report missing required sections: {missing}")
    return report


def utc_now_iso() -> str:
    """ISO-8601 UTC timestamp for the report."""
    return datetime.now(timezone.utc).isoformat()
