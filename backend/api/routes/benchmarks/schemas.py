"""Pydantic response models for the benchmark results API.

These schemas define the public HTTP contract of the benchmark endpoints and
carry the OpenAPI documentation (descriptions, examples) shown in Swagger UI.
They mirror the shape of the JSON records saved under ``results/baseline/``.
"""

from pydantic import BaseModel, ConfigDict, Field

RECORD_EXAMPLE = {
    "prompt": "Explain Artificial Intelligence",
    "model": "qwen2.5-3b-instruct",
    "response": "Artificial Intelligence (AI) is the simulation of human "
    "intelligence in machines...",
    "latency_ms": 1234.56,
    "ttft_ms": 380.12,
    "memory_mb": 307.88,
    "cpu_percent": 42.5,
    "generated_tokens": 96,
    "tokens_per_second": 77.76,
    "timestamp": "2026-08-07T04:26:03.082404+00:00",
}

SUMMARY_EXAMPLE = {
    "avg_latency_ms": 1234.56,
    "avg_memory_mb": 307.88,
    "avg_cpu_percent": 42.5,
    "total_runs": 12,
}


class BenchmarkRecord(BaseModel):
    """A single saved benchmark record (one ``results/baseline/*.json`` file)."""

    model_config = ConfigDict(json_schema_extra={"examples": [RECORD_EXAMPLE]})

    prompt: str = Field(..., description="The prompt the model generated from.")
    model: str = Field(..., description="Identifier of the model that produced the response.")
    response: str = Field(..., description="Generated text.")
    latency_ms: float = Field(..., ge=0, description="Wall-clock inference latency in milliseconds.")
    ttft_ms: float | None = Field(
        None,
        ge=0,
        description=(
            "Time-to-first-token in milliseconds: wall-clock time from the "
            "start of inference until the first generated token is produced "
            "(prefill + first decode step). Kept separate from the total "
            "latency_ms. Null when unavailable."
        ),
    )
    memory_mb: float = Field(..., ge=0, description="Peak process memory (RSS) in MB.")
    cpu_percent: float = Field(..., ge=0, description="Process CPU usage (%) during the run.")
    generated_tokens: int | None = Field(
        None,
        ge=0,
        description=(
            "Number of output tokens the model generated, counted from the "
            "model's own output token IDs (tokenizer-native; no character/word "
            "estimation). Null on records saved before this metric existed."
        ),
    )
    tokens_per_second: float | None = Field(
        None,
        ge=0,
        description=(
            "Generation throughput in tokens per second, computed as "
            "generated_tokens / (latency_ms / 1000) using the same inference "
            "time persisted as latency_ms. Null when unavailable."
        ),
    )
    timestamp: str = Field(
        ..., description="ISO-8601 UTC timestamp of the benchmark run."
    )
    engine_id: str | None = Field(
        None,
        description=(
            "Engine that produced the run (e.g. 'llamacpp-optimized', "
            "'transformers-baseline'). Null on records saved before engine "
            "selection existed, so old baseline records are never conflated "
            "with engine-tagged ones."
        ),
    )
    runtime: str | None = Field(
        None,
        description=(
            "Underlying runtime name (e.g. 'llama.cpp', 'transformers'). "
            "Null when the engine did not expose it."
        ),
    )


class BenchmarkSummary(BaseModel):
    """Aggregate statistics over all saved benchmark records."""

    model_config = ConfigDict(json_schema_extra={"examples": [SUMMARY_EXAMPLE]})

    avg_latency_ms: float = Field(..., ge=0, description="Mean latency across all runs (ms).")
    avg_memory_mb: float = Field(..., ge=0, description="Mean process memory across all runs (MB).")
    avg_cpu_percent: float = Field(..., ge=0, description="Mean CPU usage across all runs (%).")
    total_runs: int = Field(..., ge=0, description="Number of benchmark runs recorded.")
