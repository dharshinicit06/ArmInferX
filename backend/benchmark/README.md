# ArmInferX Benchmark Methodology

This document describes how ArmInferX benchmarks inference engines so results
are **scientifically comparable** across runtimes. Actual Arm64 performance
measurements will be performed later on an Arm64 cloud environment — nothing
here claims one runtime is faster than another.

## Goal

Run the **exact same benchmark procedure** against the engine:

| engine_id | runtime | model |
|---|---|---|
| `llamacpp-optimized` | `llama.cpp` | Qwen2.5-3B-Instruct Q4_K_M (GGUF, single file) |

The engine implements the shared `InferenceEngine` interface
(`load_model` / `generate` / `get_model_info`).

## Procedure

A single run is defined by `BenchmarkConfig` (`backend/benchmark/config.py`):

| Field | Default | Meaning |
|---|---|---|
| `prompt` | — | the identical prompt for every call |
| `max_new_tokens` | `128` | identical output-length cap for both engines |
| `temperature` | `None` | `None` = each engine's deterministic greedy default; when set must be in `(0, 2]` and is passed verbatim to both |
| `chat_template` | `False` | raw-completion policy (see below) |
| `repeats` | `5` | number of timed repetitions |
| `warmup` | `1` | number of untimed warmup calls before timing |

`BenchmarkRunner.run(engine, config)`:

1. Reads engine identity via `engine.get_model_info()`.
2. Runs `warmup` **untimed** calls (same prompt + same kwargs).
3. Runs exactly `repeats` **timed** calls through the existing
   `BenchmarkService.measure()` (wall-clock latency, TTFT, memory, CPU,
   tokenizer-native token counts, tokens/sec).
4. Tags every result with `engine_id`, `runtime`, `model_id` (and `model_path`
   when the engine reports it).
5. Persists each repeat to `results/benchmarks/<engine_id>/` (engine-aware
   store; the baseline store under `results/baseline/` is untouched).
6. Computes aggregates: mean/median/p90 latency, mean TTFT, mean generated
   tokens, mean tokens/sec, peak memory, mean CPU.

## Raw-completion policy

The llama.cpp engine performs **raw completion** and does not apply a chat
template. The benchmark policy is therefore:

> **`chat_template = False`** — llama.cpp always generates from the raw
> prompt; no chat-template kwargs are ever sent to it.

The runner passes `use_chat_template` **only** to engines that opt in via
`CHAT_TEMPLATE_ENGINE_IDS` (currently none) and **never** to llama.cpp.

## Generation kwargs policy

Only arguments supported by **both** engines are used:

- `max_new_tokens` — always
- `temperature` — only when explicitly set

`do_sample` is never passed. llama.cpp never receives `use_chat_template` or
`do_sample`. This keeps the benchmark procedure identical and prevents
engine-specific kwargs from leaking into either runtime.

## Engine identity

Every benchmark result identifies the engine that produced it:

- `engine_id`: `llamacpp-optimized`
- `runtime`: `llama.cpp`
- `model_id`: derived from the loaded model
- `model_path`: engine-specific path when available (e.g. the GGUF path)

Identity is carried in `BenchmarkMetrics.extra` on every measured result and
in every persisted record.

## Storage

- **Unchanged**: `BaselineResultStore` (`results/baseline/baseline-*.json`) —
  existing records load exactly as before; no files are rewritten or deleted.
- **New**: `EngineResultStore(engine_id)` writes `results/benchmarks/<engine_id>/benchmark-*.json`
  (atomic temp-file + rename), so per-engine results are distinguishable by
  directory.

## Engine registry

`engines.registry` maps `engine_id` → engine class:

```python
from engines import load_engine, available_engines

engine = load_engine("llamacpp-optimized")          # LlamaCppOptimizedEngine
```

`load_engine` is for benchmark orchestration; the HTTP application resolves
engines through `EngineManager`.

## Running a benchmark

```python
from engines import load_engine
from benchmark import BenchmarkConfig, BenchmarkRunner

runner = BenchmarkRunner()
engine = load_engine("llamacpp-optimized")
run = runner.run(engine, BenchmarkConfig(prompt="Explain AI inference."))
print(run.to_dict())
```

## Measurement notes

- **Persisted/aggregated `latency_ms` is the engine-reported inference
  latency** (`result.latency_ms` from `generate()`), the same value the HTTP
  `/generate` record stores — so runner records and API records are directly
  comparable. The llama.cpp engine measures it around the streaming
  completion.
- The wall-clock `Stopwatch` latency of the whole `generate()` call is
  measured by `BenchmarkService.measure()` and available in-memory as
  `BenchmarkMetrics.latency_ms` (surfaced as `inference_latency_ms` in
  `metrics.extra`); the runner persists the engine-reported value above.
- TTFT is measured by the engine: llama.cpp times the first streamed token.
- Token counts are tokenizer-native (emitted token IDs / streamed tokens) —
  never character or word estimates.
