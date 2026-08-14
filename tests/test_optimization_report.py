"""Tests for the STEP 11 optimization evidence layer.

Covers, with **no real model loaded** (a tiny synthetic GGUF is written to a
temp file):

- ``optimization.gguf_metadata``: header/magic/version parsing, KV metadata
  extraction (strings, ints, arrays), tensor parameter counts, SHA-256, and
  file-type mapping.
- ``optimization.model_footprint``: size totals and reduction math on
  synthetic files, plus the explicit "storage only, not speedup" framing.
- ``optimization.report``: full report schema, the strict measured vs
  not-measured classification, and the absence of any fabricated FP16
  metric or speedup claim.

The existing benchmark architecture is untouched by these tests (regression
is covered by the rest of the suite).
"""

import hashlib
import json
import struct
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from optimization import (  # noqa: E402
    GGUFMetadataError,
    analyze_gguf,
    build_optimization_report,
    compute_footprint,
    file_type_name,
    sha256_file,
    tensor_type_name,
)

# ---------------------------------------------------------------------------
# Synthetic GGUF writer (deterministic, tiny)
# ---------------------------------------------------------------------------

# GGUF value type codes
_VAL = {
    "uint8": 0,
    "int8": 1,
    "uint16": 2,
    "int16": 3,
    "uint32": 4,
    "int32": 5,
    "float32": 6,
    "bool": 7,
    "string": 8,
    "array": 9,
    "uint64": 10,
    "int64": 11,
    "float64": 12,
}


def _pack_value(buf: bytearray, vtype: int, value) -> None:
    if vtype == _VAL["uint8"]:
        buf += struct.pack("<B", value)
    elif vtype == _VAL["int8"]:
        buf += struct.pack("<b", value)
    elif vtype == _VAL["uint16"]:
        buf += struct.pack("<H", value)
    elif vtype == _VAL["int16"]:
        buf += struct.pack("<h", value)
    elif vtype == _VAL["uint32"]:
        buf += struct.pack("<I", value)
    elif vtype == _VAL["int32"]:
        buf += struct.pack("<i", value)
    elif vtype == _VAL["float32"]:
        buf += struct.pack("<f", value)
    elif vtype == _VAL["bool"]:
        buf += struct.pack("<B", 1 if value else 0)
    elif vtype == _VAL["string"]:
        raw = value.encode("utf-8")
        buf += struct.pack("<Q", len(raw)) + raw
    elif vtype == _VAL["uint64"]:
        buf += struct.pack("<Q", value)
    elif vtype == _VAL["int64"]:
        buf += struct.pack("<q", value)
    elif vtype == _VAL["float64"]:
        buf += struct.pack("<d", value)
    elif vtype == _VAL["array"]:
        element_type, items = value
        buf += struct.pack("<I", element_type)
        buf += struct.pack("<Q", len(items))
        for item in items:
            _pack_value(buf, element_type, item)
    else:
        raise AssertionError(f"unhandled test value type {vtype}")


def write_synthetic_gguf(
    path: Path,
    *,
    version: int = 3,
    kv: dict | None = None,
    tensors: list[dict] | None = None,
    magic: bytes = b"GGUF",
) -> None:
    """Write a minimal GGUF file. ``kv`` maps key -> (vtype, value)."""
    kv = kv or {}
    tensors = tensors or []
    buf = bytearray()
    buf += magic
    buf += struct.pack("<I", version)
    buf += struct.pack("<Q", len(tensors))
    buf += struct.pack("<Q", len(kv))
    for key, (vtype, value) in kv.items():
        raw = key.encode("utf-8")
        buf += struct.pack("<Q", len(raw)) + raw
        buf += struct.pack("<I", vtype)
        _pack_value(buf, vtype, value)
    for tensor in tensors:
        name = tensor["name"].encode("utf-8")
        dims = tensor["dims"]
        buf += struct.pack("<Q", len(name)) + name
        buf += struct.pack("<I", len(dims))
        for dim in dims:
            buf += struct.pack("<Q", dim)  # modern GGUF: uint64 dims
        buf += struct.pack("<I", tensor["type"])
        buf += struct.pack("<Q", tensor.get("offset", 0))
    path.write_bytes(bytes(buf))


# ---------------------------------------------------------------------------
# gguf_metadata: parsing
# ---------------------------------------------------------------------------

def test_parse_synthetic_gguf_metadata():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "synthetic.gguf"
        write_synthetic_gguf(
            path,
            kv={
                "general.architecture": (_VAL["string"], "qwen2"),
                "general.name": (_VAL["string"], "Qwen2.5-0.5B-Instruct"),
                "general.file_type": (_VAL["uint32"], 15),  # legacy: Q4_K_M
                "general.quantization_version": (_VAL["uint32"], 2),
                "qwen2.context_length": (_VAL["uint32"], 2048),
                "tokenizer.ggml.model": (_VAL["string"], "gpt2"),
                "general.architectures": (_VAL["array"], (_VAL["string"], ["Qwen2ForCausalLM"])),
                "some.count": (_VAL["int64"], -7),
                "some.flag": (_VAL["bool"], True),
                "some.ratio": (_VAL["float64"], 0.5),
            },
        )
        meta = analyze_gguf(path, include_sha256=False)

    assert meta["file_name"] == "synthetic.gguf"
    assert meta["gguf_version"] == 3
    assert meta["tensor_count"] == 0
    assert meta["architecture"] == "qwen2"
    assert meta["model_name"] == "Qwen2.5-0.5B-Instruct"
    assert meta["quantization_version"] == 2
    assert meta["context_length"] == 2048
    assert meta["file_type"] == {"code": 15, "name": "Q4_K_M"}
    assert meta["metadata"]["general.architectures"] == ["Qwen2ForCausalLM"]
    assert meta["metadata"]["some.count"] == -7
    assert meta["metadata"]["some.flag"] is True
    assert abs(meta["metadata"]["some.ratio"] - 0.5) < 1e-9
    print("PASS: synthetic GGUF metadata parsed correctly")

def test_parameter_count_from_tensor_table():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "tensors.gguf"
        write_synthetic_gguf(
            path,
            kv={"general.architecture": (_VAL["string"], "qwen2")},
            tensors=[
                {"name": "token_embd.weight", "dims": [16, 8], "type": 0},   # 128 params, F32
                {"name": "output_norm.weight", "dims": [8], "type": 1},      # 8 params, F16
                {"name": "blk.0.attn_q.weight", "dims": [4, 4], "type": 12},  # 16 params, Q4_K
            ],
        )
        meta = analyze_gguf(path, include_sha256=False)

    assert meta["tensor_count"] == 3
    assert meta["parameter_count"] == 128 + 8 + 16
    assert meta["tensor_types"] == {"F32": 1, "F16": 1, "Q4_K": 1}
    names = [t["name"] for t in meta["tensor_infos"]]
    assert names == ["token_embd.weight", "output_norm.weight", "blk.0.attn_q.weight"]
    print("PASS: parameter count derived from tensor table ->", meta["parameter_count"])

def test_sha256_matches_hashlib():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "hash.gguf"
        write_synthetic_gguf(path, kv={"general.name": (_VAL["string"], "hashed")})
        expected = hashlib.sha256(path.read_bytes()).hexdigest()

        assert analyze_gguf(path, include_sha256=True)["sha256"] == expected
        assert sha256_file(path) == expected
    print("PASS: sha256 matches hashlib reference")

def test_invalid_magic_raises():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "bad.gguf"
        write_synthetic_gguf(path, kv={}, magic=b"NOTG")
        try:
            analyze_gguf(path)
        except GGUFMetadataError as exc:
            assert "not a GGUF" in str(exc)
            print("PASS: bad magic -> GGUFMetadataError")
            return
    raise AssertionError("expected GGUFMetadataError for bad magic")

def test_missing_file_raises():
    try:
        analyze_gguf("models/gguf/does-not-exist.gguf")
    except FileNotFoundError:
        print("PASS: missing file -> FileNotFoundError")
        return
    raise AssertionError("expected FileNotFoundError")

def test_file_type_and_tensor_name_mapping():
    # Legacy llama.cpp file_type numbering (matches the repo's Qwen2.5 GGUFs).
    assert file_type_name(1) == "F16"
    assert file_type_name(15) == "Q4_K_M"
    assert file_type_name(999) is None
    # Current ggml tensor type codes.
    assert tensor_type_name(12) == "Q4_K"
    assert tensor_type_name(14) == "Q6_K"
    assert tensor_type_name(0) == "F32"
    assert tensor_type_name(1) == "F16"
    print("PASS: file_type / tensor_type mapping")


def test_legacy_uint32_dims_fallback():
    """Legacy GGUF files (uint32 dims) must parse via the fallback path."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "legacy.gguf"
        # Hand-write a legacy layout: u32 dims after n_dims.
        buf = bytearray(b"GGUF")
        buf += struct.pack("<I", 3)
        buf += struct.pack("<Q", 1)  # 1 tensor
        buf += struct.pack("<Q", 1)  # 1 kv
        key = b"general.architecture"
        buf += struct.pack("<Q", len(key)) + key
        buf += struct.pack("<I", _VAL["string"])
        val = b"qwen2"
        buf += struct.pack("<Q", len(val)) + val
        name = b"output.weight"
        buf += struct.pack("<Q", len(name)) + name
        buf += struct.pack("<I", 2)       # n_dims
        buf += struct.pack("<I", 2048)    # u32 dims (legacy)
        buf += struct.pack("<I", 512)
        buf += struct.pack("<I", 12)      # type Q4_K
        buf += struct.pack("<Q", 0)       # offset
        path.write_bytes(bytes(buf))

        meta = analyze_gguf(path, include_sha256=False)

    assert meta["tensor_count"] == 1
    assert meta["tensor_infos"][0]["dims"] == [2048, 512]
    assert meta["parameter_count"] == 2048 * 512
    print("PASS: legacy uint32 dims parsed via fallback")

# ---------------------------------------------------------------------------
# model_footprint: size math
# ---------------------------------------------------------------------------

def test_footprint_totals_and_reduction():
    with tempfile.TemporaryDirectory() as tmp:
        fp16a = Path(tmp) / "fp16-00001-of-00002.gguf"
        fp16b = Path(tmp) / "fp16-00002-of-00002.gguf"
        q4 = Path(tmp) / "q4.gguf"
        fp16a.write_bytes(b"a" * 600)
        fp16b.write_bytes(b"b" * 400)
        q4.write_bytes(b"c" * 300)  # 30% of the fp16 total

        fp = compute_footprint([fp16a, fp16b], [q4])

    assert fp["fp16"]["total_bytes"] == 1000
    assert fp["fp16"]["shard_count"] == 2
    assert len(fp["fp16"]["files"]) == 2
    assert fp["q4_k_m"]["total_bytes"] == 300
    assert fp["reduction"]["bytes"] == 700
    assert fp["reduction"]["percent"] == 70.0  # 1 - 300/1000
    print("PASS: footprint totals + 70% reduction math")

def test_footprint_is_storage_only_not_speedup():
    with tempfile.TemporaryDirectory() as tmp:
        fp16 = Path(tmp) / "fp16.gguf"
        q4 = Path(tmp) / "q4.gguf"
        fp16.write_bytes(b"a" * 200)
        q4.write_bytes(b"b" * 50)
        fp = compute_footprint([fp16], [q4])

    text = json.dumps(fp)
    # Explicit labels + anti-claim framing: no positive performance claim.
    assert "Reference FP16 model" in fp["fp16"]["label"]
    assert "Validated optimized model" in fp["q4_k_m"]["label"]
    assert "does not imply any inference speedup" in fp["note"]
    assert "footprint comparison only" in fp["note"]
    # The reduction block carries only storage numbers, no performance metric.
    assert set(fp["reduction"]) == {"bytes", "mb", "percent"}
    assert "faster" not in text.lower()  # no positive-claim vocabulary
    print("PASS: footprint framing is storage-only, no performance claims")

def test_footprint_missing_file_raises():
    try:
        compute_footprint(["models/gguf/nope.gguf"], ["models/gguf/also-nope.gguf"])
    except FileNotFoundError:
        print("PASS: missing file -> FileNotFoundError")
        return
    raise AssertionError("expected FileNotFoundError")

# ---------------------------------------------------------------------------
# report: schema + measured/not-measured classification
# ---------------------------------------------------------------------------

def _sample_benchmark() -> dict:
    return {
        "engine_id": "llamacpp-optimized",
        "runtime": "llama.cpp",
        "model_id": "qwen2.5-0.5b-instruct-q4_k_m",
        "model_path": "/models/qwen2.5-0.5b-instruct-q4_k_m.gguf",
        "prompt": "Explain what an AI inference engine is.",
        "prompt_tokens": 9,
        "warmup": 1,
        "model_footprint_bytes": 1_940_000_000,
        "aggregates": {
            "runs": 5,
            "mean_latency_ms": 8000.0,
            "median_latency_ms": 7900.0,
            "p90_latency_ms": 8300.0,
            "mean_ttft_ms": 140.0,
            "mean_generated_tokens": 64.0,
            "mean_tokens_per_second": 7.9,
            "peak_memory_mb": 2342.0,
            "mean_cpu_percent": 666.0,
        },
        "per_run": [{"repeat": 1, "engine_latency_ms": 8000.0}],
        "records": ["results/benchmarks/llamacpp-optimized/benchmark-x.json"],
        "results_dir": "results/benchmarks/llamacpp-optimized",
    }


def _sample_report(benchmark: dict | None = None) -> dict:
    return build_optimization_report(
        project={"name": "ArmInferX", "phase": "STEP 11", "tool_versions": {}},
        hardware={"total_ram_gb": 7.63, "platform": "test"},
        optimized={
            "engine_id": "llamacpp-optimized",
            "runtime": "llama.cpp",
            "quantization": "Q4_K_M",
            "status": "validated",
        },
        model={"q4_k_m": {"file_name": "q4.gguf", "file_size_bytes": 100}},
        configuration={"repeats": 5, "warmup": 1},
        model_footprint={
            "fp16": {"total_bytes": 1000},
            "q4_k_m": {"total_bytes": 300},
            "reduction": {"percent": 70.0},
        },
        benchmark=benchmark,
        limitations=["test limitation"],
        reproducibility={"command": "pytest"},
    )


def test_report_has_all_required_sections():
    report = _sample_report(benchmark=_sample_benchmark())
    from optimization.report import REQUIRED_SECTIONS

    for section in REQUIRED_SECTIONS:
        assert section in report, f"missing section {section}"
    print("PASS: all required report sections present")

def test_report_measured_metrics_only_from_benchmark():
    report = _sample_report(benchmark=_sample_benchmark())
    measured = report["measured"]

    # Q4_K_M measured facts are present with the benchmark values.
    assert measured["q4_k_m_latency_mean_ms"] == 8000.0
    assert measured["q4_k_m_ttft_mean_ms"] == 140.0
    assert measured["q4_k_m_tokens_per_second_mean"] == 7.9
    assert measured["q4_k_m_peak_memory_mb"] == 2342.0
    assert measured["q4_k_m_prompt_tokens"] == 9
    assert measured["benchmark_repetitions"] == 5
    # No FP16 metric and no speedup may ever appear under "measured".
    for key in measured:
        assert not key.startswith("fp16_"), f"FP16 metric leaked into measured: {key}"
        assert "speedup" not in key and "faster" not in key
    print("PASS: measured block contains only Q4_K_M benchmark facts")

def test_report_not_measured_fp16_and_no_speedup():
    report = _sample_report(benchmark=_sample_benchmark())
    not_measured = report["not_measured"]

    # FP16 latency/throughput/TTFT + any speedup are explicitly null.
    assert not_measured["fp16_inference_latency_ms"] is None
    assert not_measured["fp16_ttft_ms"] is None
    assert not_measured["fp16_tokens_per_second"] is None
    assert not_measured["comparative_speedup_vs_fp16_percent"] is None
    # The explanation must say no speedup is claimed.
    assert "no percentage performance improvement" in report["not_measured_explanation"]
    # Feasibility block records the hardware limitation, not a measurement.
    assert report["feasibility"]["fp16"]["status"] == "not_feasible"
    assert report["feasibility"]["fp16"]["inference_completed"] is False
    assert "hardware memory constraint" in report["feasibility"]["fp16"]["classification"]
    print("PASS: FP16 metrics explicitly not measured; feasibility is a hardware limitation")

def test_report_without_benchmark_has_empty_measured():
    report = _sample_report(benchmark=None)
    assert report["measured"] == {}
    assert report["benchmark"] is None
    assert report["not_measured"]["fp16_inference_latency_ms"] is None
    print("PASS: no benchmark -> measured empty, not_measured still explicit")

def test_report_is_json_serializable():
    report = _sample_report(benchmark=_sample_benchmark())
    payload = json.dumps(report)
    assert json.loads(payload)["feasibility"]["fp16"]["status"] == "not_feasible"
    print("PASS: report round-trips through JSON")


if __name__ == "__main__":
    test_parse_synthetic_gguf_metadata()
    test_parameter_count_from_tensor_table()
    test_legacy_uint32_dims_fallback()
    test_sha256_matches_hashlib()
    test_invalid_magic_raises()
    test_missing_file_raises()
    test_file_type_and_tensor_name_mapping()
    test_footprint_totals_and_reduction()
    test_footprint_is_storage_only_not_speedup()
    test_footprint_missing_file_raises()
    test_report_has_all_required_sections()
    test_report_measured_metrics_only_from_benchmark()
    test_report_not_measured_fp16_and_no_speedup()
    test_report_without_benchmark_has_empty_measured()
    test_report_is_json_serializable()
    print(json.dumps({"result": "all optimization report tests passed"}))
