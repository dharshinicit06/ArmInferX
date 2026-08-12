"""HTTP router for the benchmark results domain.

Read-only endpoints over the records saved under ``results/baseline/`` by the
inference router. The router only adapts: raw data access goes through
``BaselineResultStore`` (infrastructure) and validation against the API's own
Pydantic contract happens here, so a malformed result file is skipped instead
of failing the request. Swagger documentation is generated automatically from
the response models, endpoint docstrings and ``responses`` metadata.
"""

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, status
from pydantic import ValidationError

from api.routes.benchmarks.schemas import BenchmarkRecord, BenchmarkSummary
from benchmark.storage import BaselineResultStore

logger = logging.getLogger(__name__)

router = APIRouter(tags=["benchmarks"])

# Stateless helper sharing the same results directory as the inference router.
result_store = BaselineResultStore()


def _load_records(engine_id: str | None = None) -> list[BenchmarkRecord]:
    """Load all saved records, skipping any that fail validation.

    The store only guarantees parseable JSON objects; a file that parses but
    has the wrong shape (e.g. a non-numeric ``latency_ms``) is treated as
    corrupt here and skipped with a warning, so one bad file can never break
    the read API.

    Args:
        engine_id: Optional filter — only records tagged with this engine id
            are returned. Records without an engine tag (legacy baseline runs)
            are only returned when no filter is given, so engine-specific views
            never mix in unidentified runs.
    """
    records: list[BenchmarkRecord] = []
    for raw in result_store.list_records():
        if engine_id is not None and raw.get("engine_id") != engine_id:
            continue
        try:
            records.append(BenchmarkRecord(**raw))
        except ValidationError:
            logger.warning(
                "Skipping invalid benchmark record (timestamp=%s)",
                raw.get("timestamp", "<unknown>"),
            )
    return records


def _compute_summary(records: list[BenchmarkRecord]) -> dict[str, Any]:
    """Aggregate average latency/memory/CPU over ``records``.

    Returns zeroed values for an empty list so the summary endpoint always has
    a stable shape.
    """
    total = len(records)
    if total == 0:
        return {
            "avg_latency_ms": 0.0,
            "avg_memory_mb": 0.0,
            "avg_cpu_percent": 0.0,
            "total_runs": 0,
        }
    return {
        "avg_latency_ms": round(sum(r.latency_ms for r in records) / total, 2),
        "avg_memory_mb": round(sum(r.memory_mb for r in records) / total, 2),
        "avg_cpu_percent": round(sum(r.cpu_percent for r in records) / total, 2),
        "total_runs": total,
    }


def _record_to_dict(record: BenchmarkRecord) -> dict[str, Any]:
    """Serialize a record for the HTTP response.

    Engine identity fields (``engine_id``/``runtime``) are omitted when null so
    legacy engine-less records keep their exact historical shape; other null
    metrics (e.g. missing ``generated_tokens``) are kept as explicit nulls so
    clients can distinguish "unavailable" from "zero" and old records stay
    readable.
    """
    data = record.model_dump()
    for key in ("engine_id", "runtime"):
        if data.get(key) is None:
            data.pop(key)
    return data


@router.get(
    "/benchmarks",
    response_model=None,
    responses={
        status.HTTP_200_OK: {
            "model": list[BenchmarkRecord],
            "description": "All saved benchmark records, oldest first.",
        },
    },
    summary="List all benchmark records",
    description=(
        "Returns every saved benchmark record from results/baseline/, ordered "
        "oldest first. Records are the auto-saved outputs of POST /generate, "
        "tagged with their engine when the engine exposed one. Pass engine_id "
        "to see only records from one engine (e.g. 'llamacpp-optimized'); "
        "untagged legacy records are excluded when filtering. Returns an empty "
        "list when no runs have been recorded yet. Corrupt result files "
        "(unparseable or schema-invalid) are skipped rather than failing the "
        "request."
    ),
)
def list_benchmarks(engine_id: str | None = None) -> list[dict[str, Any]]:
    return [_record_to_dict(r) for r in _load_records(engine_id=engine_id)]


@router.get(
    "/benchmarks/latest",
    response_model=None,
    responses={
        status.HTTP_200_OK: {
            "model": BenchmarkRecord,
            "description": "The most recent benchmark record.",
        },
        status.HTTP_404_NOT_FOUND: {
            "description": "No benchmark records found",
            "content": {
                "application/json": {
                    "examples": {"no_records": {"value": {"detail": "No benchmark records found"}}},
                }
            },
        },
    },
    summary="Get the latest benchmark record",
    description=(
        "Returns the most recently saved benchmark record from "
        "results/baseline/. Returns 404 when no benchmark runs have been "
        "recorded yet."
    ),
)
def latest_benchmark() -> dict[str, Any]:
    records = _load_records()
    if not records:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No benchmark records found",
        )
    return _record_to_dict(records[-1])


@router.get(
    "/benchmarks/summary",
    response_model=BenchmarkSummary,
    summary="Summarize benchmark runs",
    description=(
        "Returns aggregate statistics over all saved benchmark records: "
        "average latency (ms), average memory (MB), average CPU (%), and the "
        "total number of runs. Averages are 0 and total_runs is 0 when no "
        "runs have been recorded yet."
    ),
)
def benchmark_summary(engine_id: str | None = None) -> BenchmarkSummary:
    summary = _compute_summary(_load_records(engine_id=engine_id))
    logger.info(
        "Benchmark summary over %d records: avg_latency=%.2fms avg_memory=%.2fMB avg_cpu=%.2f%%",
        summary["total_runs"],
        summary["avg_latency_ms"],
        summary["avg_memory_mb"],
        summary["avg_cpu_percent"],
    )
    return BenchmarkSummary(**summary)
