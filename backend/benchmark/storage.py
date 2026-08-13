"""JSON persistence for benchmark results.

Infrastructure layer: serializes and writes result records to unique JSON
files (atomic temp-file + rename) under a configurable directory with a
configurable filename prefix. It knows nothing about how records are produced
— callers pass a plain dict.

Two stores are provided:

- ``BaselineResultStore`` — the original store: writes ``baseline-*.json``
  under ``results/baseline/`` (repo root, gitignored). Behavior and file
  layout are unchanged, so existing benchmark records keep loading and no old
  file is rewritten.
- ``EngineResultStore`` — engine-aware store: writes ``benchmark-*.json``
  under ``results/benchmarks/<engine_id>/`` so results from different engines
  are distinguishable by directory without touching the baseline store.

Any ``OSError`` during a write is wrapped in a typed ``ResultWriteError`` so
callers can decide whether a persistence failure is fatal.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# Repo root is 2 levels up from this module (backend/benchmark/storage.py).
PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Original baseline results location (kept for backward compatibility).
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results" / "baseline"
# Engine-aware results location: results/benchmarks/<engine_id>/.
ENGINE_RESULTS_DIR = PROJECT_ROOT / "results" / "benchmarks"


class ResultWriteError(RuntimeError):
    """Raised when a benchmark result cannot be persisted to disk."""


class JsonResultStore:
    """Writes JSON records atomically under a directory with a filename prefix."""

    #: Filename prefix, e.g. "baseline-" or "benchmark-".
    filename_prefix: str = "result-"
    #: Default root directory used when no explicit directory is given.
    default_results_dir: Path = DEFAULT_RESULTS_DIR

    def __init__(self, root_dir: str | Path | None = None) -> None:
        self._root = Path(root_dir) if root_dir else self.default_results_dir

    @property
    def root_dir(self) -> Path:
        return self._root

    def save(self, record: dict) -> Path:
        """Serialize ``record`` and write it atomically to a unique file.

        The filename combines a UTC timestamp (with microseconds) and a random
        suffix, so names are unique even for concurrent requests. Raises
        ``ResultWriteError`` if the directory cannot be created or the file
        cannot be written.
        """
        try:
            self._root.mkdir(parents=True, exist_ok=True)
            path = self._root / self._unique_filename()
            tmp_path = path.with_suffix(".tmp")
            tmp_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
            tmp_path.replace(path)  # atomic on the same filesystem
        except OSError as exc:
            raise ResultWriteError(
                f"Failed to write benchmark result under {self._root}: {exc}"
            ) from exc
        logger.info("Saved benchmark result to %s", path)
        return path

    def list_records(self) -> list[dict]:
        """Return all saved records, oldest first.

        Only files matching the store's ``<prefix>*.json`` pattern are read.
        Files that cannot be parsed (corrupt, partial, or foreign) are skipped
        with a warning so a single bad file never breaks the read API. Order
        follows each record's ``timestamp`` field, so it is independent of
        filesystem enumeration order.
        """
        records: list[dict] = []
        if not self._root.is_dir():
            return records
        for path in sorted(self._root.glob(f"{self.filename_prefix}*.json")):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("Skipping unreadable benchmark record %s: %s", path, exc)
                continue
            if isinstance(record, dict):
                records.append(record)
            else:
                logger.warning("Skipping non-object benchmark record %s", path)
        records.sort(key=lambda r: r.get("timestamp", ""))
        return records

    def latest(self) -> dict | None:
        """Return the most recently saved record, or ``None``."""
        records = self.list_records()
        return records[-1] if records else None

    def _unique_filename(self) -> str:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
        return f"{self.filename_prefix}{stamp}-{uuid.uuid4().hex[:8]}.json"


class BaselineResultStore(JsonResultStore):
    """The original baseline results store (behavior unchanged).

    Writes ``baseline-*.json`` under ``results/baseline/`` by default.
    """

    filename_prefix = "baseline-"
    default_results_dir = DEFAULT_RESULTS_DIR


class EngineResultStore(JsonResultStore):
    """Engine-aware results store.

    Writes ``benchmark-*.json`` under ``results/benchmarks/<engine_id>/`` by
    default, so results from different engines are stored in separate
    directories and are distinguishable without touching the baseline store.
    Pass ``root_dir`` to redirect (used by tests).

    Args:
        engine_id: Stable engine identifier (e.g. ``llamacpp-optimized``).
            Used as the results subdirectory.
        root_dir: Optional explicit destination directory; defaults to
            ``results/benchmarks/<engine_id>/``.
    """

    filename_prefix = "benchmark-"

    def __init__(self, engine_id: str, root_dir: str | Path | None = None) -> None:
        if not isinstance(engine_id, str) or not engine_id.strip():
            raise ValueError("engine_id must be a non-empty string")
        self.engine_id = engine_id
        default_dir = ENGINE_RESULTS_DIR / engine_id
        super().__init__(root_dir or default_dir)
