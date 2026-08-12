"""Unit tests for the baseline result store.

Verifies unique filenames, JSON content, and that write errors are wrapped in
a typed ResultWriteError. No model, GPU, or network needed.
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from benchmark.storage import BaselineResultStore, ResultWriteError  # noqa: E402


def test_save_writes_unique_json_files():
    with tempfile.TemporaryDirectory() as tmp:
        store = BaselineResultStore(tmp)
        p1 = store.save({"prompt": "a", "latency_ms": 1.0})
        p2 = store.save({"prompt": "b", "latency_ms": 2.0})

        assert p1.exists() and p2.exists()
        assert p1 != p2
        assert p1.parent == Path(tmp)
        assert p1.name.startswith("baseline-") and p1.suffix == ".json"
        assert json.loads(p1.read_text(encoding="utf-8"))["prompt"] == "a"
        print("PASS: unique files ->", p1.name, p2.name)


def test_list_records_returns_records_oldest_first():
    with tempfile.TemporaryDirectory() as tmp:
        store = BaselineResultStore(tmp)
        older = {"timestamp": "2026-08-07T04:00:00+00:00", "latency_ms": 100.0}
        newer = {"timestamp": "2026-08-07T06:00:00+00:00", "latency_ms": 300.0}
        middle = {"timestamp": "2026-08-07T05:00:00+00:00", "latency_ms": 200.0}
        store.save(older)
        store.save(newer)
        store.save(middle)

        records = store.list_records()
        assert [r["latency_ms"] for r in records] == [100.0, 200.0, 300.0]
        assert store.latest()["latency_ms"] == 300.0
        print("PASS: list_records oldest-first ->", [r["latency_ms"] for r in records])


def test_list_records_empty_and_latest_none():
    with tempfile.TemporaryDirectory() as tmp:
        store = BaselineResultStore(tmp)
        assert store.list_records() == []
        assert store.latest() is None
        print("PASS: empty store -> [] and None")


def test_list_records_skips_corrupt_and_foreign_files():
    with tempfile.TemporaryDirectory() as tmp:
        store = BaselineResultStore(tmp)
        store.save({"timestamp": "2026-08-07T04:00:00+00:00", "latency_ms": 1.0})
        (Path(tmp) / "baseline-20260807-000000-000000-deadbeef.json").write_text(
            "{ not valid json", encoding="utf-8"
        )
        (Path(tmp) / "baseline-20260807-000000-000000-cafebabe.json").write_text(
            "[1, 2, 3]", encoding="utf-8"
        )
        (Path(tmp) / "notes.txt").write_text("ignored", encoding="utf-8")

        records = store.list_records()
        assert len(records) == 1 and records[0]["latency_ms"] == 1.0
        print("PASS: corrupt/foreign files skipped ->", len(records), "record(s)")


def test_save_raises_result_write_error():
    with tempfile.TemporaryDirectory() as tmp:
        blocker = Path(tmp) / "blocker"
        blocker.write_text("a file, not a directory", encoding="utf-8")
        store = BaselineResultStore(blocker)  # mkdir(parents=True) will fail
        try:
            store.save({"prompt": "x"})
        except ResultWriteError as exc:
            assert "blocker" in str(exc)
            print("PASS: write error wrapped as ResultWriteError ->", exc)
            return
    raise AssertionError("expected ResultWriteError")


if __name__ == "__main__":
    test_save_writes_unique_json_files()
    test_list_records_returns_records_oldest_first()
    test_list_records_empty_and_latest_none()
    test_list_records_skips_corrupt_and_foreign_files()
    test_save_raises_result_write_error()
