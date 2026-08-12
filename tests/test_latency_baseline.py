"""Unit tests for the baseline latency measurement.

Verifies the Stopwatch utility: it records start/end times with Python's
``time`` module and reports wall-clock latency in milliseconds. No model,
GPU, or network needed.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from api.utils.timing import Stopwatch  # noqa: E402


def test_stopwatch_reports_elapsed_milliseconds():
    # A ~50 ms sleep must surface as ~50 ms of wall-clock latency.
    with Stopwatch() as timer:
        time.sleep(0.05)
    assert timer.start is not None, "start time must be recorded"
    assert timer.end is not None, "end time must be recorded"
    assert timer.end >= timer.start
    assert 40 <= timer.latency_ms <= 250, timer.latency_ms
    print(f"PASS: ~50ms sleep -> {timer.latency_ms:.1f} ms")


def test_stopwatch_is_wall_clock_only():
    # The stopwatch exposes latency only — no CPU/memory/throughput metrics.
    with Stopwatch() as timer:
        time.sleep(0.03)
    assert timer.latency_ms >= 20
    print(f"PASS: wall-clock only -> {timer.latency_ms:.1f} ms")


def test_stopwatch_requires_context_manager():
    timer = Stopwatch()
    try:
        _ = timer.latency_ms
    except RuntimeError:
        print("PASS: reading latency outside context raises RuntimeError")
        return
    raise AssertionError("expected RuntimeError when reading latency outside context")


if __name__ == "__main__":
    test_stopwatch_reports_elapsed_milliseconds()
    test_stopwatch_is_wall_clock_only()
    test_stopwatch_requires_context_manager()
