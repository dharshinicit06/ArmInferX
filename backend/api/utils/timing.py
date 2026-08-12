"""Baseline latency measurement — wall-clock only.

This is the *baseline* latency implementation for ArmInferX. It measures only
inference wall-clock time (start -> end) using Python's ``time`` module. It
deliberately does NOT measure CPU usage, memory, tokens/sec or throughput —
those belong to the future benchmarking engine.

The stopwatch is a small reusable context manager so every inference path
(the HTTP service today, benchmark harnesses later) applies the same
measurement.
"""

from __future__ import annotations

import time


class Stopwatch:
    """Wall-clock stopwatch for measuring inference latency.

    Usage::

        with Stopwatch() as timer:
            output = model.generate(**inputs)
        latency_ms = timer.latency_ms

    ``start`` and ``end`` (seconds, from ``time.perf_counter``) are kept for
    transparency and logging; ``latency_ms`` is the derived value in
    milliseconds. Reading ``latency_ms`` outside the ``with`` block (before
    both times are recorded) raises ``RuntimeError``.
    """

    __slots__ = ("_start", "_end")

    def __init__(self) -> None:
        self._start: float | None = None
        self._end: float | None = None

    def __enter__(self) -> "Stopwatch":
        # Start time: recorded immediately before the timed block executes.
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc_info) -> None:
        # End time: recorded as soon as the timed block finishes.
        self._end = time.perf_counter()

    @property
    def start(self) -> float | None:
        return self._start

    @property
    def end(self) -> float | None:
        return self._end

    @property
    def latency_ms(self) -> float:
        """Elapsed wall-clock time in milliseconds (``end - start``)."""
        if self._start is None or self._end is None:
            raise RuntimeError(
                "Stopwatch must be used as a context manager before reading latency_ms"
            )
        return (self._end - self._start) * 1000.0
