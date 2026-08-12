"""Time-to-first-token (TTFT) instrumentation for ArmInferX.

``model.generate(streamer=...)`` pushes each sampled token to the streamer's
``put()`` the moment it is produced. ``TTFTStreamer`` only observes those
tokens — it never buffers or modifies them — so attaching it does not change
the generation output or the tokens-per-second/latency behavior of the
service. It records the wall-clock elapsed from its own creation (immediately
before ``model.generate()`` starts, i.e. the start of inference) until the
first token arrives. That interval covers the prompt prefill plus the first
decode step, which is the standard definition of time-to-first-token.
"""

from __future__ import annotations

import threading
import time

from transformers.generation.streamers import BaseStreamer


class TTFTStreamer(BaseStreamer):
    """Measures the elapsed time until the first generated token is produced.

    Usage (handled internally by :class:`~api.routes.inference.inference_service.InferenceService`):

        streamer = TTFTStreamer()          # timestamp ≈ start of inference
        model.generate(..., streamer=streamer)
        ttft_ms = streamer.ttft_ms         # None if no token ever arrived

    The value is computed with ``time.perf_counter`` (the same monotonic clock
    used by the latency ``Stopwatch``). ``put()`` is thread-safe: the first
    arrival wins and later tokens never overwrite it.
    """

    def __init__(self) -> None:
        self._start = time.perf_counter()
        self._ttft_ms: float | None = None
        self._lock = threading.Lock()

    def put(self, value) -> None:  # noqa: ARG002 - token value is irrelevant
        """Record the elapsed time on the first token arrival, then ignore."""
        with self._lock:
            if self._ttft_ms is None:
                self._ttft_ms = (time.perf_counter() - self._start) * 1000.0

    def end(self) -> None:
        """Signal the end of generation. Nothing to do for TTFT."""

    @property
    def ttft_ms(self) -> float | None:
        """Time-to-first-token in milliseconds, or ``None`` if no token arrived."""
        with self._lock:
            return self._ttft_ms
