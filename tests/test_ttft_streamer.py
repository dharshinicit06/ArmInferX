"""Unit tests for the TTFT streamer used to measure time-to-first-token.

No model or GPU is needed: the streamer is driven directly, the same way
``model.generate(streamer=...)`` drives it — one ``put()`` per produced token
and a final ``end()``.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from api.routes.inference.streamer import TTFTStreamer  # noqa: E402


def test_ttft_is_none_until_first_token():
    streamer = TTFTStreamer()
    assert streamer.ttft_ms is None
    # Simulate a generation that never produces a token (e.g. immediate stop).
    streamer.end()
    assert streamer.ttft_ms is None
    print("PASS: ttft_ms is None before any token arrives")


def test_ttft_measures_elapsed_time_to_first_token():
    streamer = TTFTStreamer()
    time.sleep(0.05)  # simulate prefill + first decode step
    streamer.put(123)  # first generated token becomes available
    ttft = streamer.ttft_ms
    assert ttft is not None
    assert 40 <= ttft <= 200, ttft  # ~50ms sleep, generous bounds
    print(f"PASS: ttft_ms ~50ms sleep -> {ttft:.1f} ms")


def test_ttft_is_not_overwritten_by_later_tokens():
    streamer = TTFTStreamer()
    time.sleep(0.02)
    streamer.put(10)
    first = streamer.ttft_ms
    time.sleep(0.05)  # more tokens arrive after the first one
    streamer.put(11)
    streamer.put(12)
    streamer.end()
    assert streamer.ttft_ms == first, "TTFT must freeze at the first token"
    print(f"PASS: ttft_ms frozen at first token -> {first:.1f} ms")


if __name__ == "__main__":
    test_ttft_is_none_until_first_token()
    test_ttft_measures_elapsed_time_to_first_token()
    test_ttft_is_not_overwritten_by_later_tokens()
