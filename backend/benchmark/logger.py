"""Dedicated logging for the benchmarking subsystem.

Kept separate from the HTTP service loggers so benchmark records can be
filtered or redirected independently. Records are single-line and parseable
(e.g. ``armiferx.benchmark: Benchmark done: latency=...``).

``get_benchmark_logger()`` attaches a handler only when the process has no
logging configuration yet (e.g. standalone benchmark scripts), so it never
duplicates handlers when the FastAPI app has already configured the root
logger via ``api.utils.logging_config``.
"""

from __future__ import annotations

import logging

LOGGER_NAME = "armiferx.benchmark"
_DEFAULT_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def get_benchmark_logger(level: int = logging.INFO) -> logging.Logger:
    """Return the module-level benchmark logger, configured for standalone use."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    if not logger.handlers and not logging.getLogger().handlers:
        logging.basicConfig(level=level, format=_DEFAULT_FORMAT)
    return logger
