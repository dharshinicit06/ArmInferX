"""Centralized logging configuration for the ArmInferX API.

Call ``configure_logging()`` once at process start (in ``main.py``) so every
module logs through a single, consistent handler and format.
"""

from __future__ import annotations

import logging
import sys

_DEFAULT_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def configure_logging(level: int = logging.INFO, format_: str = _DEFAULT_FORMAT) -> None:
    """Idempotently configure the root logger with a single stdout handler.

    Replacing ``root.handlers`` (rather than appending) keeps the config
    deterministic across reloads and test runners.
    """
    root = logging.getLogger()
    root.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(format_))
    root.handlers = [handler]
