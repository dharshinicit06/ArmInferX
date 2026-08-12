"""Benchmark results domain: reading and summarizing saved baseline records.

Convenience re-exports so callers can import the public surface from the
package instead of individual modules.
"""

from api.routes.benchmarks.router import router
from api.routes.benchmarks.schemas import BenchmarkRecord, BenchmarkSummary

__all__ = [
    "BenchmarkRecord",
    "BenchmarkSummary",
    "router",
]
