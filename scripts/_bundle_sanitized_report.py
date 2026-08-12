"""Write a path-free copy of the optimization report for the frontend bundle.

Uses the same sanitizer as the backend router (api.routes.optimization.router)
so the bundled static fallback never ships filesystem paths to the browser.
Run: python scripts/_bundle_sanitized_report.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from api.routes.optimization.router import _sanitize_report  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT_ROOT / "results" / "optimization_report.json"
TARGET = PROJECT_ROOT / "frontend" / "public" / "optimization-report.json"

if not SOURCE.is_file():
    raise SystemExit(f"Report not found: {SOURCE}")

data = json.loads(SOURCE.read_text(encoding="utf-8"))
clean = _sanitize_report(data)
TARGET.write_text(json.dumps(clean, indent=2), encoding="utf-8")

# Guard: no absolute Windows paths may survive into the browser bundle.
serialized = json.dumps(clean)
assert "D:\\" not in serialized, "filesystem path leaked into bundled report"
print(f"sanitized copy written: {TARGET} ({TARGET.stat().st_size} bytes)")
print(f"evidence preserved: mean_latency={clean['benchmark']['aggregates']['mean_latency_ms']}ms "
      f"reduction={clean['model_footprint']['reduction']['percent']}% "
      f"measured_keys={len(clean['measured'])} not_measured_keys={len(clean['not_measured'])}")
