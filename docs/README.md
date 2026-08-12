# ArmInferX — Documentation

## Guides

| Document | What it covers |
|---|---|
| [`architecture.md`](architecture.md) | Component map: frontend → FastAPI → EngineManager → engine registry → benchmark/storage; lazy loading, caching, engine selection, streaming, persistence, report + dashboard data flow (STEP 15). |
| [`demo-runbook.md`](demo-runbook.md) | The exact demo script for the Windows laptop (no Docker, no ARM64): start backend + frontend, stream a prompt, show metadata, the Optimization Dashboard, and the two honest caveats (STEP 15). |
| [`optimization.md`](optimization.md) | STEP 11 measured optimization evidence: the Q4_K_M llama.cpp story, measured vs not-measured boundary, benchmark methodology, hardware limitation. |
| [`arm64-deployment.md`](arm64-deployment.md) | STEP 14A ARM64 Docker deployment preparation: image build flags, model bind-mount, verification, the one-shot smoke script. Preparation only — no Arm64 numbers. |
| [`step14b-runbook.md`](step14b-runbook.md) | STEP 14B runbook to execute on a real Linux ARM64 host. **Postponed** until such a host exists. |
| [`final-demo-checklist.md`](final-demo-checklist.md) | The 3–5 minute hackathon demo sequence (A–L) with timing budget and failure fallbacks (STEP 16). |
| [`final-validation.md`](final-validation.md) | STEP 16 final validation report: environment, model integrity, tests, smoke test, evidence, documentation, git status. |
| [`cloud-deploy.md`](cloud-deploy.md) | Cloud deployment runbook (ARM64 Graviton path A / AMD64 VPS path B): frontend service, VITE_API_URL, CORS, firewall, verification. **Not yet executed.** |

## Benchmark methodology

The engine-agnostic benchmark procedure (identical for both engines, greedy,
1 warmup + 5 timed runs) is documented in
[`backend/benchmark/README.md`](../backend/benchmark/README.md).
Root-level [`benchmark/README.md`](../benchmark/README.md) has tooling notes.

## Evidence

- Canonical measured values: `results/optimization_report.json` (gitignored
  generated artifact; regenerate with
  `backend/.venv/Scripts/python scripts/run_optimization_report.py`).
- Dashboard fallback copy: `frontend/public/optimization-report.json`.
