# ArmInferX — Cloud Deployment Runbook

This document explains how to put ArmInferX behind a **live URL** so judges (or
anyone) can open the demo from a browser without touching your laptop.

> **Status: NOT yet executed.** STEP 14B (real ARM64 deployment + validation)
> remains **postponed** until a real Linux ARM64 host is available, and no
> cloud deployment has been run. This page is a step-by-step plan built from
> the artifacts that exist in the repo. Nothing here claims Arm64 benchmark
> results or a successful cloud deployment.

---

## Two deployment paths

| Path | Best for | Effort | Prereqs |
|---|---|---|---|
| **A. ARM64 cloud VM (Graviton / Ampere)** | The *actual* ArmInferX story: Arm64 cloud inference | ~1–2 h | AWS account, ~8 GiB RAM VM, $/hr cost |
| **B. AMD64 VPS** | Fastest way to a working public URL | ~30–45 min | Any Linux VPS ≥ 8 GiB RAM, Docker + Compose |

Both paths run **CPU-only llama.cpp with the same Q4_K_M GGUF**
(Qwen2.5-0.5B-Instruct). Path B is a quick public demo; Path A additionally
exercises the ARM64 deployment story (`docs/arm64-deployment.md` + the STEP
14B runbook `docs/step14b-runbook.md`).

> **Memory requirement (both paths):** the 0.5B Q4_K_M model loads to roughly
> **0.5–0.8 GB RSS**, so a **1 GB RAM instance (e.g. Railway) or a 2 GiB VM is
> sufficient**. Never select the FP16 Transformers baseline on such machines —
> it needs ≥ 16 GiB.

---

## What gets deployed

```
Browser  ──►  Frontend (nginx :80)   ──►   Backend (FastAPI :8000)   ──►  llama.cpp + Q4_K_M
              (static Vite build)            (uvicorn, CPU-only)          (GGUF bind-mounted)
```

| Piece | Where | Notes |
|---|---|---|
| Backend | `docker/backend/Dockerfile` | Multi-stage, llama.cpp-only, non-root runtime |
| Frontend | `docker/frontend/Dockerfile` + `nginx.conf` | Multi-stage (node build → nginx static), SPA fallback |
| Compose | `docker-compose.yml` | `backend` (:8000) + `frontend` (:8080→:80) services |
| Model | `models/gguf/qwen2.5-0.5b-instruct-q4_k_m.gguf` | Downloaded into the image at build time, SHA-256-verified (gitignored, ~470 MB); the compose bind-mount below is optional |
| CORS | `ARMINFERX_CORS_ORIGINS` env var on the backend | Backend allows the deployed frontend origin |
| API URL | `VITE_API_URL` build arg on the frontend | Baked into the JS bundle so the UI knows where the API lives |

The frontend calls the backend at an **absolute** URL (`VITE_API_URL`, default
`http://localhost:8000`), and the backend only allows known browser origins
(`localhost:3000/5173/4173` + `ARMINFERX_CORS_ORIGINS`). Cloud deployment
therefore needs exactly two configuration values:

1. `VITE_API_URL` — where the browser can reach the backend (the VM's public
   IP/hostname + port 8000).
2. `ARMINFERX_CORS_ORIGINS` — the frontend origin the browser is served from.

---

## Path B — AMD64 VPS (fastest public demo)

### 1. Provision

Any Linux VPS with **≥ 8 GiB RAM**, Docker + Docker Compose plugin installed.
(Example: a 2 vCPU / 8 GiB droplet; exact provider is your choice.)

### 2. Get the code and the model on the VM

```bash
git clone <your-repo-url> /opt/arminferx && cd /opt/arminferx
# The image downloads + verifies the GGUF at build time, so no manual copy is
# required. Only if you prefer the read-only bind-mount path (docker-compose
# mounts ./models/gguf): copy models/gguf/qwen2.5-0.5b-instruct-q4_k_m.gguf
# there yourself, e.g.
#   scp models/gguf/qwen2.5-0.5b-instruct-q4_k_m.gguf user@<VM-IP>:~/arminferx/models/gguf/
mkdir -p models/gguf
# Optional identity check of the file you copied:
sha256sum models/gguf/qwen2.5-0.5b-instruct-q4_k_m.gguf
# expected: 74a4da8c…a9db  (the exact verified hash)
```

### 3. Configure and build

```bash
SERVER_IP=<the VM's public IP or domain>
export VITE_API_URL=http://$SERVER_IP:8000
export ARMINFERX_CORS_ORIGINS=http://$SERVER_IP:8080
docker compose up --build -d
```

What happens:
- `frontend` builds with the API URL baked in, serves the SPA on the VM's
  port **8080** (`8080:80`).
- `backend` starts with nothing loaded (lazy `EngineManager`), downloads and
  SHA-256-verifies the GGUF during the image build, and allows the frontend
  origin.

### 4. Open your firewall

Allow TCP **8080** (frontend) and **8000** (backend) from the internet (or
from wherever the judges will connect).

### 5. Verify

```bash
curl -fsS http://localhost:8000/health          # loaded_engines: [] before use
curl -fsS -X POST http://localhost:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Say hello in one short sentence.","engine_id":"llamacpp-optimized"}'
# then: curl -fsS http://localhost:8000/health  # loaded_engines: ["llamacpp-optimized"]
```

Open `http://$SERVER_IP:8080` in a browser → Studio + Optimization Dashboard
work, `API · http://$SERVER_IP:8000` in the header.

---

## Path A — ARM64 cloud VM (the ArmInferX story)

Follow this runbook *plus* the existing **STEP 14A / STEP 14B** material, which
were written exactly for this moment:

1. **Provision an ARM64 VM** — e.g. AWS Graviton (`t4g.2xlarge` = 8 vCPU /
   32 GiB, or `c7g.large` if you must go smaller but stay ≥ 2 GiB), Ubuntu
   22.04/24.04, Docker installed.
2. **Copy the repo + GGUF** exactly as in Path B step 2.
3. **Build the ARM64 image natively** on the ARM64 host (no emulation needed):

   ```bash
   docker compose build backend     # or: docker build -f docker/backend/Dockerfile -t arminferx-backend .
   ```

4. **Execute STEP 14B properly** — this is the *intended* use of
   `docs/step14b-runbook.md` and the one-shot `scripts/arm64_deploy_smoke.sh`:

   ```bash
   bash scripts/arm64_deploy_smoke.sh    # env report → build → arch check → SHA-256 → health → 2 real generates
   ```

   That script is the full STEP 14B acceptance sequence and stops with a FAIL
   at the exact stage on any problem.
5. **Optionally run the existing 5-repeat benchmark** for genuine Arm64
   numbers (this is what STEP 14B would produce — do not present anything as
   measured until it is):

   ```bash
   docker compose exec backend python /app/scripts/run_llamacpp_benchmark.py
   ```

6. **Frontend + CORS**: same as Path B steps 3–5
   (`VITE_API_URL=http://$SERVER_IP:8000`,
   `ARMINFERX_CORS_ORIGINS=http://$SERVER_IP:8080`).

> If you run the benchmark on the VM, save the output JSON and update
> `results/optimization_report.json` / the dashboard's fallback copy —
> **then** it becomes measured evidence. Until then, keep every
> measured/not-measured label exactly as shipped.

---

## Environment variables reference

| Variable | Where | Default | Meaning |
|---|---|---|---|
| `VITE_API_URL` | compose build arg → frontend image | `http://localhost:8000` | API base URL baked into the JS bundle |
| `ARMINFERX_CORS_ORIGINS` | compose env → backend | empty | Extra comma-separated browser origins the API allows |
| `ARMINFERX_DEFAULT_ENGINE` | compose env → backend | `llamacpp-optimized` | Default engine (unchanged) |
| `ARMINFERX_DEVICE` | compose env → backend | `cpu` | CPU-only inference (unchanged) |
| `INSTALL_BASELINE_DEPS` | backend build arg | `0` | Set `1` only on ≥ 16 GiB hosts to build the FP16 baseline deps |

Engine defaults (`n_ctx=1024`, `n_threads=2`, `n_gpu_layers=0`, greedy,
`max_new_tokens=64` for the benchmark) are tuned for 1 GB RAM / 2 vCPU hosts
and are not overridden anywhere.

---

## Security / hygiene notes

- The API has **no authentication**. For a hackathon demo that is acceptable,
  but do not expose port 8000 with the FP16 baseline enabled or on a machine
  with data you care about. Prefer restricting the firewall to demo-time.
- Do **not** set `INSTALL_BASELINE_DEPS=1` on the cloud VM unless you have
  ≥ 16 GiB RAM — the FP16 baseline is deliberately out of scope for this
  deployment.
- The GGUF is mounted read-only; the container cannot modify it.
- `results/` persistence on the VM is optional (bind-mount it if you want the
  benchmark records kept — the container writes as uid 1000; create the
  directory writable by that user first).

---

## Cleanup after the demo

```bash
docker compose down          # stop
docker rmi arminferx-backend arminferx-frontend   # remove images (optional)
# Terminate the VM to stop billing — cloud VMs bill while running.
```
