# ArmInferX — STEP 14B Runbook (execute on the Linux ARM64 host)

Run every command **on the ARM64 host, from the repo root**. Paste the full
transcript (or `step14b.log`) back afterwards; the structured STEP 14B report
is produced from it.

> Expected values used throughout:
> - model SHA-256: `74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db`
> - model size: `491400032` bytes
> - llama-cpp-python: `0.3.34` · image arch: `linux/arm64`
> - default engine: `llamacpp-optimized` · engine defaults `1024 / 2 / 0 / 0.0 / 64`

---

## Option A — one-shot script (recommended; runs every check)

```bash
cd /path/to/ArmInferX
bash scripts/arm64_deploy_smoke.sh 2>&1 | tee step14b.log
```

The script performs all 12 stages with PASS/FAIL per stage and stops at the
exact failing stage. Use `--skip-build` to reuse an already-built image.

---

## Option B — manual step-by-step

### [0] Pre-flight (confirm repo + model are on the host)

```bash
pwd                                   # must be the repo root
git rev-parse --show-toplevel         # same path, or "not a git repo" (fine)
ls -la models/gguf/qwen2.5-0.5b-instruct-q4_k_m.gguf
```

If the model file is missing, copy it from the dev machine before continuing
(it is gitignored, so it is not part of the repo clone):

```bash
# run on the DEV machine, not here:
scp -r models/gguf user@<arm64-host>:<repo-path>/models/
```

### [1] Environment validation

```bash
uname -m                              # MUST print aarch64 (else STOP)
nproc                                 # core count
free -h                               # total + available RAM (keep >= 4 GiB free)
docker --version
docker compose version
```

### [2] Model validation (do not download/quantize/modify)

```bash
sha256sum models/gguf/qwen2.5-0.5b-instruct-q4_k_m.gguf
```

Expected output ends with `74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db`.

### [3] Build the ARM64 image

```bash
docker compose build backend 2>&1 | tee build.log
docker image inspect arminferx-backend:latest --format '{{.Os}}/{{.Architecture}}'
# expect: linux/arm64
docker compose run --rm backend python -c "import llama_cpp; print(llama_cpp.__version__)"
# expect: 0.3.34
grep -nE -- "--no-binary|GGML_NATIVE=OFF|GGML_OPENMP=ON" docker/backend/Dockerfile
```

If the build fails: capture the tail of `build.log`, do **not** change the
compiler flags, and report the error back before proceeding.

### [4] Start + model mount verification

```bash
docker compose up -d backend
docker compose exec backend ls -la /app/models/gguf/
docker compose exec backend python -c "
import hashlib, os
p = '/app/models/gguf/qwen2.5-0.5b-instruct-q4_k_m.gguf'
print('size  =', os.path.getsize(p))
h = hashlib.sha256()
with open(p, 'rb') as f:
    for chunk in iter(lambda: f.read(1024 * 1024), b''):
        h.update(chunk)
print('sha256 =', h.hexdigest())
"
```

Both must match the host values above.

### [5] Startup + /health BEFORE inference (lazy loading proof)

```bash
docker compose logs backend --tail=20     # uvicorn started, NO model-load line
curl -s http://localhost:8000/health
# expect: "default_engine": "llamacpp-optimized", "loaded_engines": []
```

If `curl` is missing on the host, use the container python instead:

```bash
docker compose exec backend python -c "import json,urllib.request; print(json.dumps(json.load(urllib.request.urlopen('http://127.0.0.1:8000/health')), indent=2))"
```

### [6] Real ARM64 inference — request 1

```bash
curl -s -X POST http://localhost:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Hello! Who are you?","engine_id":"llamacpp-optimized"}'
```

Expected fields: `status=success`, non-empty `response`, `engine_id=llamacpp-optimized`,
`runtime=llama.cpp`, `model=qwen2.5-0.5b-instruct-q4_k_m`, `generated_tokens>0`,
`latency_ms>0`, `tokens_per_second>0`, `ttft_ms` when available. First request
includes the one-time model load.

### [7] Caching validation — request 2 + /health AFTER

```bash
curl -s -X POST http://localhost:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Hello! Who are you?","engine_id":"llamacpp-optimized"}'
curl -s http://localhost:8000/health
# expect: "loaded_engines": ["llamacpp-optimized"]
```

The second request is served by the **cached** engine (no reload). The
first-vs-second latency difference is lifecycle evidence only — do not
interpret it as a performance/speedup claim.

### [8] Engine configuration validation

```bash
docker compose exec backend python -c "
from engines.llamacpp_optimized import (
    DEFAULT_N_CTX, DEFAULT_N_THREADS, N_GPU_LAYERS,
    DEFAULT_TEMPERATURE, DEFAULT_MAX_NEW_TOKENS,
)
print(DEFAULT_N_CTX, DEFAULT_N_THREADS, N_GPU_LAYERS, DEFAULT_TEMPERATURE, DEFAULT_MAX_NEW_TOKENS)
"
# expect: 1024 2 0 0.0 64  (n_ctx, n_threads, n_gpu_layers, temperature/greedy, max_new_tokens)
```

### [9] Baseline safety (mandatory)

- Do **not** load or benchmark `transformers-baseline`.
- Do **not** run the FP16 model or any FP16 comparison (documented limitation).
- Do **not** run the 1-warmup + 5-timed benchmark (that is STEP 14C).

---

## What to paste back

Either the full `step14b.log` (Option A) or, for Option B, the outputs of:
`[1]` (uname -m, nproc, free -h, docker versions), `[2]` sha256, `[3]` image
arch + llama_cpp version, `[4]` container size + sha256, `[5]` /health JSON,
`[6]` first response JSON, `[7]` second response JSON + /health JSON, `[8]`
config line, and any build warnings/errors.
