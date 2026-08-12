#!/usr/bin/env bash
# =============================================================================
# ArmInferX STEP 14B — ARM64 Docker build, deployment & smoke test (one-shot)
# =============================================================================
#
# Runs the full STEP 14B sequence on a REAL Linux ARM64 host:
#   [1] environment report      uname -m / RAM / docker / compose versions
#   [2] model pre-flight        Q4_K_M exists on host + SHA-256 matches
#   [3] build                   docker compose build backend (STEP 14A config)
#   [4] image architecture      must be linux/arm64
#   [5] dependency check        llama-cpp-python==0.3.34 inside the image;
#                               Dockerfile GGML flags + --no-binary confirmed
#   [6] startup                 uvicorn main:app, no model loaded at boot
#   [7] /health before          default_engine=llamacpp-optimized,
#                               loaded_engines=[]
#   [8] smoke test              TWO POST /generate requests (engine_id
#                               llamacpp-optimized, prompt "Hello! Who are you?")
#                               -> validates response fields, latencies,
#                               generated tokens, tokens/sec, TTFT
#   [9] model in container      size + streaming SHA-256 of the mounted GGUF
#  [10] /health after           loaded_engines=["llamacpp-optimized"]
#  [11] configuration           n_ctx=2048, n_threads=8, n_gpu_layers=0,
#                               greedy decoding, max_new_tokens=64 (unchanged)
#  [12] summary
#
# The 5-repeat benchmark is deliberately NOT run. No FP16/Windows comparisons.
#
# Getting the repo + model to the host (run on the DEV machine first):
#   git clone <your-repo-url> ArmInferX && cd ArmInferX
#   scp -r models/gguf user@<arm64-host>:<path>/ArmInferX/models/   # ~2 GB GGUF
#   # (the GGUF is gitignored, so it is not part of the repo clone)
#
# Usage (on the ARM64 host, from the repo root, Docker + Compose installed):
#   bash scripts/arm64_deploy_smoke.sh                 # full run (build)
#   bash scripts/arm64_deploy_smoke.sh --skip-build    # reuse existing image
#   bash scripts/arm64_deploy_smoke.sh --tag=myimg:1   # custom image tag
#
# Exit codes: 0 = all checks PASS, 1 = a check failed (exact stage reported),
# 2 = usage error.
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration / expected values (from the verified Q4_K_M model + STEP 14A)
# ---------------------------------------------------------------------------
EXPECTED_SHA256="626b4a6678b86442240e33df819e00132d3ba7dddfe1cdc4fbb18e0a9615c62d"
EXPECTED_SIZE=2104932768
EXPECTED_LLAMA_CPP="0.3.34"
EXPECTED_DEFAULT_ENGINE="llamacpp-optimized"
EXPECTED_MODEL_ID="qwen2.5-3b-instruct-q4_k_m"
MODEL_REL="models/gguf/qwen2.5-3b-instruct-q4_k_m.gguf"
SERVICE="backend"

SKIP_BUILD=0
IMAGE_TAG="arminferx-backend:latest"
for arg in "$@"; do
  case "$arg" in
    --skip-build) SKIP_BUILD=1 ;;
    --tag=*) IMAGE_TAG="${arg#--tag=}" ;;
    -h | --help) sed -n '2,34p' "$0"; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

fail() { echo; echo "STEP 14B FAILED at: $1" >&2; exit 1; }
pass() { echo "[PASS] $1"; }
note() { echo "[note] $1"; }

# Container-side python helper (always available inside the image).
EXEC_PY() { docker compose exec -T "$SERVICE" python "$@"; }
HEALTH_URL="http://127.0.0.1:8000/health"

echo "========================================================================"
echo "ArmInferX STEP 14B — ARM64 Docker build, deployment & smoke test"
echo "========================================================================"

# ---------------------------------------------------------------------------
# [1] Environment report
# ---------------------------------------------------------------------------
echo
echo "[1] Environment"
ARCH="$(uname -m)"
echo "    uname -m         = $ARCH"
echo "    docker           = $(docker --version 2>/dev/null || echo MISSING)"
echo "    docker compose   = $(docker compose version 2>/dev/null || echo MISSING)"
if command -v free >/dev/null 2>&1; then
  echo "    RAM              = $(free -h | awk 'NR==2 {print $2 " total, " $7 " available"}')"
elif [ -r /proc/meminfo ]; then
  echo "    RAM              = $(grep -E 'MemTotal|MemAvailable' /proc/meminfo | tr '\n' ' ')"
fi
case "$ARCH" in
  aarch64 | arm64) pass "host architecture is ARM64/aarch64 ($ARCH)" ;;
  *) fail "host architecture is '$ARCH' — STEP 14B requires a real ARM64/aarch64 host (run this on the ARM64 machine)" ;;
esac
docker info >/dev/null 2>&1 || fail "Docker daemon is not reachable"

# ---------------------------------------------------------------------------
# [2] Model pre-flight on the host (source of the read-only mount)
# ---------------------------------------------------------------------------
echo
echo "[2] Model pre-flight (host)"
[ -f "$MODEL_REL" ] || fail "model not found at $MODEL_REL — copy the Q4_K_M GGUF to the host (see script header)"
echo "    file = $MODEL_REL ($(stat -c%s "$MODEL_REL" 2>/dev/null || echo '?') bytes)"
if command -v sha256sum >/dev/null 2>&1; then
  HOST_SHA="$(sha256sum "$MODEL_REL" | awk '{print $1}')"
  echo "    sha256 = $HOST_SHA"
  if [ "$HOST_SHA" != "$EXPECTED_SHA256" ]; then
    fail "host model SHA-256 does not match the verified Q4_K_M hash"
  fi
  pass "host model SHA-256 matches the verified Q4_K_M model"
else
  note "sha256sum not found on host; the container-side SHA-256 check in [9] is authoritative"
fi

# ---------------------------------------------------------------------------
# [3] Build (STEP 14A configuration, llama.cpp-only)
# ---------------------------------------------------------------------------
if [ "$SKIP_BUILD" = "1" ]; then
  echo
  note "--skip-build: using existing image $IMAGE_TAG"
  docker image inspect "$IMAGE_TAG" >/dev/null 2>&1 || fail "image $IMAGE_TAG not found (drop --skip-build)"
else
  echo
  echo "[3] Building the backend image (llama-cpp-python source build; this can take a while)..."
  docker compose build backend
  pass "image built"
fi

# ---------------------------------------------------------------------------
# [4] Image architecture must be linux/arm64
# ---------------------------------------------------------------------------
echo
echo "[4] Image architecture"
IMG_ARCH="$(docker image inspect "$IMAGE_TAG" --format '{{.Os}}/{{.Architecture}}')"
echo "    $IMAGE_TAG = $IMG_ARCH"
if [ "$IMG_ARCH" != "linux/arm64" ]; then
  fail "image is $IMG_ARCH — expected linux/arm64 (STEP 14A flags only; do not change them unless the build genuinely fails)"
fi
pass "image architecture is linux/arm64"

# ---------------------------------------------------------------------------
# [5] Dependency + build-flag verification (static Dockerfile flags + runtime)
# ---------------------------------------------------------------------------
echo
echo "[5] Dependency verification"
DF="docker/backend/Dockerfile"
for flag in "--no-binary llama-cpp-python" "-DGGML_NATIVE=OFF" "-DGGML_OPENMP=ON" "llama-cpp-python==$EXPECTED_LLAMA_CPP"; do
  grep -qF "$flag" "$DF" || fail "Dockerfile no longer contains: $flag"
done
echo "    Dockerfile flags confirmed: --no-binary llama-cpp-python, -DGGML_NATIVE=OFF, -DGGML_OPENMP=ON, llama-cpp-python==$EXPECTED_LLAMA_CPP"

docker compose up -d "$SERVICE" >/dev/null 2>&1 || fail "docker compose up"
LLAMA_VERSION=""
for _ in $(seq 1 15); do
  LLAMA_VERSION="$(EXEC_PY -c "import llama_cpp; print(llama_cpp.__version__)" 2>/dev/null || true)"
  [ -n "$LLAMA_VERSION" ] && break
  sleep 2
  note "waiting for the container to accept exec ..."
done
LLAMA_VERSION="${LLAMA_VERSION:-MISSING}"
echo "    llama-cpp-python in image = $LLAMA_VERSION"
if [ "$LLAMA_VERSION" != "$EXPECTED_LLAMA_CPP" ]; then
  fail "llama-cpp-python in the image is '$LLAMA_VERSION' — expected $EXPECTED_LLAMA_CPP"
fi
pass "llama-cpp-python $EXPECTED_LLAMA_CPP installed"

# ---------------------------------------------------------------------------
# [6] Startup + wait for /health (no model loaded at boot)
# ---------------------------------------------------------------------------
echo
echo "[6] Waiting for the backend to answer /health ..."
HEALTHY=0
for _ in $(seq 1 60); do
  if EXEC_PY -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('$HEALTH_URL', timeout=3).getcode()==200 else 1)" >/dev/null 2>&1; then
    HEALTHY=1
    break
  fi
  sleep 2
done
[ "$HEALTHY" = "1" ] || fail "backend did not become healthy within 120s (check: docker compose logs $SERVICE)"
pass "backend started (uvicorn main:app --host 0.0.0.0 --port 8000)"

# ---------------------------------------------------------------------------
# [7] /health before inference
# ---------------------------------------------------------------------------
echo
echo "[7] /health before inference"
HEALTH_BEFORE="$(EXEC_PY -c "import json,urllib.request; print(json.dumps(json.load(urllib.request.urlopen('$HEALTH_URL', timeout=5))))")"
echo "    $HEALTH_BEFORE"
case "$HEALTH_BEFORE" in
  *'"default_engine": "'"$EXPECTED_DEFAULT_ENGINE"'"'*) pass "default_engine = $EXPECTED_DEFAULT_ENGINE" ;;
  *) fail "default_engine is not $EXPECTED_DEFAULT_ENGINE: $HEALTH_BEFORE" ;;
esac
case "$HEALTH_BEFORE" in
  *'"loaded_engines": []'*) pass "loaded_engines = [] (no model loaded at startup; baseline never auto-loaded)" ;;
  *) fail "loaded_engines is not empty at startup: $HEALTH_BEFORE" ;;
esac

# ---------------------------------------------------------------------------
# [8] Real ARM64 smoke test: TWO /generate requests against llamacpp-optimized
# ---------------------------------------------------------------------------
echo
echo "[8] Smoke test: two POST /generate requests (engine_id=$EXPECTED_DEFAULT_ENGINE)"
EXEC_PY - <<'PY'
import json
import time
import urllib.request

URL = "http://127.0.0.1:8000/generate"
PROMPT = "Hello! Who are you?"
ENGINE_ID = "llamacpp-optimized"
EXPECTED_MODEL = "qwen2.5-3b-instruct-q4_k_m"

def post(prompt):
    body = json.dumps({"prompt": prompt, "engine_id": ENGINE_ID}).encode("utf-8")
    req = urllib.request.Request(
        URL, data=body, headers={"Content-Type": "application/json"}
    )
    started = time.perf_counter()
    with urllib.request.urlopen(req, timeout=900) as resp:
        data = json.load(resp)
    return data, (time.perf_counter() - started) * 1000.0

failures = []
results = []
for label, prompt in (("first", PROMPT), ("second", PROMPT)):
    data, wall_ms = post(prompt)
    status = data.get("status")
    engine_id = data.get("engine_id")
    runtime = data.get("runtime")
    model = data.get("model")
    response = (data.get("response") or "").strip()
    generated = data.get("generated_tokens")
    latency = data.get("latency_ms")
    tps = data.get("tokens_per_second")
    ttft = data.get("ttft_ms")

    checks = {
        "HTTP status == success": status == "success",
        "engine_id == llamacpp-optimized": engine_id == ENGINE_ID,
        "runtime == llama.cpp": runtime == "llama.cpp",
        f"model == {EXPECTED_MODEL}": model == EXPECTED_MODEL,
        "non-empty response": len(response) > 0,
        "generated_tokens > 0": isinstance(generated, int) and generated > 0,
        "latency_ms > 0": isinstance(latency, (int, float)) and latency > 0,
        "tokens_per_second > 0": isinstance(tps, (int, float)) and tps > 0,
    }
    for name, ok in checks.items():
        if not ok:
            failures.append(f"{label} request: {name}")
    results.append(
        {
            "label": label,
            "wall_ms": round(wall_ms, 1),
            "latency_ms": round(latency, 1) if latency is not None else None,
            "ttft_ms": round(ttft, 1) if ttft is not None else None,
            "generated_tokens": generated,
            "tokens_per_second": tps,
            "response": response[:90],
        }
    )

for r in results:
    print(
        f"    {r['label']:6s} | engine_latency_ms={r['latency_ms']} | "
        f"request_wall_ms={r['wall_ms']} | ttft_ms={r['ttft_ms']} | "
        f"tokens={r['generated_tokens']} | tok/s={r['tokens_per_second']}"
    )
    print(f"            response: {r['response']!r}")

if failures:
    raise SystemExit(f"smoke test assertions failed: {failures}")

first_lat = results[0]["latency_ms"]
second_lat = results[1]["latency_ms"]
if (
    first_lat is not None
    and second_lat is not None
    and second_lat < first_lat
):
    print("    model caching: second request was served by the already-loaded engine (lower latency)")
else:
    print("    model caching: second request succeeded; latency not clearly lower (informational only)")

print("SMOKE_OK")
PY
pass "real ARM64 inference succeeded (2/2 requests, all fields valid)"

# ---------------------------------------------------------------------------
# [9] Model inside the container: existence + size + streaming SHA-256
# ---------------------------------------------------------------------------
echo
echo "[9] Model inside the container"
CONTAINER_MODEL="/app/models/gguf/qwen2.5-3b-instruct-q4_k_m.gguf"
MODEL_INFO="$(EXEC_PY - "$CONTAINER_MODEL" <<'PY'
import hashlib
import os
import sys

path = sys.argv[1]
size = os.path.getsize(path)
h = hashlib.sha256()
with open(path, "rb") as f:
    for chunk in iter(lambda: f.read(1024 * 1024), b""):
        h.update(chunk)
print(f"size={size} sha256={h.hexdigest()}")
PY
)"
echo "    $MODEL_INFO"
CONTAINER_SIZE="$(echo "$MODEL_INFO" | sed -n 's/^size=\([0-9]*\).*/\1/p')"
CONTAINER_SHA="$(echo "$MODEL_INFO" | sed -n 's/.*sha256=\([0-9a-f]*\).*/\1/p')"
[ "$CONTAINER_SIZE" = "$EXPECTED_SIZE" ] || fail "container model size mismatch: '$CONTAINER_SIZE' != $EXPECTED_SIZE"
[ "$CONTAINER_SHA" = "$EXPECTED_SHA256" ] || fail "container model SHA-256 mismatch: '$CONTAINER_SHA'"
pass "mounted model verified in container ($CONTAINER_SIZE bytes, sha256 $CONTAINER_SHA)"

# ---------------------------------------------------------------------------
# [10] /health after inference
# ---------------------------------------------------------------------------
echo
echo "[10] /health after inference"
HEALTH_AFTER="$(EXEC_PY -c "import json,urllib.request; print(json.dumps(json.load(urllib.request.urlopen('$HEALTH_URL', timeout=5))))")"
echo "    $HEALTH_AFTER"
case "$HEALTH_AFTER" in
  *'"loaded_engines": ["llamacpp-optimized"]'*) pass "loaded_engines = [\"llamacpp-optimized\"] (model loaded and cached)" ;;
  *) fail "loaded_engines is not [\"llamacpp-optimized\"] after inference: $HEALTH_AFTER" ;;
esac

# ---------------------------------------------------------------------------
# [11] Configuration verification (engine defaults unchanged)
# ---------------------------------------------------------------------------
echo
echo "[11] Configuration verification (engine defaults)"
EXEC_PY - <<'PY'
from engines.llamacpp_optimized import (
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_N_CTX,
    DEFAULT_N_THREADS,
    DEFAULT_TEMPERATURE,
    N_GPU_LAYERS,
)
expected = (2048, 8, 0, 0.0, 64)  # n_ctx, n_threads, n_gpu_layers, temperature, max_new_tokens
actual = (DEFAULT_N_CTX, DEFAULT_N_THREADS, N_GPU_LAYERS, DEFAULT_TEMPERATURE, DEFAULT_MAX_NEW_TOKENS)
if actual != expected:
    raise SystemExit(f"engine defaults drifted: {actual} != {expected}")
print(f"    n_ctx={DEFAULT_N_CTX} n_threads={DEFAULT_N_THREADS} "
      f"n_gpu_layers={N_GPU_LAYERS} temperature={DEFAULT_TEMPERATURE} "
      f"(greedy) max_new_tokens={DEFAULT_MAX_NEW_TOKENS}")
PY
pass "configuration preserved: n_ctx=2048, n_threads=8, n_gpu_layers=0, greedy, max_new_tokens=64"

# ---------------------------------------------------------------------------
# [12] Summary
# ---------------------------------------------------------------------------
echo
echo "========================================================================"
echo "STEP 14B SUMMARY — all checks PASS"
echo "========================================================================"
echo "  host architecture : $ARCH"
echo "  image             : $IMAGE_TAG ($IMG_ARCH)"
echo "  llama-cpp-python  : $LLAMA_VERSION (source-built with STEP 14A flags)"
echo "  model (container) : $CONTAINER_MODEL"
echo "  model sha256      : $CONTAINER_SHA"
echo "  default engine    : $EXPECTED_DEFAULT_ENGINE"
echo "  config            : n_ctx=2048, n_threads=8, n_gpu_layers=0, greedy, max_new_tokens=64"
echo "  benchmark         : NOT run (STEP 14B is deployment validation only)"
echo "  comparisons       : none (no Windows/FP16/Arm64 comparisons produced)"
echo "# Status: PASS"
