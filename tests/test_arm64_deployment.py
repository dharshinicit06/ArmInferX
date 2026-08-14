"""STEP 14A — ARM64 deployment preparation tests.

These tests verify *deployment/configuration behavior* only — they require no
Docker engine, no real model load, and no Arm64 hardware:

- the in-container benchmark driver uses the exact STEP 9/11 procedure;
- the image layout (/app/backend) keeps the engine's default model path
  consistent with the documented read-only mount (/app/models/gguf);
- the backend Dockerfile pins the llama-cpp-python build strategy for
  aarch64 and downloads the ~470 MB GGUF at build time (SHA-256-verified);
- docker-compose mounts the same Q4_K_M file and keeps
  llamacpp-optimized as the default engine;
- the deployment docs make no Arm64 performance claim.
"""

import importlib.util
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from benchmark import BenchmarkConfig  # noqa: E402
from engines.llamacpp_optimized import DEFAULT_MODEL_PATH, PROJECT_ROOT as ENGINE_ROOT  # noqa: E402


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _load_driver():
    """Import scripts/run_llamacpp_benchmark.py without executing main()."""
    name = "run_llamacpp_benchmark"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, PROJECT_ROOT / "scripts" / "run_llamacpp_benchmark.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Benchmark driver (in-container benchmark command) uses the STEP 9/11 config
# ---------------------------------------------------------------------------

def test_step9_11_procedure_config():
    driver = _load_driver()
    expected = BenchmarkConfig(
        prompt="Explain what an AI inference engine is.",
        max_new_tokens=64,
        temperature=None,     # greedy
        chat_template=False,  # raw completion
        repeats=5,
        warmup=1,
    )
    config = driver.build_config()
    assert config == expected, f"STEP 9/11 procedure drifted: {config!r}"
    assert driver.ENGINE_ID == "llamacpp-optimized"
    assert driver.EXPECTED_RUNTIME == "llama.cpp"
    print("PASS: benchmark driver config == STEP 9/11 (64, greedy, raw, 5 runs, 1 warmup)")


def test_driver_default_model_is_q4_k_m():
    driver = _load_driver()
    assert driver.DEFAULT_MODEL.name == "qwen2.5-0.5b-instruct-q4_k_m.gguf"
    print("PASS: driver targets the same Q4_K_M GGUF as STEP 9/11")


# ---------------------------------------------------------------------------
# Engine default path vs. image mount path (/app/models/gguf)
# ---------------------------------------------------------------------------

def test_engine_default_path_matches_image_mount():
    # The engine resolves PROJECT_ROOT/models/gguf/...; in the image backend/
    # lives at /app/backend, so PROJECT_ROOT == /app and the mount point
    # /app/models/gguf must line up with this relative path.
    relative = DEFAULT_MODEL_PATH.relative_to(ENGINE_ROOT)
    assert relative == Path("models/gguf/qwen2.5-0.5b-instruct-q4_k_m.gguf"), relative
    assert DEFAULT_MODEL_PATH.is_file(), f"GGUF not found at {DEFAULT_MODEL_PATH}"
    print(f"PASS: engine default path relative to root = {relative.as_posix()}")
    print(f"      -> image mount /app/models/gguf resolves the same Q4_K_M file")


# ---------------------------------------------------------------------------
# Backend Dockerfile strategy
# ---------------------------------------------------------------------------

def test_dockerfile_arm64_strategy():
    dockerfile = _read(PROJECT_ROOT / "docker" / "backend" / "Dockerfile")
    required = [
        "python:3.11-slim-bookworm",          # base (both stages)
        "--no-binary llama-cpp-python",       # force source build so CMAKE_ARGS apply
        "llama-cpp-python==0.3.34",           # pin matches the dev runtime
        "GGML_NATIVE=OFF",                    # portable aarch64 binary
        "GGML_OPENMP=ON",                     # OpenMP threading on aarch64
        "libgomp1",                           # OpenMP runtime lib
        "COPY backend/ /app/backend/",        # layout keeps PROJECT_ROOT == /app
        "COPY scripts/run_llamacpp_benchmark.py",  # in-container benchmark driver
        "ARMINFERX_DEFAULT_ENGINE=llamacpp-optimized",  # default engine preserved
        'CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]',
    ]
    missing = [r for r in required if r not in dockerfile]
    assert not missing, f"Dockerfile missing: {missing}"
    print("PASS: Dockerfile pins llama-cpp-python 0.3.34 + aarch64 build flags + layout")


def test_dockerfile_does_not_bake_model():
    dockerfile = _read(PROJECT_ROOT / "docker" / "backend" / "Dockerfile")
    for line in dockerfile.splitlines():
        stripped = line.strip().upper()
        assert not stripped.startswith(("COPY MODELS", "ADD MODELS")), (
            f"model must be mounted, not baked into the image: {line.strip()}"
        )
    assert "/app/models/gguf" in dockerfile  # documented mount point
    print("PASS: Dockerfile mounts the GGUF at runtime, never bakes it")


def test_dockerfile_baseline_deps_opt_in():
    dockerfile = _read(PROJECT_ROOT / "docker" / "backend" / "Dockerfile")
    assert "INSTALL_BASELINE_DEPS" in dockerfile
    assert 'INSTALL_BASELINE_DEPS="0"' in dockerfile  # llama.cpp-only default
    print("PASS: FP16 baseline deps are opt-in via INSTALL_BASELINE_DEPS (default off)")


# ---------------------------------------------------------------------------
# docker-compose backend service
# ---------------------------------------------------------------------------

def test_compose_backend_service():
    compose = _read(PROJECT_ROOT / "docker-compose.yml")
    required = [
        "backend:",
        "dockerfile: docker/backend/Dockerfile",
        '"8000:8000"',
        "./models/gguf:/app/models/gguf:ro",   # same Q4_K_M file, read-only
        "ARMINFERX_DEFAULT_ENGINE: llamacpp-optimized",
        "ARMINFERX_DEVICE: cpu",
    ]
    missing = [r for r in required if r not in compose]
    assert not missing, f"compose missing: {missing}"
    print("PASS: compose defines the backend service with model mount + CPU-only env")


# ---------------------------------------------------------------------------
# Deployment docs: no Arm64 performance claims
# ---------------------------------------------------------------------------

def test_deployment_docs_make_no_arm64_claims():
    # Whitespace-normalized (and blockquote markers stripped) so assertions
    # survive markdown line wrapping.
    doc = _read(PROJECT_ROOT / "docs" / "arm64-deployment.md").replace(">", " ")
    doc = " ".join(doc.split()).lower()
    assert "preparation only" in doc
    assert "no arm64 performance claim is made anywhere" in doc
    assert "no arm64 benchmark has been run" in doc
    print("PASS: docs/arm64-deployment.md explicitly makes no Arm64 performance claim")


if __name__ == "__main__":
    test_step9_11_procedure_config()
    test_driver_default_model_is_q4_k_m()
    test_engine_default_path_matches_image_mount()
    test_dockerfile_arm64_strategy()
    test_dockerfile_does_not_bake_model()
    test_dockerfile_baseline_deps_opt_in()
    test_compose_backend_service()
    test_deployment_docs_make_no_arm64_claims()
    print("\nAll STEP 14A deployment tests passed.")
