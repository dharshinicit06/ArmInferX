"""ArmInferX API entry point (composition root).

Wires configuration, logging, the engine manager (engine selection + lazy
loading), and HTTP error handling together.

Lifecycle: **no model is loaded at startup.** The
:class:`~engines.manager.EngineManager` stored on ``app.state`` loads the
selected engine on first use and reuses it for every subsequent request. This
keeps startup fast and — critically — never automatically loads the FP16
Transformers baseline, which is infeasible on this machine's 7.63 GiB RAM.

Default engine: ``ARMINFERX_DEFAULT_ENGINE`` (default ``llamacpp-optimized``,
the verified Q4_K_M llama.cpp runtime on this machine). Set it to
``transformers-baseline`` on hardware where that baseline is feasible; the
baseline is only ever loaded when explicitly requested.
"""

from contextlib import asynccontextmanager
import logging
import os
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from api.routes.benchmarks.router import router as benchmarks_router
from api.routes.inference.router import router as inference_router
from api.routes.optimization.router import router as optimization_router
from api.utils import configure_logging, register_exception_handlers
from engines.manager import EngineManager

PROJECT_ROOT = Path(__file__).resolve().parent.parent

logger = logging.getLogger("armiferx")


# ---------------------------------------------------------------------------
# Configuration (overridable via environment variables)
# ---------------------------------------------------------------------------
def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _model_dir_from_env() -> Path:
    value = _env("ARMINFERX_MODEL_DIR", "")
    if value:
        path = Path(value)
        return path if path.is_absolute() else PROJECT_ROOT / path
    return PROJECT_ROOT / "models" / "downloaded" / "qwen2.5-3b-instruct"


#: Engine used when POST /generate omits engine_id. Defaults to the verified
#: Q4_K_M llama.cpp engine on this machine; the FP16 baseline is never loaded
#: automatically (it needs more RAM than this machine has free).
DEFAULT_ENGINE_ID = _env("ARMINFERX_DEFAULT_ENGINE", "llamacpp-optimized")

MODEL_DIR = _model_dir_from_env()
DEVICE = _env("ARMINFERX_DEVICE", "cpu")
DTYPE = _env("ARMINFERX_DTYPE", "float16")
MAX_CPU_MEMORY = _env("ARMINFERX_MAX_CPU_MEMORY", "3GiB") or None


def _baseline_engine_kwargs() -> dict:
    """Load kwargs for the transformers baseline (used only when requested)."""
    kwargs: dict = {
        "model_dir": str(MODEL_DIR),
        "device": DEVICE,
        "dtype": DTYPE,
    }
    if MAX_CPU_MEMORY:
        kwargs["max_cpu_memory"] = MAX_CPU_MEMORY
    return kwargs


def _build_engine_manager() -> EngineManager:
    """Engine manager with per-engine load configuration."""
    return EngineManager(
        default_engine_id=DEFAULT_ENGINE_ID,
        engine_kwargs={
            # llama.cpp: engine defaults only (n_ctx=2048, n_threads=8,
            # n_gpu_layers=0, temperature=0.0, max_new_tokens=64) — unchanged.
            "llamacpp-optimized": {},
            # Transformers baseline: model directory / device / dtype / memory
            # cap from environment (mirrors the pre-STEP-13 startup behavior).
            "transformers-baseline": _baseline_engine_kwargs(),
        },
    )


# ---------------------------------------------------------------------------
# Application lifecycle: no model loaded at startup; engines load lazily
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    manager = _build_engine_manager()
    app.state.engine_manager = manager
    logger.info(
        "Engine manager ready (default=%s; nothing loaded at startup). "
        "Engines load lazily on first use.",
        manager.default_engine_id,
    )
    yield
    logger.info("ArmInferX backend shutting down.")


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------
def create_app() -> FastAPI:
    configure_logging()

    app = FastAPI(
        title="ArmInferX Backend",
        description=(
            "AI Inference Optimization Studio for Arm64 Cloud.\n\n"
            "Serves text generation through selectable inference engines "
            "(llama.cpp Q4_K_M by default, Transformers baseline on request) "
            "with per-engine benchmark metadata. Interactive docs for every "
            "endpoint are available here in Swagger UI."
        ),
        version="0.5.0",
        lifespan=lifespan,
        openapi_tags=[
            {
                "name": "inference",
                "description": "Text generation against a selected engine.",
            },
            {
                "name": "benchmarks",
                "description": "Read-only access to saved benchmark results.",
            },
            {
                "name": "optimization",
                "description": "Optimization evidence report (measured facts).",
            },
            {
                "name": "system",
                "description": "Application-level endpoints (health, docs).",
            },
        ],
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:3000",
            "http://localhost:5173",
            "http://localhost:4173",  # vite preview (production build check)
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    register_exception_handlers(app)

    # --- Route registration -------------------------------------------------
    app.include_router(inference_router)
    app.include_router(benchmarks_router)
    app.include_router(optimization_router)

    @app.get("/health", tags=["system"], summary="Liveness and engine status")
    def health_check(request: Request):
        manager = getattr(request.app.state, "engine_manager", None)
        loaded = manager.snapshot() if manager is not None else {}
        default_engine = manager.default_engine_id if manager is not None else None
        default_info = loaded.get(default_engine) if default_engine else None
        return {
            "status": "healthy",
            "service": "ArmInferX Backend",
            "default_engine": default_engine,
            "available_engines": (
                manager.available_ids() if manager is not None else []
            ),
            "loaded_engines": list(loaded),
            "model_id": (default_info or {}).get("model_id"),
            "model_loaded": default_info is not None,
        }

    return app


app = create_app()
