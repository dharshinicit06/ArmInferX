"""Smoke test: POST /generate contract, error handling, and auto-saved results.

Uses fake services so no model or GPU is needed. Benchmark records are
redirected to a temp directory so the real results/ dir stays clean.
"""

import importlib
import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from fastapi.testclient import TestClient  # noqa: E402

import main  # noqa: E402
from api.utils.exceptions import InvalidPromptError  # noqa: E402
from benchmark.storage import BaselineResultStore  # noqa: E402

# NOTE: `import api.routes.inference.router as x` binds the APIRouter object,
# because the package __init__ re-exports `router`. importlib gives us the
# real module so monkeypatching result_store in the tests below works.
router_module = importlib.import_module("api.routes.inference.router")

RECORD_FIELDS = {
    "prompt",
    "model",
    "response",
    "latency_ms",
    "ttft_ms",
    "memory_mb",
    "cpu_percent",
    "generated_tokens",
    "tokens_per_second",
    "timestamp",
}


class FakeResult:
    generated_text = "hello world"
    model_id = "fake-model"
    prompt_tokens = 1
    generated_tokens = 2
    latency_ms = 12.34
    ttft_ms = 5.0


class FakeService:
    model_id = "fake-model"

    def generate(self, prompt):
        if not prompt.strip():
            raise InvalidPromptError("prompt must be a non-empty string")
        time.sleep(0.01)  # simulate a tiny bit of inference time
        return FakeResult()


def _client(service):
    app = main.app
    app.state.inference = service
    return TestClient(app)


def test_generate_success_contract_and_saved_record():
    with tempfile.TemporaryDirectory() as tmp:
        router_module.result_store = BaselineResultStore(tmp)
        client = _client(FakeService())

        resp = client.post("/generate", json={"prompt": "Explain Artificial Intelligence"})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body == {
            "status": "success",
            "model": "fake-model",
            "response": "hello world",
            "latency_ms": 12.34,
        }, body

        # Every inference request auto-saves a unique JSON benchmark record.
        files = sorted(Path(tmp).glob("baseline-*.json"))
        assert len(files) == 1, files
        record = json.loads(files[0].read_text(encoding="utf-8"))
        assert set(record) == RECORD_FIELDS, record
        assert record["prompt"] == "Explain Artificial Intelligence"
        assert record["model"] == "fake-model"
        assert record["response"] == "hello world"
        assert record["latency_ms"] == 12.34
        # TTFT is a separate metric, always <= total latency in a real run.
        assert record["ttft_ms"] == 5.0
        assert record["memory_mb"] > 0
        assert record["cpu_percent"] >= 0
        # Token metrics: the fake reports 2 generated tokens; tokens/sec
        # reuses the recorded latency (12.34 ms) so the record stays
        # internally consistent.
        assert record["generated_tokens"] == 2
        assert record["tokens_per_second"] == round(2 / (12.34 / 1000), 2), record
        assert record["timestamp"]
        print("PASS: success contract ->", body)
        print("PASS: saved record ->", files[0].name, record)


def test_generate_saves_unique_filenames():
    with tempfile.TemporaryDirectory() as tmp:
        router_module.result_store = BaselineResultStore(tmp)
        client = _client(FakeService())
        client.post("/generate", json={"prompt": "first"})
        client.post("/generate", json={"prompt": "second"})
        files = sorted(Path(tmp).glob("baseline-*.json"))
        assert len(files) == 2, files
        assert len({f.name for f in files}) == 2, "filenames must be unique"
        print("PASS: unique filenames ->", [f.name for f in files])


def test_generate_rejects_empty_or_whitespace_prompts():
    client = _client(FakeService())

    # Missing field and empty string are rejected by Pydantic (422).
    resp = client.post("/generate", json={})
    assert resp.status_code == 422, resp.text
    resp = client.post("/generate", json={"prompt": ""})
    assert resp.status_code == 422, resp.text

    # Whitespace-only is rejected by the domain service (400).
    resp = client.post("/generate", json={"prompt": "   "})
    assert resp.status_code == 400, resp.text
    assert resp.json()["detail"] == "prompt must be a non-empty string"
    print("PASS: empty (422) and whitespace-only (400) prompts rejected")


def test_generate_service_unavailable():
    client = _client(None)
    resp = client.post("/generate", json={"prompt": "hello"})
    assert resp.status_code == 503, resp.text
    assert "model not loaded" in resp.json()["detail"].lower(), resp.text
    print("PASS: 503 when service is missing ->", resp.json())


if __name__ == "__main__":
    test_generate_success_contract_and_saved_record()
    test_generate_saves_unique_filenames()
    test_generate_rejects_empty_or_whitespace_prompts()
    test_generate_service_unavailable()
