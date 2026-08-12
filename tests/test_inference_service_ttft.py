"""Integration tests: TTFT through the real ``InferenceService.generate()``.

Uses a fake model + tokenizer (no ML weights or GPU) whose ``generate()``
mimics ``model.generate(streamer=...)``: it sleeps to simulate prefill + decode
steps and pushes tokens to the streamer as they are produced. This verifies the
real service wiring — the streamer is actually attached, ``ttft_ms`` is
populated, the generated text is unchanged, and beam search is rejected with a
clear error instead of failing inside ``generate()``.
"""

import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

import torch  # noqa: E402

from api.routes.inference.inference_service import InferenceService  # noqa: E402
from api.utils.exceptions import InvalidGenerationParamsError  # noqa: E402


class FakeModel:
    config = SimpleNamespace(max_position_embeddings=4096)

    def generate(self, input_ids, max_new_tokens=None, do_sample=None, streamer=None, **kwargs):
        assert streamer is not None, "service must attach the TTFT streamer"
        time.sleep(0.02)  # prefill + first decode step
        streamer.put(torch.tensor([[5]]))  # first token becomes available
        time.sleep(0.02)  # remaining decode steps
        streamer.put(torch.tensor([[6]]))
        # output = prompt + 2 generated token ids (5, 6)
        return torch.cat([input_ids, torch.tensor([[5, 6]])], dim=1)


class FakeTokenizer:
    # Absent chat template -> tests must pass use_chat_template=False.
    chat_template = None

    def __call__(self, text, return_tensors="pt"):
        ids = [ord(ch) % 50 + 1 for ch in text]
        return {"input_ids": torch.tensor([ids])}

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(int(i) % 26 + 97) for i in ids)


def _make_service() -> InferenceService:
    model = SimpleNamespace(
        model=FakeModel(),
        tokenizer=FakeTokenizer(),
        model_id="fake-model",
    )
    return InferenceService(model)


def test_generate_populates_ttft_and_keeps_output():
    service = _make_service()
    result = service.generate("hi", use_chat_template=False)

    # Generated text is unchanged by the attached streamer (fake decodes 5,6).
    assert result.generated_text == "fg"
    assert result.generated_tokens == 2
    # TTFT is populated from the streamer and stays separate from total latency.
    assert result.ttft_ms is not None and 10 <= result.ttft_ms <= 300, result.ttft_ms
    assert result.latency_ms >= result.ttft_ms, "total latency >= ttft"
    print(
        f"PASS: generate() -> text={result.generated_text!r} "
        f"ttft_ms={result.ttft_ms:.1f} latency_ms={result.latency_ms:.1f}"
    )


def test_generate_rejects_beam_search_kwargs():
    service = _make_service()
    try:
        service.generate("hi", use_chat_template=False, num_beams=4)
    except InvalidGenerationParamsError as exc:
        assert "beam" in str(exc).lower() or "streaming" in str(exc).lower(), exc
        print("PASS: beam search rejected ->", exc)
        return
    raise AssertionError("expected InvalidGenerationParamsError for num_beams=4")


def test_generate_rejects_caller_provided_streamer():
    service = _make_service()
    try:
        service.generate("hi", use_chat_template=False, streamer=object())
    except InvalidGenerationParamsError:
        print("PASS: caller-provided streamer rejected")
        return
    raise AssertionError("expected InvalidGenerationParamsError for streamer kwarg")


if __name__ == "__main__":
    test_generate_populates_ttft_and_keeps_output()
    test_generate_rejects_beam_search_kwargs()
    test_generate_rejects_caller_provided_streamer()
