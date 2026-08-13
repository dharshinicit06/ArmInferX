"""Download the verified Q4_K_M GGUF into the image's model path.

Runs at Docker build time on Hugging Face Spaces (the ~2 GB GGUF is
gitignored and therefore not present in the repo). Streams the file to
``/app/models/gguf/qwen2.5-3b-instruct-q4_k_m.gguf`` and verifies its
SHA-256 against the exact hash the project validated locally, so a
corrupted/partial download fails the build instead of shipping a broken
model. Skips the download when a verified file is already present (layer
cache hits on rebuild).
"""

from __future__ import annotations

import hashlib
import pathlib
import sys
import urllib.request

MODEL_URL = (
    "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/"
    "qwen2.5-3b-instruct-q4_k_m.gguf"
)
#: Exact SHA-256 of the locally validated model (see docs/final-validation.md
#: and scripts/verify_model.py).
EXPECTED_SHA256 = "626b4a6678b86442240e33df819e00132d3ba7dddfe1cdc4fbb18e0a9615c62d"
DEST = pathlib.Path("/app/models/gguf/qwen2.5-3b-instruct-q4_k_m.gguf")


def sha256_of(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    if DEST.exists():
        if sha256_of(DEST) == EXPECTED_SHA256:
            print(f"Model already present and verified: {DEST}", flush=True)
            return 0
        print(f"Existing file failed verification; re-downloading: {DEST}", flush=True)
        DEST.unlink()

    DEST.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading Q4_K_M GGUF (~2 GB) from {MODEL_URL} ...", flush=True)
    tmp = DEST.with_suffix(".gguf.part")
    with urllib.request.urlopen(MODEL_URL) as resp, tmp.open("wb") as out:
        total = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
            downloaded += len(chunk)
            if total:
                print(f"  {downloaded / 1024 / 1024:.0f} / {total / 1024 / 1024:.0f} MB", flush=True)
    tmp.replace(DEST)

    actual = sha256_of(DEST)
    if actual != EXPECTED_SHA256:
        print(
            f"FATAL: SHA-256 mismatch after download.\n"
            f"  expected: {EXPECTED_SHA256}\n"
            f"  actual:   {actual}",
            file=sys.stderr,
        )
        return 1
    print(f"Verified: {DEST} ({DEST.stat().st_size / 1024 / 1024:.0f} MB)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
