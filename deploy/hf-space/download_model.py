"""Download the verified Q4_K_M GGUF into the image's model path.

Runs at Docker build time (Hugging Face Spaces and Railway; the ~470 MB
GGUF is gitignored and therefore not present in the repo). Streams the file
to ``/app/models/gguf/qwen2.5-0.5b-instruct-q4_k_m.gguf`` and verifies its
SHA-256 against the exact hash the project validated locally, so a
corrupted/partial download fails the build instead of shipping a broken
model. Skips the download when a verified file is already present (layer
cache hits on rebuild).

Qwen2.5-0.5B-Instruct Q4_K_M (~491 MB, ~0.5 GB in RAM) keeps the container
within Railway's 1 GB RAM / 2 vCPU limit.
"""

from __future__ import annotations

import hashlib
import pathlib
import sys
import urllib.request

MODEL_URL = (
    "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/"
    "qwen2.5-0.5b-instruct-q4_k_m.gguf"
)
#: Exact SHA-256 of the downloaded/verified file (491,400,032 bytes; matches
#: the Hugging Face LFS oid for qwen2.5-0.5b-instruct-q4_k_m.gguf).
EXPECTED_SHA256 = "74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db"
DEST = pathlib.Path("/app/models/gguf/qwen2.5-0.5b-instruct-q4_k_m.gguf")


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
    print(f"Downloading Q4_K_M GGUF (~470 MB) from {MODEL_URL} ...", flush=True)
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
