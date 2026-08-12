"""Download the ArmInferX baseline model into models/downloaded/.

Usage (from the repo root):
    backend/.venv/Scripts/python scripts/download_model.py
    backend/.venv/Scripts/python scripts/download_model.py --repo-id Qwen/Qwen2.5-1.5B-Instruct

Downloads Qwen2.5-3B-Instruct (Apache 2.0) into models/downloaded/qwen2.5-3b-instruct/
by default. The download resumes automatically if interrupted. Pass --repo-id /
--out-dir to stage other models for benchmarking.
"""

import argparse

from huggingface_hub import snapshot_download

DEFAULT_REPO_ID = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_OUT_DIR = "models/downloaded/qwen2.5-3b-instruct"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help="Hugging Face repo to download (default: %(default)s)",
    )
    parser.add_argument(
        "--out-dir",
        default=DEFAULT_OUT_DIR,
        help="Destination directory, relative to the repo root (default: %(default)s)",
    )
    args = parser.parse_args()

    print(f"Downloading {args.repo_id} -> {args.out_dir} ...")
    path = snapshot_download(
        repo_id=args.repo_id,
        local_dir=args.out_dir,
    )
    print(f"Done. Model saved to: {path}")
    verify_cmd = "backend/.venv/Scripts/python scripts/verify_model.py"
    if args.out_dir != DEFAULT_OUT_DIR:
        verify_cmd += f" --model-dir {args.out_dir}"
    print(f"Verify it loads:  {verify_cmd}")


if __name__ == "__main__":
    main()
