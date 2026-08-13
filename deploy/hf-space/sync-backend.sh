#!/usr/bin/env bash
# =============================================================================
# Populate deploy/hf-space/backend from the project's backend/ so the Space's
# Docker build context (this folder) contains everything it needs.
#
# Usage (from deploy/hf-space/):
#   bash sync-backend.sh
#
# Afterwards, in the cloned Space repo:
#   git add -A && git commit -m "Deploy ArmInferX backend" && git push
#
# Uses `tar` (present in Git Bash and every Linux) so dev-only artifacts
# (.venv, __pycache__) are excluded during the copy — never copied then
# deleted. .venv is multi-GB with read-only files on Windows; copying it is
# both slow and hard to clean up.
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

SOURCE="$(cd ../../backend && pwd)"
DEST="$(pwd)/backend"

rm -rf "$DEST"
mkdir -p "$DEST"

tar -C "$SOURCE" \
    --exclude='.venv' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.pytest_cache' \
    -cf - . | tar -C "$DEST" -xf -

echo "Synced backend/ from $SOURCE"
echo "Next: commit and push the Space repo (or run from a clone of it)."
