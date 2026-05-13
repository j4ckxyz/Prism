#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$(dirname "$REPO_DIR")"

set -a
source .env
set +a

source "$REPO_DIR/.venv/bin/activate"
PYTHONPATH="$REPO_DIR" uvicorn app.main:app --app-dir "$REPO_DIR" --reload
