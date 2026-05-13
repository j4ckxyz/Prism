#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

set -a
source .env
set +a

source bluesky-local-feed/.venv/bin/activate
PYTHONPATH=bluesky-local-feed uvicorn app.main:app --app-dir bluesky-local-feed --reload
