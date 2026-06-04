#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
lsof -ti:8765 | xargs kill -9 2>/dev/null || true
uv run uvicorn serve:app --port 8765 --reload
