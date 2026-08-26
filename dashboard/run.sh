#!/usr/bin/env bash
# Cacheon submissions dashboard launcher (read-only).
set -euo pipefail
cd "$(dirname "$0")"
PY="${CACHEON_DASH_PY:-/root/miniconda3/envs/prod/bin/python}"
export CACHEON_DASH_HOST="${CACHEON_DASH_HOST:-127.0.0.1}"
export CACHEON_DASH_PORT="${CACHEON_DASH_PORT:-8788}"
exec "$PY" app.py
