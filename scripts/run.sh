#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Load simple KEY=VALUE pairs from .env while tolerating Windows CRLF endings.
load_env_file() {
  local env_file="$1"
  [ -f "$env_file" ] || return 0

  while IFS= read -r line || [ -n "$line" ]; do
    line="${line%$'\r'}"
    [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue
    [[ "$line" != *=* ]] && continue

    local key="${line%%=*}"
    local value="${line#*=}"

    key="$(printf '%s' "$key" | xargs)"
    value="$(printf '%s' "$value" | sed 's/[[:space:]]*$//')"

    export "$key=$value"
  done < "$env_file"
}

load_env_file ".env"

# Normalize truthy env flags like true/1/on/yes.
bool_is_true() {
  local value="${1:-false}"
  value="$(printf '%s' "$value" | tr '[:upper:]' '[:lower:]')"
  [[ "$value" == "1" || "$value" == "true" || "$value" == "yes" || "$value" == "on" ]]
}

# Allow local tooling and startup behavior to be controlled from .env.
PYTHON_BIN="${PYTHON_BIN:-python}"
NPM_BIN="${NPM_BIN:-npm}"
PREPARE_CATALOG_ON_RUN_VALUE="${PREPARE_CATALOG_ON_RUN:-false}"
BUILD_INDEXES_ON_RUN_VALUE="${BUILD_INDEXES_ON_RUN:-true}"
FORCE_REBUILD_INDEXES_ON_RUN_VALUE="${FORCE_REBUILD_INDEXES_ON_RUN:-false}"
BACKEND_READY_TIMEOUT_VALUE="${BACKEND_READY_TIMEOUT:-120}"

if bool_is_true "$PREPARE_CATALOG_ON_RUN_VALUE"; then
  PREPARE_ARGS=(-m backend.prepare_kaggle_catalog)
  if [[ -n "${KAGGLE_MAX_PRODUCTS:-}" ]]; then
    PREPARE_ARGS+=(--max-products "$KAGGLE_MAX_PRODUCTS")
  fi
  if [[ -n "${KAGGLE_MAX_PER_ARTICLE:-}" ]]; then
    PREPARE_ARGS+=(--max-per-article "$KAGGLE_MAX_PER_ARTICLE")
  fi
  if [[ -n "${KAGGLE_MIN_ARTICLE_COUNT:-}" ]]; then
    PREPARE_ARGS+=(--min-article-count "$KAGGLE_MIN_ARTICLE_COUNT")
  fi
  if [[ -n "${KAGGLE_SAMPLE_SEED:-}" ]]; then
    PREPARE_ARGS+=(--seed "$KAGGLE_SAMPLE_SEED")
  fi
  "$PYTHON_BIN" "${PREPARE_ARGS[@]}"
fi

if bool_is_true "$BUILD_INDEXES_ON_RUN_VALUE"; then
  BUILD_ARGS=(-m backend.build_indexes)
  if bool_is_true "$FORCE_REBUILD_INDEXES_ON_RUN_VALUE"; then
    BUILD_ARGS+=(--force)
  fi
  "$PYTHON_BIN" "${BUILD_ARGS[@]}"
fi

BACKEND_HOST_VALUE="${BACKEND_HOST:-0.0.0.0}"
BACKEND_PORT_VALUE="${BACKEND_PORT:-8000}"
BACKEND_RELOAD_VALUE="${BACKEND_RELOAD:-false}"
BACKEND_READY_PATH_VALUE="${BACKEND_READY_PATH:-/readyz}"

UVICORN_ARGS=(
  -m
  uvicorn
  backend.main:app
  --host
  "$BACKEND_HOST_VALUE"
  --port
  "$BACKEND_PORT_VALUE"
)

if [[ "${BACKEND_RELOAD_VALUE,,}" == "true" ]]; then
  UVICORN_ARGS+=(
    --reload
    --reload-dir
    "$ROOT_DIR/backend"
  )
fi

"$PYTHON_BIN" "${UVICORN_ARGS[@]}" &
BACKEND_PID=$!

cleanup() {
  trap - EXIT INT TERM
  if kill -0 "$BACKEND_PID" 2>/dev/null; then
    kill "$BACKEND_PID" 2>/dev/null || true
    wait "$BACKEND_PID" 2>/dev/null || true
  fi
}

trap cleanup EXIT INT TERM

# Wait for the backend health endpoint before starting the frontend.
PROBE_HOST_VALUE="$BACKEND_HOST_VALUE"
if [[ "$PROBE_HOST_VALUE" == "0.0.0.0" ]]; then
  PROBE_HOST_VALUE="127.0.0.1"
fi
READY_URL="http://${PROBE_HOST_VALUE}:${BACKEND_PORT_VALUE}${BACKEND_READY_PATH_VALUE}"
READY_DEADLINE=$((SECONDS + BACKEND_READY_TIMEOUT_VALUE))

until curl --silent --fail "$READY_URL" >/dev/null 2>&1; do
  if ! kill -0 "$BACKEND_PID" 2>/dev/null; then
    echo "Backend exited during startup." >&2
    exit 1
  fi
  if (( SECONDS >= READY_DEADLINE )); then
    echo "Backend did not become ready in time: $READY_URL" >&2
    exit 1
  fi
  sleep 1
done

cd frontend
"$NPM_BIN" run dev -- --host 0.0.0.0 --port 3000
