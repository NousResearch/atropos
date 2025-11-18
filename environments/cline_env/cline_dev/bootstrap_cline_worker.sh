#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CLINE_SRC_DIR=${CLINE_SRC_DIR:-/tmp/nous-cline}
CLINE_REPO_URL=${CLINE_REPO_URL:-https://github.com/NousResearch/cline}
WORKSPACE_ROOT=${WORKSPACE_ROOT:-/workspace}
TASK_BOOTSTRAP_SCRIPT=${TASK_BOOTSTRAP_SCRIPT:-}
PROTOBUS_PORT=${PROTOBUS_PORT:-26040}
HOSTBRIDGE_PORT=${HOSTBRIDGE_PORT:-26041}
USE_C8=${USE_C8:-false}
NODE_OPTIONS=${NODE_OPTIONS:---max-old-space-size=4096}

log() {
  printf '[cline-worker] %s\n' "$*"
}

fetch_cline_repo() {
  if [[ -d "$CLINE_SRC_DIR/.git" ]]; then
    log "Updating existing Cline repo at $CLINE_SRC_DIR"
    git -C "$CLINE_SRC_DIR" fetch origin main
    git -C "$CLINE_SRC_DIR" checkout main
    git -C "$CLINE_SRC_DIR" reset --hard origin/main
  else
    log "Cloning $CLINE_REPO_URL into $CLINE_SRC_DIR"
    rm -rf "$CLINE_SRC_DIR"
    git clone "$CLINE_REPO_URL" "$CLINE_SRC_DIR"
  fi
}

build_cline() {
  pushd "$CLINE_SRC_DIR" >/dev/null
  if [[ ! -d node_modules ]]; then
    log "Installing npm dependencies"
    npm install
  fi
  log "Running proto generation"
  npm run protos
  log "Running lint"
  npm run lint
  log "Building standalone bundle"
  node esbuild.mjs --standalone
  log "Packaging standalone artifacts"
  node scripts/package-standalone.mjs
  popd >/dev/null
}

bootstrap_task_workspace() {
  if [[ -n "$TASK_BOOTSTRAP_SCRIPT" ]]; then
    if [[ ! -x "$TASK_BOOTSTRAP_SCRIPT" ]]; then
      log "Bootstrap script $TASK_BOOTSTRAP_SCRIPT is not executable"
      exit 1
    fi
    log "Running task bootstrap script $TASK_BOOTSTRAP_SCRIPT"
    WORKSPACE_ROOT="$WORKSPACE_ROOT" "$TASK_BOOTSTRAP_SCRIPT" "$WORKSPACE_ROOT"
  else
    log "No TASK_BOOTSTRAP_SCRIPT provided; assuming workspace already prepared at $WORKSPACE_ROOT"
  fi
}

start_cline_core() {
  export NODE_OPTIONS
  export PROTOBUS_PORT HOSTBRIDGE_PORT
  export WORKSPACE_DIR=${WORKSPACE_DIR:-$WORKSPACE_ROOT}
  export DEV_WORKSPACE_FOLDER=${DEV_WORKSPACE_FOLDER:-$WORKSPACE_ROOT}
  export CLINE_DISABLE_BANNERS=true
  export CLINE_DISABLE_REMOTE_CONFIG=true
  export E2E_TEST=true
  export CLINE_ENVIRONMENT=local

  log "Launching Cline core in $WORKSPACE_DIR"
  pushd "$CLINE_SRC_DIR" >/dev/null
  npx tsx scripts/test-standalone-core-api-server.ts
  popd >/dev/null
}

main() {
  fetch_cline_repo
  build_cline
  bootstrap_task_workspace
  start_cline_core
}

main "$@"
