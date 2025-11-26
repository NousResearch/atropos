#!/usr/bin/env bash
# Wrapper script for Nomad worker that handles Nix profile invocation
set -euo pipefail

echo "[nomad-wrapper] Starting Cline worker for profile: ${CLINE_PROFILE_KEY:-unknown}"

# Parse any extra task environment from TASK_ENV_JSON
if [ -n "${TASK_ENV_JSON:-}" ] && [ "${TASK_ENV_JSON}" != "{}" ]; then
    echo "[nomad-wrapper] Injecting task environment variables from TASK_ENV_JSON"
    eval "$(echo "$TASK_ENV_JSON" | jq -r 'to_entries | .[] | "export \(.key)=\"\(.value)\""')"
fi

BOOTSTRAP_SCRIPT="${ATROPOS_ROOT}/environments/cline_env/cline_dev/bootstrap_cline_worker.sh"

# If PROFILE_DIR is set, use nix develop to run the bootstrap script
if [ -n "${PROFILE_DIR:-}" ] && [ -d "${PROFILE_DIR}" ]; then
    echo "[nomad-wrapper] Using Nix profile at: ${PROFILE_DIR}"
    exec nix develop "${PROFILE_DIR}" --command "${BOOTSTRAP_SCRIPT}"
else
    echo "[nomad-wrapper] No Nix profile specified, running bootstrap directly"
    exec "${BOOTSTRAP_SCRIPT}"
fi
