#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_ROOT=${1:-/tmp/ratatui-workspace}
REPO_URL=${REPO_URL:-https://github.com/ratatui/ratatui}
REPO_NAME=ratatui
REPO_PATH="$WORKSPACE_ROOT/$REPO_NAME"
TASK_PROMPT="Add support for vertical gauges to Ratatui's gauge widget (see issue #2148)."

printf "[bootstrap] workspace: %s\n" "$WORKSPACE_ROOT"
mkdir -p "$WORKSPACE_ROOT"

if [[ -d "$REPO_PATH/.git" ]]; then
  echo "[bootstrap] repo exists, fetching latest main"
  git -C "$REPO_PATH" fetch origin main
  git -C "$REPO_PATH" checkout main
  git -C "$REPO_PATH" reset --hard origin/main
else
  echo "[bootstrap] cloning $REPO_URL"
  git clone "$REPO_URL" "$REPO_PATH"
fi

pushd "$REPO_PATH" >/dev/null

if command -v cargo >/dev/null 2>&1; then
  echo "[bootstrap] running cargo fetch"
  cargo fetch
  echo "[bootstrap] building workspace (tests skipped)"
  cargo build --workspace --all-targets
else
  echo "[bootstrap] cargo not found; skipping build"
fi

popd >/dev/null

cat >"$WORKSPACE_ROOT/cline-workspace.json" <<META
{
  "repo_name": "$REPO_NAME",
  "repo_url": "$REPO_URL",
  "workspace_dir": "$REPO_PATH",
  "task_prompt": "$TASK_PROMPT",
  "notes": "Bootstrap prepared via environments/cline_env/cline_dev/examples/ratatui_vertical_gauge"
}
META

echo "[bootstrap] metadata written to $WORKSPACE_ROOT/cline-workspace.json"
echo "[bootstrap] ready to launch Cline core with WORKSPACE_DIR=$REPO_PATH"
