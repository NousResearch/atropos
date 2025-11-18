#!/usr/bin/env bash
set -euo pipefail

LOG_FILE=${LOG_FILE:-/tmp/nomad-dev.log}

if pgrep -f "nomad agent" >/dev/null; then
  echo "Nomad agent is already running." >&2
  exit 0
fi

echo "Starting Nomad dev agent (logs: $LOG_FILE)"
nohup nomad agent -dev -bind=127.0.0.1 -log-level=INFO > "$LOG_FILE" 2>&1 &
echo $! > /tmp/nomad-dev.pid
echo "Nomad agent started with PID $(cat /tmp/nomad-dev.pid)"
