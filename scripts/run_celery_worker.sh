#!/bin/bash
# scripts/run_celery_worker.sh

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Navigate to the project root (assuming scripts/ is one level down from root)
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Add project root to PYTHONPATH to ensure modules are found
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

echo "Starting Celery worker..."
echo "Project Root: $PROJECT_ROOT"
echo "PYTHONPATH: $PYTHONPATH"

# Activate virtual environment if it exists and is specified
# For example, if you have a venv in the project root:
# if [ -d "$PROJECT_ROOT/venv" ]; then
#   echo "Activating virtual environment..."
#   source "$PROJECT_ROOT/venv/bin/activate"
# fi

# Run the Celery worker
# -A: Specify the Celery application instance
# worker: The command to start a worker
# -l info: Log level (info, debug, warning, error, critical)
# You might want to add other options like:
# -P gevent (or eventlet) for I/O bound tasks, if you install the library
# -c <number> for concurrency (number of worker processes/threads)
# --pool=solo for debugging if tasks are not running as expected

# Default to process-based pool. For I/O bound tasks (like most of ours making API calls),
# eventlet or gevent might be more efficient if installed and configured.
# For simplicity, we start with the default.

cd "$PROJECT_ROOT" || exit

celery -A backend_services.celery_app worker -l INFO

# To run with eventlet (install eventlet first: pip install eventlet):
# celery -A backend_services.celery_app worker -l INFO -P eventlet -c 10

# To run with a specific queue (if you define queues later):
# celery -A backend_services.celery_app worker -l INFO -Q my_custom_queue

echo "Celery worker stopped." 