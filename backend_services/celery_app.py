# backend_services/celery_app.py

from celery import Celery
import os
from typing import List, Optional

# TODO: Make Redis URL configurable via environment variables for production
# This will be used by the create_celery_app function
DEFAULT_REDIS_URL = "redis://localhost:6379/0"
CELERY_BROKER_URL_ENV = os.environ.get("CELERY_BROKER_URL", DEFAULT_REDIS_URL)
CELERY_BACKEND_URL_ENV = os.environ.get("CELERY_RESULT_BACKEND_URL", DEFAULT_REDIS_URL) # Can be same or different

# Removed the old celery_pipeline_app definition here.

"""
Celery app setup and configuration for the AI Research Pipeline.
"""

# from celery import Celery # Already imported

# Create Celery app with Redis backend/broker
def create_celery_app(mock_mode: bool = False, include_tasks_from: Optional[List[str]] = None) -> Celery:
    """
    Creates and configures the Celery app.
    
    Args:
        mock_mode (bool): If True, use eager mode for Celery (no broker/backend needed).
        include_tasks_from (Optional[List[str]]): List of modules to import when the worker starts.
    
    Returns:
        Celery: Configured Celery app
    """
    app_name = 'ai_research_pipeline'
    tasks_to_include = include_tasks_from if include_tasks_from is not None else []

    if mock_mode:
        print("Creating Celery app in MOCK mode (eager execution).")
        # In mock mode, use eager execution (no Redis needed)
        app = Celery(app_name, include=tasks_to_include)
        app.conf.update(
            task_always_eager=True,  # Tasks execute locally instead of being sent to workers
            task_eager_propagates=True,  # Propagate exceptions in eager mode
            task_serializer='json',
            accept_content=['json'],
            result_serializer='json',
            timezone='UTC',
            enable_utc=True,
            task_track_started=True # Good for knowing task state
        )
    else:
        print(f"Creating Celery app in LIVE mode. Broker: {CELERY_BROKER_URL_ENV}, Backend: {CELERY_BACKEND_URL_ENV}")
        # In normal mode, use Redis (or configured broker/backend)
        app = Celery(
            app_name,
            broker=CELERY_BROKER_URL_ENV,
            backend=CELERY_BACKEND_URL_ENV,
            include=tasks_to_include
        )
        app.conf.update(
            task_serializer="json",
            accept_content=["json"],  # Ignore other content
            result_serializer="json",
            timezone="UTC",
            enable_utc=True,
            task_track_started=True, # To record when tasks start
            # Optional: More advanced settings for production
            # broker_connection_retry_on_startup=True, # Good for production readiness
            # worker_prefetch_multiplier=1, # Can be useful for long-running tasks
            # task_acks_late=True, # If tasks are idempotent and can be retried on worker failure
        )
    
    return app

# Determine mock_mode from an environment variable or a global config if available.
# For simplicity, defaulting to False (live mode) unless overridden.
# In a real app, this might come from your main application's config.
PIPELINE_MOCK_MODE = os.environ.get("PIPELINE_MOCK_MODE", "False").lower() == "true"

# Default app instance
# The tasks in backend_services.celery_tasks will register themselves with this instance.
celery_app = create_celery_app(
    mock_mode=PIPELINE_MOCK_MODE, 
    include_tasks_from=["backend_services.celery_tasks"]
)

# The following comment is now handled by the `include_tasks_from` argument:
# # Import tasks to ensure they're registered with the Celery app
# # This has to go at the bottom to avoid circular imports
# # These will be imported elsewhere when needed 

if __name__ == "__main__":
    # This allows running the celery worker directly using:
    # python -m backend_services.celery_app worker -l info -P eventlet (if using eventlet)
    # Ensure that your project root is in PYTHONPATH.
    # Typically, you run celery worker from the command line: 
    # celery -A backend_services.celery_app worker -l info
    print(f"Starting Celery worker for app '{celery_app.main}'...")
    celery_app.start() # This is a blocking call to start a worker directly 