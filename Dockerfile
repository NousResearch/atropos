# Phase 1: Builder stage (optional, but good for multi-stage builds if you had compilation steps)
FROM python:3.11-slim AS builder

# Set environment variables for Python
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /opt/app

# Install build dependencies if any (e.g., if some pip packages need compilation)
# RUN apt-get update && apt-get install -y --no-install-recommends gcc libpq-dev && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip wheel --no-cache-dir --wheel-dir /opt/wheels -r requirements.txt

# Phase 2: Final stage
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
# PORT is the standard environment variable for Cloud Run services
ENV PORT 8080

# Create a non-root user and group
RUN groupadd --gid 1001 appuser && \
    useradd --uid 1001 --gid 1001 --shell /bin/bash --create-home appuser

WORKDIR /home/appuser/app

# Copy Python dependencies from builder stage (wheels)
COPY --from=builder /opt/wheels /opt/wheels/

# Install Python dependencies from wheels (faster and more secure than from requirements.txt directly in final image)
# Ensure all necessary system libraries for these wheels are present in this stage if they were not part of python:3.11-slim
RUN pip install --no-cache-dir --no-index --find-links=/opt/wheels /opt/wheels/*

# Copy application code as the non-root user
# Only copy necessary files to keep the image lean and secure
COPY --chown=appuser:appuser app.py ./
COPY --chown=appuser:appuser production_research_pipeline.py ./
COPY --chown=appuser:appuser enhanced_padres_perplexity.py ./
COPY --chown=appuser:appuser bigquery_manager.py ./
COPY --chown=appuser:appuser paper_generator.py ./
COPY --chown=appuser:appuser run_single_padres_test.py ./

# Switch to the non-root user
USER appuser

# Expose the port the app runs on
EXPOSE 8080

# Command to run the Gunicorn server
# Gunicorn will look for an ASGI app instance named "app" in the "app" module (app.py)
# Use exec form to make Gunicorn the PID 1 process
CMD exec gunicorn --bind 0.0.0.0:$PORT --workers 2 --worker-class uvicorn.workers.UvicornWorker --timeout 120 app:app 