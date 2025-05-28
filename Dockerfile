
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY cloudvr_perfguard/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY cloudvr_perfguard/ ./cloudvr_perfguard/
COPY production_deployment.py .
COPY cicd_integration.py .

# Create necessary directories
RUN mkdir -p production/config production/logs production/monitoring

# Set environment variables
ENV PYTHONPATH=/app
ENV ENVIRONMENT=production

# Expose port for API
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Default command
CMD ["python", "production_deployment.py"]
