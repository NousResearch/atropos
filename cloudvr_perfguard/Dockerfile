FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Ensure the database directory exists and has correct permissions if necessary
RUN mkdir -p /app/database

# Set the PYTHONPATH to include the project root
ENV PYTHONPATH=/app

# Expose the port the API will run on (if applicable, assuming 8000 for FastAPI)
EXPOSE 8000

# Command to run the continuous research pipeline
# The main script and its command might need adjustment based on the actual entry point
# For now, assuming the continuous pipeline is the primary executable
CMD ["python", "cloudvr_perfguard/ai_integration/continuous_research_pipeline.py"]
