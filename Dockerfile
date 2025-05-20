# Phase 1: Base image with Python and system dependencies
FROM python:3.10-slim-bullseye AS builder

# Install system dependencies required for PyBullet and other libraries
# xvfb is for running PyBullet headlessly in environments like Cloud Run
# libgl1-mesa-glx is a common OpenGL dependency
RUN apt-get update && apt-get install -y --no-install-recommends \
    xvfb \
    libgl1-mesa-glx \
    libxrender1 \
    libxext6 \
    # Add any other system dependencies that PyBullet or your other libs might need
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set up the working directory
WORKDIR /app

# Copy the Conda environment.yml file first to leverage Docker layer caching
# If you prefer requirements.txt, adjust accordingly
COPY environment.yml ./

# Create Conda environment from environment.yml
# This approach is more robust for Conda environments than trying to pip install from a Conda list
# However, Conda itself adds significant size to the image.
# Consider a pip-based requirements.txt for smaller images if possible.
# For this example, we'll show a pip-based install from requirements.txt for a slimmer final image.

# Let's switch to a pip-based approach for a slimmer image as Conda is heavy.
# If you have a requirements.txt generated from your Conda env (pip freeze > requirements.txt within Conda env)
# that would be ideal. For now, we'll list key pip packages.

# Phase 1 (Alternative if using pip directly without full Conda env in image):
# Base Python image
FROM python:3.10-slim-bullseye

# Install system dependencies (as above)
RUN apt-get update && apt-get install -y --no-install-recommends \
    xvfb \
    libgl1-mesa-glx \
    libxrender1 \
    libxext6 \
    # For tk (sometimes a PyBullet rendering dependency, though maybe not for headless)
    # tk-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Copy only requirements.txt first to leverage Docker cache
# Create a minimal requirements.txt based on our knowledge:
# You would generate this properly from your environment.
# For now, creating a placeholder requirements.txt content here.
# RUN echo "pybullet" > requirements.txt && \
#     echo "websockets" >> requirements.txt && \
#     echo "numpy" >> requirements.txt && \
#     echo "anthropic" >> requirements.txt && \
#     echo "python-dotenv" >> requirements.txt && \
#     echo "wandb" >> requirements.txt
# Instead, we expect a requirements.txt file to be present in the build context.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
# Assuming your Padres code (spatial_rl_mvp, llm_services, etc.) is in spatial_rl_mvp/
# and other necessary files like .env.example are at the root of the build context.
COPY spatial_rl_mvp/ ./spatial_rl_mvp/
COPY .env.example ./.env.example
# If llm_services.py is outside spatial_rl_mvp, copy it too, e.g.:
# COPY llm_services.py ./

# Expose ports
# Port 8765 for WebSockets (Padres visualization)
EXPOSE 8765
# Port 8080 for a potential future control/health API (as per plan)
EXPOSE 8080

# Command to run the application
# This assumes your spatial_env.py is set up to run the server mode by default
# or can be configured via environment variables if it has multiple modes.
# For Cloud Run, it's often good to run under xvfb-run for headless GUI apps.
# The main script `spatial_env.py` is inside `spatial_rl_mvp` package.
CMD ["xvfb-run", "--auto-servernum", "python", "-m", "spatial_rl_mvp.spatial_env"] 