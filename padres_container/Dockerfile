# Dockerfile for Padres Environment Service API

# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /padres_app

# Install system dependencies and build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    python3-pip \
    python3-setuptools \
    python3-wheel \
    python3-venv \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install PyBullet system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libopengl0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy the actual spatial_rl_mvp code first
COPY ./spatial_rl_mvp /padres_app/spatial_rl_mvp/

# Copy the new requirements file for the FastAPI app and install dependencies
COPY ./padres_container/app/requirements.txt /padres_app/app/requirements.txt
RUN pip install --no-cache-dir -r app/requirements.txt

# Copy the existing requirements file (if it has other specific Padres deps) and install them
COPY ./padres_container/requirements.txt /padres_app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code (FastAPI app) into the container
COPY ./padres_container/app/ /padres_app/app/

# Make port 8088 available to the world outside this container
EXPOSE 8088

# Set PYTHONPATH to include the app directory
ENV PYTHONPATH=/padres_app

# Run the FastAPI application
CMD ["sh", "-c", "cd /padres_app/app && python -m uvicorn main:app --host 0.0.0.0 --port 8088"]

# TEMPORARY CMD for debugging startup issues:
# CMD ["python", "app/main.py"] 