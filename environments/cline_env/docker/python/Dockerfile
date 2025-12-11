# Cline Python Environment
# Extends base with Python development tools
#
# Build: docker build -t nousresearch/cline-python:latest .
# Run:   docker run -it -e ANTHROPIC_API_KEY=... -e CLINE_TASK="..." nousresearch/cline-python:latest

ARG BASE_IMAGE=nousresearch/cline-base:latest
FROM ${BASE_IMAGE}

USER root

# Python development packages
RUN apt-get update && apt-get install -y \
    python3.10-dev \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Install common Python tools
RUN pip3 install --no-cache-dir \
    pip \
    setuptools \
    wheel \
    virtualenv \
    poetry \
    pipenv \
    pytest \
    black \
    ruff \
    mypy \
    ipython

# Verify Python setup
RUN python3 --version && pip3 --version && poetry --version

USER cline
WORKDIR /workspace

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
