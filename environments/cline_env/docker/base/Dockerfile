# Cline Base Image
# Contains Node.js, Python, and the official Cline CLI
#
# Build: docker build -t nousresearch/cline-base:latest -f Dockerfile .
# Run:   docker run -it -e ANTHROPIC_API_KEY=... -e CLINE_TASK="..." nousresearch/cline-base:latest

FROM ubuntu:22.04

LABEL org.opencontainers.image.source="https://github.com/NousResearch/atropos"
LABEL org.opencontainers.image.description="Cline AI agent base image with official CLI"
LABEL maintainer="NousResearch"

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# System packages
RUN apt-get update && apt-get install -y \
    # Build essentials
    build-essential \
    cmake \
    pkg-config \
    # Python
    python3.10 \
    python3.10-venv \
    python3-pip \
    # Git and utilities
    git \
    curl \
    wget \
    ca-certificates \
    gnupg \
    # Shell utilities
    bash \
    jq \
    unzip \
    tar \
    gzip \
    # Misc
    locales \
    && rm -rf /var/lib/apt/lists/*

# Set locale
RUN locale-gen en_US.UTF-8
ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US:en
ENV LC_ALL=en_US.UTF-8

# Install Node.js 22.x (LTS) - required for Cline CLI
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Verify installations
RUN node --version && npm --version && python3 --version && pip3 --version

# Set up Python symlink
RUN ln -sf /usr/bin/python3.10 /usr/bin/python

# Install the official Cline CLI
RUN npm install -g cline

# Verify Cline installation
RUN cline version || echo "Cline installed"

# Create cline user (non-root for security)
RUN useradd -m -s /bin/bash cline
ENV HOME=/home/cline

# Create workspace and cline data directories
RUN mkdir -p /workspace /home/cline/.cline && \
    chown -R cline:cline /workspace /home/cline/.cline

# Create npm cache directory for cline user
RUN mkdir -p /home/cline/.npm && chown -R cline:cline /home/cline/.npm

# Environment variables
ENV CLINE_HOME=/home/cline/.cline
# Default model (can be overridden)
ENV CLINE_MODEL=claude-sonnet-4-5-20250929

# Copy entrypoint script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Switch to non-root user
USER cline
WORKDIR /workspace

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
