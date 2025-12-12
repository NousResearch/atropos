# Cline Rust Environment
ARG BASE_IMAGE=nousresearch/cline-base:latest
FROM ${BASE_IMAGE}

USER root

# Install system dependencies for Rust compilation
RUN apt-get update && apt-get install -y \
    build-essential \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Clean up any existing Rust installation and set ownership
RUN rm -rf /home/cline/.cargo /home/cline/.rustup /root/.cargo /root/.rustup \
    && mkdir -p /home/cline/.cargo /home/cline/.rustup \
    && chown -R cline:cline /home/cline/.cargo /home/cline/.rustup

# Install Rust for cline user
USER cline
ENV HOME=/home/cline
ENV CARGO_HOME=/home/cline/.cargo
ENV RUSTUP_HOME=/home/cline/.rustup
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
ENV PATH="/home/cline/.cargo/bin:${PATH}"

# Verify Rust installation
RUN rustc --version && cargo --version

WORKDIR /workspace
