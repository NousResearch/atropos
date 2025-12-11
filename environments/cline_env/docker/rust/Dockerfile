# Cline Rust Environment
ARG BASE_IMAGE=nousresearch/cline-base:latest
FROM ${BASE_IMAGE}

USER root

# Install Rust via rustup
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
ENV PATH="/root/.cargo/bin:${PATH}"

# Install Rust for cline user
USER cline
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
ENV PATH="/home/cline/.cargo/bin:${PATH}"

# Install common Rust tools
RUN cargo install cargo-watch cargo-edit

WORKDIR /workspace
