# Cline Elixir Environment
ARG BASE_IMAGE=nousresearch/cline-base:latest
FROM ${BASE_IMAGE}

USER root

# Install Erlang and Elixir
RUN apt-get update && apt-get install -y \
    erlang \
    elixir \
    && rm -rf /var/lib/apt/lists/*

# Setup for cline user
RUN mkdir -p /home/cline/.mix /home/cline/.hex \
    && chown -R cline:cline /home/cline/.mix /home/cline/.hex

USER cline
ENV MIX_HOME=/home/cline/.mix
ENV HEX_HOME=/home/cline/.hex

# Verify installation (hex/rebar will be installed at runtime if needed)
RUN elixir --version || true

WORKDIR /workspace
