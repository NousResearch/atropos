# Cline Lua Environment
ARG BASE_IMAGE=nousresearch/cline-base:latest
FROM ${BASE_IMAGE}

USER root

# Install Lua and LuaRocks
RUN apt-get update && apt-get install -y \
    lua5.4 \
    liblua5.4-dev \
    luarocks \
    && rm -rf /var/lib/apt/lists/*

# Setup for cline user
RUN mkdir -p /home/cline/.luarocks \
    && chown -R cline:cline /home/cline/.luarocks

USER cline
ENV PATH="/home/cline/.luarocks/bin:${PATH}"

# Verify installation
RUN lua -v

WORKDIR /workspace
