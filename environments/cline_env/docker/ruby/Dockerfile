# Cline Ruby Environment
ARG BASE_IMAGE=nousresearch/cline-base:latest
FROM ${BASE_IMAGE}

USER root

# Install Ruby and bundler
RUN apt-get update && apt-get install -y \
    ruby-full \
    ruby-bundler \
    && rm -rf /var/lib/apt/lists/*

# Setup for cline user
RUN mkdir -p /home/cline/.gem /home/cline/.bundle \
    && chown -R cline:cline /home/cline/.gem /home/cline/.bundle

USER cline
ENV GEM_HOME=/home/cline/.gem
ENV PATH="/home/cline/.gem/bin:${PATH}"

# Verify installation
RUN ruby --version && gem --version

WORKDIR /workspace
