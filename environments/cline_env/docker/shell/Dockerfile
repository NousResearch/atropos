# Cline Shell/Bash Environment
ARG BASE_IMAGE=nousresearch/cline-base:latest
FROM ${BASE_IMAGE}

USER root

RUN apt-get update && apt-get install -y \
    shellcheck \
    bats \
    shfmt \
    && rm -rf /var/lib/apt/lists/*

USER cline
WORKDIR /workspace
