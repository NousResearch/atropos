# Cline C++ Environment
ARG BASE_IMAGE=nousresearch/cline-base:latest
FROM ${BASE_IMAGE}

USER root

RUN apt-get update && apt-get install -y \
    g++ \
    gcc \
    gdb \
    cmake \
    make \
    ninja-build \
    clang \
    clang-format \
    libboost-all-dev \
    && rm -rf /var/lib/apt/lists/*

USER cline
WORKDIR /workspace
