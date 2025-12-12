# Cline Dart Environment
ARG BASE_IMAGE=nousresearch/cline-base:latest
FROM ${BASE_IMAGE}

USER root

# Install Dart SDK
RUN apt-get update && apt-get install -y apt-transport-https \
    && wget -qO- https://dl-ssl.google.com/linux/linux_signing_key.pub | gpg --dearmor -o /usr/share/keyrings/dart.gpg \
    && echo 'deb [signed-by=/usr/share/keyrings/dart.gpg arch=amd64] https://storage.googleapis.com/download.dartlang.org/linux/debian stable main' | tee /etc/apt/sources.list.d/dart_stable.list \
    && apt-get update \
    && apt-get install -y dart \
    && rm -rf /var/lib/apt/lists/*

# Setup for cline user
RUN mkdir -p /home/cline/.pub-cache \
    && chown -R cline:cline /home/cline/.pub-cache

USER cline
ENV PATH="/usr/lib/dart/bin:/home/cline/.pub-cache/bin:${PATH}"
ENV PUB_CACHE=/home/cline/.pub-cache

# Verify installation
RUN dart --version

WORKDIR /workspace
