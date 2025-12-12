# Cline Go Environment
ARG BASE_IMAGE=nousresearch/cline-base:latest
FROM ${BASE_IMAGE}

USER root

# Install Go 1.24 (latest stable)
RUN wget -q https://go.dev/dl/go1.24.4.linux-amd64.tar.gz \
    && tar -C /usr/local -xzf go1.24.4.linux-amd64.tar.gz \
    && rm go1.24.4.linux-amd64.tar.gz

ENV PATH="/usr/local/go/bin:/home/cline/go/bin:${PATH}"
ENV GOPATH="/home/cline/go"
ENV GOCACHE="/home/cline/.cache/go-build"

# Create directories with proper permissions for cline user
RUN mkdir -p /home/cline/go/{bin,src,pkg} /home/cline/.cache/go-build \
    && chown -R cline:cline /home/cline/go /home/cline/.cache

USER cline

# Verify Go installation
RUN go version

WORKDIR /workspace
