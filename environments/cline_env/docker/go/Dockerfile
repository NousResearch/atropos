# Cline Go Environment
ARG BASE_IMAGE=nousresearch/cline-base:latest
FROM ${BASE_IMAGE}

USER root

# Install Go
RUN wget -q https://go.dev/dl/go1.22.0.linux-amd64.tar.gz \
    && tar -C /usr/local -xzf go1.22.0.linux-amd64.tar.gz \
    && rm go1.22.0.linux-amd64.tar.gz

ENV PATH="/usr/local/go/bin:/home/cline/go/bin:${PATH}"
ENV GOPATH="/home/cline/go"

USER cline
RUN mkdir -p /home/cline/go/{bin,src,pkg}

# Install common Go tools
RUN go install golang.org/x/tools/gopls@latest \
    && go install github.com/go-delve/delve/cmd/dlv@latest

WORKDIR /workspace
