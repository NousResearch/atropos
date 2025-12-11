# Cline Node/TypeScript/JavaScript Environment
# Base already has Node.js 22, just add extras
ARG BASE_IMAGE=nousresearch/cline-base:latest
FROM ${BASE_IMAGE}

USER root

# Additional Node tools
RUN npm install -g \
    typescript \
    ts-node \
    tsx \
    pnpm \
    yarn \
    eslint \
    prettier

USER cline
WORKDIR /workspace
