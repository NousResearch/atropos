# Cline Haskell Environment
ARG BASE_IMAGE=nousresearch/cline-base:latest
FROM ${BASE_IMAGE}

USER root

# Install GHC (Haskell compiler) and Cabal
RUN apt-get update && apt-get install -y \
    ghc \
    cabal-install \
    && rm -rf /var/lib/apt/lists/*

# Setup for cline user
RUN mkdir -p /home/cline/.cabal /home/cline/.ghc \
    && chown -R cline:cline /home/cline/.cabal /home/cline/.ghc

USER cline
ENV PATH="/home/cline/.cabal/bin:${PATH}"

# Update cabal package list
RUN cabal update

# Verify installation
RUN ghc --version

WORKDIR /workspace
