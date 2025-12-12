# Cline Swift Environment
ARG BASE_IMAGE=nousresearch/cline-base:latest
FROM ${BASE_IMAGE}

USER root

# Install Swift dependencies
RUN apt-get update && apt-get install -y \
    binutils \
    libc6-dev \
    libcurl4-openssl-dev \
    libedit2 \
    libgcc-9-dev \
    libpython3.8 \
    libsqlite3-0 \
    libstdc++-9-dev \
    libxml2-dev \
    libz3-dev \
    pkg-config \
    tzdata \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Download and install Swift (for amd64)
RUN SWIFT_VERSION=5.9.2 \
    && wget -q https://download.swift.org/swift-${SWIFT_VERSION}-release/ubuntu2204/swift-${SWIFT_VERSION}-RELEASE/swift-${SWIFT_VERSION}-RELEASE-ubuntu22.04.tar.gz \
    && tar xzf swift-${SWIFT_VERSION}-RELEASE-ubuntu22.04.tar.gz \
    && mv swift-${SWIFT_VERSION}-RELEASE-ubuntu22.04 /usr/share/swift \
    && rm swift-${SWIFT_VERSION}-RELEASE-ubuntu22.04.tar.gz

# Setup for cline user
RUN mkdir -p /home/cline/.swiftpm \
    && chown -R cline:cline /home/cline/.swiftpm

USER cline
ENV PATH="/usr/share/swift/usr/bin:${PATH}"

WORKDIR /workspace
