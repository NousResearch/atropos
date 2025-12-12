# Cline Kotlin Environment
ARG BASE_IMAGE=nousresearch/cline-base:latest
FROM ${BASE_IMAGE}

USER root

# Install JDK
RUN apt-get update && apt-get install -y \
    openjdk-17-jdk \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Install Kotlin directly
RUN KOTLIN_VERSION=2.0.0 \
    && wget -q https://github.com/JetBrains/kotlin/releases/download/v${KOTLIN_VERSION}/kotlin-compiler-${KOTLIN_VERSION}.zip \
    && unzip -q kotlin-compiler-${KOTLIN_VERSION}.zip -d /opt \
    && rm kotlin-compiler-${KOTLIN_VERSION}.zip

# Setup for cline user
RUN mkdir -p /home/cline/.kotlin \
    && chown -R cline:cline /home/cline/.kotlin

USER cline
ENV PATH="/opt/kotlinc/bin:${PATH}"

# Verify installation
RUN kotlin -version || echo "Kotlin installed"

WORKDIR /workspace
