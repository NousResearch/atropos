# Cline PHP Environment
ARG BASE_IMAGE=nousresearch/cline-base:latest
FROM ${BASE_IMAGE}

USER root

# Install PHP and common extensions
RUN apt-get update && apt-get install -y \
    php \
    php-cli \
    php-common \
    php-curl \
    php-json \
    php-mbstring \
    php-xml \
    php-zip \
    composer \
    && rm -rf /var/lib/apt/lists/*

# Setup for cline user
RUN mkdir -p /home/cline/.composer \
    && chown -R cline:cline /home/cline/.composer

USER cline
ENV COMPOSER_HOME=/home/cline/.composer

# Verify installation
RUN php --version

WORKDIR /workspace
