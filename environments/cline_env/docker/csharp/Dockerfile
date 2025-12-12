# Cline C# Environment
ARG BASE_IMAGE=nousresearch/cline-base:latest
FROM ${BASE_IMAGE}

USER root

# Install .NET SDK
RUN wget https://packages.microsoft.com/config/ubuntu/22.04/packages-microsoft-prod.deb -O packages-microsoft-prod.deb \
    && dpkg -i packages-microsoft-prod.deb \
    && rm packages-microsoft-prod.deb \
    && apt-get update \
    && apt-get install -y dotnet-sdk-8.0 \
    && rm -rf /var/lib/apt/lists/*

# Setup for cline user
RUN mkdir -p /home/cline/.dotnet /home/cline/.nuget \
    && chown -R cline:cline /home/cline/.dotnet /home/cline/.nuget

USER cline
ENV DOTNET_CLI_HOME=/home/cline/.dotnet
ENV NUGET_PACKAGES=/home/cline/.nuget/packages

# Verify installation
RUN dotnet --version

WORKDIR /workspace
