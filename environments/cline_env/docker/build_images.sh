#!/bin/bash
set -e

# Build Cline Docker Images for Modal (linux/amd64)
# Usage: ./build_images.sh [profile...]
# Examples:
#   ./build_images.sh                    # Build all
#   ./build_images.sh base python rust   # Build specific
#   PUSH=1 ./build_images.sh             # Build and push

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REGISTRY=${DOCKER_REGISTRY:-nousresearch}
TAG=${DOCKER_TAG:-latest}
PLATFORM=${DOCKER_PLATFORM:-linux/amd64}
PUSH=${PUSH:-0}

log() {
    echo "[build] $(date '+%H:%M:%S') $*"
}

# Ensure buildx is available and set up
setup_buildx() {
    log "Setting up Docker buildx for cross-platform builds..."
    
    # Check if buildx is available
    if ! docker buildx version &>/dev/null; then
        log "ERROR: Docker buildx not available. Please install Docker Desktop or buildx plugin."
        exit 1
    fi
    
    # Create/use a builder that supports multi-platform
    BUILDER_NAME="cline-builder"
    if ! docker buildx inspect "$BUILDER_NAME" &>/dev/null; then
        log "Creating buildx builder: $BUILDER_NAME"
        docker buildx create --name "$BUILDER_NAME" --use --bootstrap
    else
        docker buildx use "$BUILDER_NAME"
    fi
    
    log "Using builder: $(docker buildx inspect --bootstrap 2>/dev/null | head -1 || echo $BUILDER_NAME)"
}

build_image() {
    local profile=$1
    local dockerfile="$SCRIPT_DIR/$profile/Dockerfile"
    local image_name="$REGISTRY/cline-$profile:$TAG"
    
    if [[ ! -f "$dockerfile" ]]; then
        log "SKIP: No Dockerfile for $profile"
        return 0
    fi
    
    log "Building $image_name (platform: $PLATFORM)"
    
    # Build args
    local build_args=(
        --platform "$PLATFORM"
        -t "$image_name"
        -f "$dockerfile"
    )
    
    # Add push flag if requested
    if [[ "$PUSH" == "1" ]]; then
        build_args+=(--push)
    else
        build_args+=(--load)
    fi
    
    if [[ "$profile" == "base" ]]; then
        docker buildx build "${build_args[@]}" "$SCRIPT_DIR/base"
    else
        docker buildx build \
            --build-arg BASE_IMAGE="$REGISTRY/cline-base:$TAG" \
            "${build_args[@]}" \
            "$SCRIPT_DIR/$profile"
    fi
    
    log "SUCCESS: $image_name"
}

# All available profiles
ALL_PROFILES=(
    base
    python
    rust
    node
    go
    cpp
    c
    java
    csharp
    kotlin
    php
    scala
    ruby
    dart
    lua
    elixir
    jupyter
    haskell
    swift
    shell
)

# If no args, build all; otherwise build specified
if [[ $# -eq 0 ]]; then
    profiles=("${ALL_PROFILES[@]}")
else
    profiles=("$@")
fi

# Setup buildx
setup_buildx

# Always build base first if included
if [[ " ${profiles[*]} " =~ " base " ]]; then
    build_image base
    profiles=("${profiles[@]/base/}")
fi

# Build remaining
for profile in "${profiles[@]}"; do
    [[ -n "$profile" ]] && build_image "$profile"
done

log "Done! Images built for platform: $PLATFORM"
docker images | grep "$REGISTRY/cline" | head -20 || true
