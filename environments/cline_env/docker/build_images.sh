#!/bin/bash
set -e

# Build Cline Docker Images
# Usage: ./build_images.sh [profile...]
# Examples:
#   ./build_images.sh                    # Build all
#   ./build_images.sh base python rust   # Build specific

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REGISTRY=${DOCKER_REGISTRY:-nousresearch}
TAG=${DOCKER_TAG:-latest}

log() {
    echo "[build] $(date '+%H:%M:%S') $*"
}

build_image() {
    local profile=$1
    local dockerfile="$SCRIPT_DIR/$profile/Dockerfile"
    local image_name="$REGISTRY/cline-$profile:$TAG"
    
    if [[ ! -f "$dockerfile" ]]; then
        log "SKIP: No Dockerfile for $profile"
        return 0
    fi
    
    log "Building $image_name"
    
    if [[ "$profile" == "base" ]]; then
        docker build -t "$image_name" -f "$dockerfile" "$SCRIPT_DIR/base"
    else
        docker build \
            --build-arg BASE_IMAGE="$REGISTRY/cline-base:$TAG" \
            -t "$image_name" \
            -f "$dockerfile" \
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

# Always build base first if included
if [[ " ${profiles[*]} " =~ " base " ]]; then
    build_image base
    profiles=("${profiles[@]/base/}")
fi

# Build remaining
for profile in "${profiles[@]}"; do
    [[ -n "$profile" ]] && build_image "$profile"
done

log "Done! Images built:"
docker images | grep "$REGISTRY/cline" | head -20
