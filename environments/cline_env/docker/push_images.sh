#!/bin/bash
set -e

# Push Cline Docker Images to Registry
# Usage: ./push_images.sh [profile...]
# 
# Requires: docker login first
# Example: docker login -u nousresearch

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REGISTRY=${DOCKER_REGISTRY:-nousresearch}
TAG=${DOCKER_TAG:-latest}

log() {
    echo "[push] $(date '+%H:%M:%S') $*"
}

push_image() {
    local profile=$1
    local image_name="$REGISTRY/cline-$profile:$TAG"
    
    if ! docker image inspect "$image_name" &>/dev/null; then
        log "SKIP: Image $image_name not found locally"
        return 0
    fi
    
    log "Pushing $image_name"
    docker push "$image_name"
    log "SUCCESS: $image_name"
}

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

if [[ $# -eq 0 ]]; then
    profiles=("${ALL_PROFILES[@]}")
else
    profiles=("$@")
fi

# Push base first
if [[ " ${profiles[*]} " =~ " base " ]]; then
    push_image base
    profiles=("${profiles[@]/base/}")
fi

for profile in "${profiles[@]}"; do
    [[ -n "$profile" ]] && push_image "$profile"
done

log "Done!"
