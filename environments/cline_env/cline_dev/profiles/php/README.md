# PHP Profile

This profile defines the toolchain used for PHP-based repos. It bundles:

- PHP (from the pinned `nixos-24.05` nixpkgs, typically PHP 8.x).
- Composer (`phpPackages.composer`) for dependency management.
- Common native build deps: `cmake`, `pkg-config`, GNU coreutils, etc.

## Usage

### Dev Shell

```bash
cd environments/cline_env/cline_dev/profiles/php
nix develop
```

This drops you into a shell with `php`, `composer`, git, and build tools available for running tests and scripts.

### Container Image

```bash
nix build .#php-env-image
```

The resulting tarball (`result`) can be loaded into Docker:

```bash
docker load < result
```

Use this image as the base for Nomad/Docker worker jobs that handle PHP tasks.

