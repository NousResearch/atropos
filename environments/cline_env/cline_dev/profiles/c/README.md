# C Profile

This profile defines the toolchain used for C-based repos. It bundles:

- GCC from the pinned `nixos-24.05` nixpkgs (supports `-std=c23`).
- Common native build deps: `cmake`, `ninja`, `pkg-config`, GNU coreutils, etc.

## Usage

### Dev Shell

```bash
cd environments/cline_env/cline_dev/profiles/c
nix develop
```

This drops you into a shell with `gcc`, `cmake`, and other build tools available for running C23 projects.

### Container Image

```bash
nix build .#c-env-image
```

The resulting tarball (`result`) can be loaded into Docker:

```bash
docker load < result
```

Use this image as the base for Nomad/Docker worker jobs that handle C tasks.

