# Lua Profile

This profile defines the toolchain used for Lua-based repos. It bundles:

- Lua 5.4 (from the pinned `nixos-24.05` nixpkgs) and Luarocks.
- Common native build deps: `cmake`, `pkg-config`, GNU coreutils, etc.

## Usage

### Dev Shell

```bash
cd environments/cline_env/cline_dev/profiles/lua
nix develop
```

This drops you into a shell with `lua`, `luarocks`, git, and build tools available for running tests and scripts.

### Container Image

```bash
nix build .#lua-env-image
```

The resulting tarball (`result`) can be loaded into Docker:

```bash
docker load < result
```

Use this image as the base for Nomad/Docker worker jobs that handle Lua tasks.

