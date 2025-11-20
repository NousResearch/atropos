# Go Profile

This profile defines the toolchain used for Go-based repos. It bundles:

- Go from the pinned `nixos-24.05` nixpkgs (recent stable Go toolchain).
- Common native build deps: `openssl`, `pkg-config`, `cmake`, GNU coreutils, etc.

## Usage

### Dev Shell

```bash
cd environments/cline_env/cline_dev/profiles/go
nix develop
```

This drops you into a shell with Go, git, and build tools available for running `go test`, `go build`, etc.

### Container Image

```bash
nix build .#go-env-image
```

The resulting tarball (`result`) can be loaded into Docker:

```bash
docker load < result
```

Use this image as the base for Nomad/Docker worker jobs that handle Go language tasks.

