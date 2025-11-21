# Ruby Profile

This profile defines the toolchain used for Ruby-based repos. It bundles:

- Ruby 3.3 (from the pinned `nixos-24.05` nixpkgs) and Bundler.
- Common native build deps: `cmake`, `pkg-config`, GNU coreutils, etc.

## Usage

### Dev Shell

```bash
cd environments/cline_env/cline_dev/profiles/ruby
nix develop
```

This drops you into a shell with `ruby`, `bundle`, git, and build tools available for running tests and scripts.

### Container Image

```bash
nix build .#ruby-env-image
```

The resulting tarball (`result`) can be loaded into Docker:

```bash
docker load < result
```

Use this image as the base for Nomad/Docker worker jobs that handle Ruby tasks.

