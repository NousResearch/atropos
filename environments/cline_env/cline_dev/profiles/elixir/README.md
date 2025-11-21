# Elixir Profile

This profile defines the toolchain used for Elixir (Erlang/OTP) repos. It bundles:

- Erlang/OTP from the pinned `nixos-24.05` nixpkgs.
- Elixir (`pkgs.elixir_1_16`, the latest available in this nixpkgs pin) and Hex.
- Common native build deps: `cmake`, `pkg-config`, GNU coreutils, etc.

## Usage

### Dev Shell

```bash
cd environments/cline_env/cline_dev/profiles/elixir
nix develop
```

This drops you into a shell with `elixir`, `mix`, `iex`, `hex`, git, and build tools available for running tests and scripts.

### Container Image

```bash
nix build .#elixir-env-image
```

The resulting tarball (`result`) can be loaded into Docker:

```bash
docker load < result
```

Use this image as the base for Nomad/Docker worker jobs that handle Elixir tasks.

