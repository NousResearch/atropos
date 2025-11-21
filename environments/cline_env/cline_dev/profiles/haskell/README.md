# Haskell Profile

This profile defines the toolchain used for Haskell repos. It bundles:

- GHC 9.8 (via `haskell.compiler.ghc948` when available in the pinned nixpkgs).
- `cabal-install` and `stack`.
- Common native build deps: `cmake`, `pkg-config`, GNU coreutils, etc.

## Usage

### Dev Shell

```bash
cd environments/cline_env/cline_dev/profiles/haskell
nix develop
```

This drops you into a shell with `ghc`, `cabal`, `stack`, git, and build tools available for running tests and scripts.

### Container Image

```bash
nix build .#haskell-env-image
```

The resulting tarball (`result`) can be loaded into Docker:

```bash
docker load < result
```

Use this image as the base for Nomad/Docker worker jobs that handle Haskell tasks.

