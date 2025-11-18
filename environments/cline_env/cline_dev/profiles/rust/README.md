# Rust Profile

This profile defines the toolchain used for Rust-based repos (e.g., `ratatui`). It bundles:

- Latest stable Rust (`rustc`, `cargo`, `clippy`, `rustfmt`).
- Node.js 22 + yarn (required for compiling Cline).
- Common native build deps: `openssl`, `pkg-config`, `cmake`, `python3`, GNU coreutils.

## Usage

### Dev Shell

```
cd environments/cline_env/cline_dev/profiles/rust
nix develop
```

This drops you into a shell with cargo + node ready for running `cargo test`, `npm install`, or `npm run compile-standalone`.

### Container Image

```
nix build .#rust-env-image
```

The resulting tarball (`result`) can be loaded into Docker:

```
docker load < result
```

Use this image as the base for Nomad/Docker worker jobs that need Rust + Cline.

## Future Work

- Add caching mounts for cargo and npm directories.
- Provide `dockerTools.buildLayer` outputs optimized for Determinate Systems Nix installer caches.
