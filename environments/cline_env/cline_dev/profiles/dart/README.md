# Dart Profile

This profile defines the toolchain used for Dart-based repos. It bundles:

- The Dart SDK from the pinned `nixos-24.05` nixpkgs.
- Common native build deps: `cmake`, `pkg-config`, GNU coreutils, etc.

## Usage

### Dev Shell

```bash
cd environments/cline_env/cline_dev/profiles/dart
nix develop
```

This drops you into a shell with the Dart CLI (`dart`, `dartfmt`, `dart pub`, etc.) on `PATH`.

### Container Image

```bash
nix build .#dart-env-image
```

The resulting tarball (`result`) can be loaded into Docker:

```bash
docker load < result
```

Use this image as the base for Nomad/Docker worker jobs that handle Dart tasks.

