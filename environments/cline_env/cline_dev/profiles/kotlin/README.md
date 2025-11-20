# Kotlin Profile

This profile defines the toolchain used for Kotlin (JVM) repos. It bundles:

- A recent JDK from the pinned `nixos-24.05` nixpkgs (`pkgs.jdk`).
- The Kotlin compiler/CLI (`pkgs.kotlin`) and Gradle (`pkgs.gradle`).
- Common native build deps: `cmake`, `pkg-config`, GNU coreutils, etc.

## Usage

### Dev Shell

```bash
cd environments/cline_env/cline_dev/profiles/kotlin
nix develop
```

This drops you into a shell with `java`, `kotlinc`, `kotlin`, and `gradle` on `PATH`, suitable for running Kotlin builds and tests.

### Container Image

```bash
nix build .#kotlin-env-image
```

The resulting tarball (`result`) can be loaded into Docker:

```bash
docker load < result
```

Use this image as the base for Nomad/Docker worker jobs that handle Kotlin tasks.

