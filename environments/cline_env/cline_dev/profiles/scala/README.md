# Scala Profile

This profile defines the toolchain used for Scala (JVM) repos. It bundles:

- A recent JDK from the pinned `nixos-24.05` nixpkgs (`pkgs.jdk`).
- Scala 3 from nixpkgs (`pkgs.scala_3`, currently 3.x, e.g. 3.3.x).
- SBT (`pkgs.sbt`) for building Scala projects.
- Common native build deps: `cmake`, `pkg-config`, GNU coreutils, etc.

## Usage

### Dev Shell

```bash
cd environments/cline_env/cline_dev/profiles/scala
nix develop
```

This drops you into a shell with `java`, `scala`, `scalac`, and `sbt` on `PATH`, suitable for running Scala builds and tests.

### Container Image

```bash
nix build .#scala-env-image
```

The resulting tarball (`result`) can be loaded into Docker:

```bash
docker load < result
```

Use this image as the base for Nomad/Docker worker jobs that handle Scala tasks.

