# C# / .NET Profile

This profile defines the toolchain used for C#-based repos. It bundles:

- A recent .NET SDK (via `dotnet-sdk_8` from the pinned `nixos-24.05` nixpkgs), providing modern C# language support.
- Common native build deps: `cmake`, `pkg-config`, GNU coreutils, etc.

## Usage

### Dev Shell

```bash
cd environments/cline_env/cline_dev/profiles/csharp
nix develop
```

This drops you into a shell with the `dotnet` CLI on `PATH`, suitable for running `dotnet build`, `dotnet test`, etc.

### Container Image

```bash
nix build .#csharp-env-image
```

The resulting tarball (`result`) can be loaded into Docker:

```bash
docker load < result
```

Use this image as the base for Nomad/Docker worker jobs that handle C# tasks.

