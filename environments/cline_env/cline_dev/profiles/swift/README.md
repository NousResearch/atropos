Swift profile
=============

This Nix flake provides a Swift toolchain suitable for building and running
Swift-based projects inside the Cline worker.

- Toolchain: `pkgs.swift` from the pinned `nixos-24.05` nixpkgs
- Included tools: `swift`, `git`, `pkg-config`, `cmake`, and common Unix utilities

The `devShell` is what the Cline worker uses via:

```bash
nix develop .#swift
```

and should be sufficient for typical Swift package manager (SwiftPM)–based
projects.

