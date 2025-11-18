# Cline Dev Harness

This directory houses per-repo experiments for building, testing, and packaging Cline worker environments before they are promoted to Nomad-managed jobs. Each subdirectory documents a reproducible workflow (Nix profile + bootstrap scripts) for a specific language/toolchain or exemplar repo.

## Layout

```
cline_dev/
  profiles/
    rust/
      flake.nix          # rust-env toolchain definition (Nix)
      default.nix        # (optional) wrapper for dockerTools.buildLayer
      README.md          # profile-specific notes
  examples/
    ratatui_vertical_gauge/
      bootstrap.sh       # clones repo, installs deps, runs smoke tests
      README.md          # describes the scenario + how to run it locally
```

## Workflow

1. Pick a dataset row / repo.
2. Create an example under `examples/` with:
   - Reference to the dataset entry and target prompt.
   - Any repo-specific setup (patches, env vars, secrets).
   - Commands to build/run the Cline core inside a container.
3. If the repo requires a new toolchain, create a profile in `profiles/` and document build instructions.
4. Once the flow works locally (docker-compose or direnv + Nix shell), wire it into the Atropos `ClineAgentEnv` via the worker manager.

This structure keeps experimental scripts versioned while we iterate on the automation story.
