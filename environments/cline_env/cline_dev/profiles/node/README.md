# Node / TypeScript Profile

This profile defines the toolchain used for TypeScript and JavaScript repos. It bundles:

- Node.js LTS (from Nixpkgs, currently `nodejs_22`) plus `yarn`.
- Common native build deps: `openssl`, `pkg-config`, `cmake`, GNU coreutils, etc.

It assumes each cloned repo manages its own TypeScript toolchain (e.g. `tsc` via `npx` or local dev scripts).

## Usage

### Dev Shell

```bash
cd environments/cline_env/cline_dev/profiles/node
nix develop
```

This drops you into a shell with Node LTS, yarn, git, and build tools available for running `npm test`, `npm run build`, `npx tsc`, etc.

### Container Image

```bash
nix build .#node-env-image
```

The resulting tarball (`result`) can be loaded into Docker:

```bash
docker load < result
```

Use this image as the base for Nomad/Docker worker jobs that handle TypeScript and JavaScript tasks.

