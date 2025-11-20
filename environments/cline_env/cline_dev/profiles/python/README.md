# Python Profile

This profile defines the toolchain used for Python-based repos. It bundles:

- Python **3.10** (CPython) with `pip` and `virtualenv` for maximum compatibility.
- Node.js 22 + yarn (for building/running Cline when needed in the same container).
- Common native build deps: `openssl`, `pkg-config`, `cmake`, GNU coreutils, etc.

## Usage

### Dev Shell

```bash
cd environments/cline_env/cline_dev/profiles/python
nix develop
```

This drops you into a shell with Python 3.10, `pip`, `virtualenv`, node, and the usual build tools available for running tests and scripts inside a Python workspace.

### Container Image

```bash
nix build .#python-env-image
```

The resulting tarball (`result`) can be loaded into Docker:

```bash
docker load < result
```

Use this image as the base for Nomad/Docker worker jobs that need a Python 3.10 workspace environment alongside Cline.

