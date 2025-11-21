# Jupyter Notebook Profile

This profile defines the toolchain used for notebook-heavy repos. It bundles:

- Python 3.10 with pip/virtualenv plus JupyterLab/Notebook from nixpkgs.
- Node.js/yarn (Jupyter extensions sometimes need Node).
- Common native build deps: `cmake`, `pkg-config`, GNU coreutils, etc.

## Usage

### Dev Shell

```bash
cd environments/cline_env/cline_dev/profiles/jupyter
nix develop
```

This drops you into a shell with `jupyter lab`, `jupyter notebook`, python, pip, etc. available.

### Container Image

```bash
nix build .#jupyter-env-image
```

The resulting tarball (`result`) can be loaded into Docker:

```bash
docker load < result
```

Use this image as the base for Nomad/Docker worker jobs that need to serve notebooks (e.g., start a headless Jupyter server that Cline interacts with).

