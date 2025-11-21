# Shell Profile

This profile provides a general-purpose POSIX shell environment for tasks that primarily require Bash/coreutils rather than a language-specific toolchain. It bundles:

- GNU coreutils/findutils/grep/sed/awk/diffutils and other standard CLI tools.
- `git`, `curl`, `wget`, `make`, `pkg-config`, `openssl`, plus Python 3 for scripting glue.

## Usage

### Dev Shell

```bash
cd environments/cline_env/cline_dev/profiles/shell
nix develop
```

This drops you into a shell with the standard GNU userland ready to run scripts/tests.

### Container Image

```bash
nix build .#shell-env-image
```

The resulting tarball (`result`) can be loaded into Docker:

```bash
docker load < result
```

Use this image as the base for Nomad/Docker worker jobs that just need a Linux shell environment.

