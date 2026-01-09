# Verifiers / Prime Intellect Environment Hub adapter for Atropos.
#
# Install an environment from Prime's Env Hub (docs):
#   1) uv tool install prime
#   2) prime login
#   3) prime env install owner/environment-name@latest --with pip   # or run in a venv (`uv venv`) without --with
#      e.g. prime env install adtygan/frozen-lake@latest --with pip
#
# Then run this env as an Atropos microservice:
#   run-api
#   python environments/verifiers_server.py serve --config environments/configs/verifiers.yaml
#
from __future__ import annotations

from atroposlib.envs.verifiers_env import VerifiersEnv

if __name__ == "__main__":
    VerifiersEnv.cli()
