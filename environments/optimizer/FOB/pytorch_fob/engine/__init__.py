from pathlib import Path

from pytorch_fob.engine.engine import Engine


def repository_root() -> Path:
    return Path(__file__).resolve().parent.parent
