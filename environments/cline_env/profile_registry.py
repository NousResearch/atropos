from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable

BASE_DIR = Path(__file__).resolve().parent
CLINE_DEV_DIR = BASE_DIR / "cline_dev"
GENERIC_BOOTSTRAP = CLINE_DEV_DIR / "examples" / "generic" / "bootstrap.sh"


@dataclass(frozen=True)
class ProfileConfig:
    profile_key: str
    profile_dir: Path
    bootstrap_script: Path


def _profile(profile_key: str, bootstrap_script: Path = GENERIC_BOOTSTRAP) -> ProfileConfig:
    return ProfileConfig(
        profile_key=profile_key,
        profile_dir=CLINE_DEV_DIR / "profiles" / profile_key,
        bootstrap_script=bootstrap_script,
    )


PROFILE_REGISTRY: Dict[str, ProfileConfig] = {
    "rust": _profile("rust"),
    "python": _profile("python"),
    "node": _profile("node"),
    "go": _profile("go"),
    "cpp": _profile("cpp"),
    "c": _profile("c"),
    "java": _profile("java"),
    "csharp": _profile("csharp"),
    "kotlin": _profile("kotlin"),
    "php": _profile("php"),
    "scala": _profile("scala"),
    "ruby": _profile("ruby"),
    "dart": _profile("dart"),
    "lua": _profile("lua"),
    "elixir": _profile("elixir"),
    "jupyter": _profile("jupyter"),
    "haskell": _profile("haskell"),
    "swift": _profile("swift"),
    "shell": _profile("shell"),
}


LANGUAGE_TO_PROFILE: Dict[str, str] = {
    "Rust": "rust",
    "Python": "python",
    "TypeScript": "node",
    "JavaScript": "node",
    "Go": "go",
    "C++": "cpp",
    "C": "c",
    "Java": "java",
    "C#": "csharp",
    "Kotlin": "kotlin",
    "PHP": "php",
    "Scala": "scala",
    "Ruby": "ruby",
    "Dart": "dart",
    "Lua": "lua",
    "Elixir": "elixir",
    "Jupyter Notebook": "jupyter",
    "Python Notebook": "jupyter",
    "Haskell": "haskell",
    "Swift": "swift",
    "Shell": "shell",
    "HTML": "node",
}


def supported_languages() -> Iterable[str]:
    return LANGUAGE_TO_PROFILE.keys()


def get_profile_config(language: str) -> ProfileConfig:
    profile_key = LANGUAGE_TO_PROFILE.get(language)
    if not profile_key:
        raise KeyError(f"Language '{language}' is not mapped to a Cline profile")
    config = PROFILE_REGISTRY.get(profile_key)
    if not config:
        raise KeyError(f"Profile '{profile_key}' is not defined")
    return config
