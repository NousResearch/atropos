"""
Profile Registry for Cline Docker Images

Maps programming languages to Docker image profiles for Modal execution.
"""

from dataclasses import dataclass
from typing import Dict, Iterable

# Docker registry configuration
DOCKER_REGISTRY = "nousresearch"
DOCKER_TAG = "latest"


@dataclass(frozen=True)
class ProfileConfig:
    """Configuration for a language profile Docker image."""
    profile_key: str
    image_name: str
    
    @property
    def full_image_name(self) -> str:
        return f"{DOCKER_REGISTRY}/cline-{self.profile_key}:{DOCKER_TAG}"


def _profile(profile_key: str) -> ProfileConfig:
    return ProfileConfig(
        profile_key=profile_key,
        image_name=f"cline-{profile_key}",
    )


# Available Docker image profiles
PROFILE_REGISTRY: Dict[str, ProfileConfig] = {
    "base": _profile("base"),
    "python": _profile("python"),
    "rust": _profile("rust"),
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


# Map dataset language names to Docker profile keys
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
    "Haskell": "haskell",
    "Swift": "swift",
    "Shell": "shell",
    "HTML": "node",  # HTML tasks use Node.js environment
}


def supported_languages() -> Iterable[str]:
    """Return all supported dataset languages."""
    return LANGUAGE_TO_PROFILE.keys()


def get_profile_config(language: str) -> ProfileConfig:
    """Get the Docker profile config for a dataset language."""
    profile_key = LANGUAGE_TO_PROFILE.get(language)
    if not profile_key:
        raise KeyError(f"Language '{language}' is not mapped to a Docker profile")
    config = PROFILE_REGISTRY.get(profile_key)
    if not config:
        raise KeyError(f"Profile '{profile_key}' is not defined")
    return config


def get_docker_image(language: str) -> str:
    """Get the full Docker image name for a language."""
    config = get_profile_config(language)
    return config.full_image_name
