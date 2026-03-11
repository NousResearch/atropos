"""
Build static environment metadata for the web hub.

Outputs:
- web/public/environments.json
- web/public/env-data/<slug>.json
"""

import json
import os
import re
import shutil
import sys
from pathlib import Path

TEXT_PREVIEW_EXTENSIONS = {
    ".css",
    ".html",
    ".js",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
}
MAX_PREVIEW_BYTES = 256 * 1024


def has_python_files(dir_path: Path) -> bool:
    """True if directory contains at least one .py file (recursively)."""
    for _ in dir_path.rglob("*.py"):
        return True
    return False


def is_excluded_dir(name: str) -> bool:
    """Exclude pure data/config dirs from being treated as environments."""
    excluded = {"configs", "ifeval_instructions", "__pycache__", ".git"}
    return name in excluded or name.startswith(".")


def get_relative_id(env_dir: Path, environments_root: Path) -> str:
    """Path relative to environments/ as id (e.g. community/word_hunt)."""
    return str(env_dir.relative_to(environments_root)).replace("\\", "/")


def slugify_env_id(env_id: str) -> str:
    """Filesystem-safe slug used for static route params."""
    return env_id.replace("/", "--")


def parse_readme(readme_path: Path) -> tuple[str, str]:
    """Extract first # title and first paragraph from README. Returns (name, description)."""
    name = ""
    description = ""
    if not readme_path.is_file():
        return name, description
    try:
        text = readme_path.read_text(encoding="utf-8", errors="replace")
        lines = text.strip().splitlines()
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                if not name:
                    name = re.sub(r"^#+\s*", "", line).strip()
                continue
            if not description and name:
                description = line
                break
            if not name:
                name = line[:80] if len(line) > 80 else line
                break
    except Exception:
        pass
    return name or "", description or ""


def derive_tags(env_id: str) -> list[str]:
    """Derive tags from path (e.g. community/word_hunt -> community)."""
    parts = env_id.split("/")
    tags = []
    if "community" in parts:
        tags.append("community")
    if "eval_environments" in parts:
        tags.append("eval")
    if "game_environments" in parts:
        tags.append("games")
    if "letter_counting" in env_id or "math" in env_id or "gsm8k" in env_id:
        tags.append("math")
    if "code" in env_id or "swe" in env_id or "coding" in env_id:
        tags.append("coding")
    return list(dict.fromkeys(tags))


def find_environment_dirs(environments_root: Path) -> list[Path]:
    """Find environment root directories: have .py and README, prefer shallowest."""
    candidates: list[Path] = []
    for root, dirs, _ in os.walk(environments_root):
        root_path = Path(root)
        dirs[:] = [d for d in dirs if not is_excluded_dir(d)]
        for d in dirs:
            sub = root_path / d
            if has_python_files(sub):
                candidates.append(sub)

    def has_readme(path: Path) -> bool:
        return (path / "README.md").is_file()

    with_readme = [candidate for candidate in candidates if has_readme(candidate)]
    minimal_readme = [
        candidate
        for candidate in with_readme
        if not any(
            candidate != other and other.is_relative_to(candidate) and has_readme(other)
            for other in with_readme
        )
    ]

    def is_category_root(path: Path) -> bool:
        rel = path.relative_to(environments_root)
        parts = rel.parts
        return len(parts) == 1 and parts[0] in (
            "community",
            "eval_environments",
            "game_environments",
        )

    minimal_readme = [
        candidate for candidate in minimal_readme if not is_category_root(candidate)
    ]
    return sorted(minimal_readme, key=lambda path: str(path))


def list_env_files(env_dir: Path) -> tuple[list[dict], int, str]:
    """Collect relative file paths and metadata for an environment directory."""
    files = []
    total_size = 0
    readme_path = ""

    for file_path in sorted(env_dir.rglob("*")):
        if not file_path.is_file():
            continue
        if any(part.startswith(".") for part in file_path.relative_to(env_dir).parts):
            continue

        relative_path = str(file_path.relative_to(env_dir)).replace("\\", "/")
        stat = file_path.stat()
        total_size += stat.st_size

        if not readme_path and relative_path.lower().endswith("readme.md"):
            readme_path = relative_path

        files.append(
            {
                "path": relative_path,
                "size": stat.st_size,
                "previewable": file_path.suffix.lower() in TEXT_PREVIEW_EXTENSIONS
                and stat.st_size <= MAX_PREVIEW_BYTES,
            }
        )

    files.sort(
        key=lambda item: (
            0 if item["path"].lower().endswith("readme.md") else 1,
            item["path"].lower(),
        )
    )
    return files, total_size, readme_path


def build_manifest(environments_path: Path, output_path: Path) -> None:
    """Scan environments_path and write static manifest JSON files."""
    env_dirs = find_environment_dirs(environments_path)
    entries = []
    details_root = output_path.parent / "env-data"

    if details_root.exists():
        shutil.rmtree(details_root)
    details_root.mkdir(parents=True, exist_ok=True)

    for env_dir in env_dirs:
        env_id = get_relative_id(env_dir, environments_path)
        slug = slugify_env_id(env_id)
        readme_file = env_dir / "README.md"
        name, description = parse_readme(readme_file)
        if not name:
            name = env_dir.name.replace("_", " ").title()
        tags = derive_tags(env_id)
        files, total_size, readme_path = list_env_files(env_dir)

        env_entry = {
            "id": env_id,
            "slug": slug,
            "name": name,
            "description": (description or "")[:500],
            "tags": tags,
            "fileCount": len(files),
            "totalSize": total_size,
            "readmePath": readme_path or None,
        }
        entries.append(env_entry)

        detail_payload = {
            "environment": env_entry,
            "files": files,
            "readmePath": readme_path or None,
            "totalSize": total_size,
        }
        (details_root / f"{slug}.json").write_text(
            json.dumps(detail_payload, indent=2),
            encoding="utf-8",
        )

        # Copy env files to static path so they can be downloaded without API
        env_files_dir = output_path.parent / "env-files" / slug
        if env_files_dir.exists():
            shutil.rmtree(env_files_dir)
        env_files_dir.mkdir(parents=True, exist_ok=True)
        for file_info in files:
            src = env_dir / file_info["path"]
            dst = env_files_dir / file_info["path"]
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    print(
        f"Wrote {len(entries)} environments to {output_path} and static details to {details_root}",
        file=sys.stderr,
    )


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    env_path = os.environ.get("ENVIRONMENTS_PATH")
    if env_path:
        environments_root = Path(env_path)
    else:
        environments_root = repo_root / "environments"
    output_path = repo_root / "web" / "public" / "environments.json"
    if len(sys.argv) > 1:
        output_path = Path(sys.argv[1])
    if not environments_root.is_dir():
        print(f"Environments path not found: {environments_root}", file=sys.stderr)
        sys.exit(1)
    build_manifest(environments_root, output_path)


if __name__ == "__main__":
    main()
