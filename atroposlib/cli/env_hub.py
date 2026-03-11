"""
Atropos CLI: install, list, delete cached environments.
"""

import os
import shutil
import sys
from pathlib import Path
from typing import Optional

import requests
import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)

console = Console()

DEFAULT_BASE_URL = "http://localhost:3000"

app = typer.Typer(
    name="atropos",
    help="Install, list, and delete Atropos environments (no auth). Uses the web app API.",
)


def get_cache_dir() -> Path:
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", os.path.expanduser("~")))
        return base / "atropos" / "envs"
    base = Path(os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")))
    return base / "atropos" / "envs"


def _env_id_to_dir_name(env_id: str) -> str:
    return env_id.replace("/", "-")


def _dir_name_to_display_id(dir_name: str) -> str:
    return dir_name.replace("-", "/", 1) if "-" in dir_name else dir_name


def _normalize_file_paths(files: list) -> list[str]:
    """Accept API response as list of strings or list of {path, size}."""
    out = []
    for x in files:
        if isinstance(x, dict) and "path" in x:
            out.append(str(x["path"]))
        elif isinstance(x, str) and x:
            out.append(x)
    return out


def download_with_progress(
    base_url: str,
    env_id: str,
    rel_path: str,
    dest_path: Path,
) -> None:
    from urllib.parse import quote

    safe_id = requests.utils.quote(env_id, safe="")
    safe_path = quote(rel_path.replace("\\", "/"), safe="/")
    url = f"{base_url.rstrip('/')}/api/environments/{safe_id}/files/{safe_path}"
    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()
    total = int(resp.headers.get("Content-Length", 0)) or None
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        DownloadColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(rel_path, total=total)
        with open(dest_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=64 * 1024):
                if chunk:
                    f.write(chunk)
                    progress.update(task, advance=len(chunk))


@app.command()
def install(
    env_id: str = typer.Argument(..., help="Environment id (e.g. community/word_hunt)"),
    base_url: str = typer.Option(
        os.environ.get("ATROPOS_BASE_URL", DEFAULT_BASE_URL),
        "--base-url",
        help="Web app base URL (e.g. http://localhost:3000)",
    ),
    cache_dir: Optional[Path] = typer.Option(
        None,
        "--cache-dir",
        help="Override cache dir",
        path_type=Path,
    ),
) -> None:
    """Download and install an environment."""
    cache = cache_dir if cache_dir is not None else get_cache_dir()
    list_url = f"{base_url.rstrip('/')}/api/environments/{requests.utils.quote(env_id, safe='')}/files"
    try:
        r = requests.get(list_url, timeout=30)
        r.raise_for_status()
        raw = r.json()
        files = _normalize_file_paths(raw if isinstance(raw, list) else [])
    except requests.RequestException as e:
        console.print(f"[red]Failed to list files: {e}[/red]")
        raise typer.Exit(1)
    if not files:
        console.print("[red]No files returned[/red]")
        raise typer.Exit(1)

    cache.mkdir(parents=True, exist_ok=True)
    dir_name = _env_id_to_dir_name(env_id)
    dest_dir = cache / dir_name
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    dest_dir.mkdir(parents=True)

    for rel_path in files:
        if ".." in rel_path or rel_path.startswith("/"):
            continue
        dest_path = dest_dir / rel_path
        try:
            download_with_progress(base_url, env_id, rel_path, dest_path)
        except requests.RequestException as e:
            console.print(f"[red]Failed to download {rel_path}: {e}[/red]")
            raise typer.Exit(1)

    console.print(f"[green]Installed to {dest_dir}[/green]")


@app.command("list")
def list_cached(
    cache_dir: Optional[Path] = typer.Option(
        None,
        "--cache-dir",
        help="Override cache dir",
        path_type=Path,
    ),
) -> None:
    """List cached environments."""
    cache = cache_dir if cache_dir is not None else get_cache_dir()
    if not cache.is_dir():
        console.print("No cached environments.")
        return
    entries = sorted(cache.iterdir())
    dirs = [e for e in entries if e.is_dir()]
    if not dirs:
        console.print("No cached environments.")
        return
    from rich.table import Table

    table = Table(show_header=True, header_style="bold")
    table.add_column("ID", style="cyan")
    table.add_column("Path")
    for d in dirs:
        table.add_row(_dir_name_to_display_id(d.name), str(d))
    console.print(table)


@app.command()
def delete(
    env_id: str = typer.Argument(..., help="Environment id to remove from cache"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
    cache_dir: Optional[Path] = typer.Option(
        None,
        "--cache-dir",
        help="Override cache dir",
        path_type=Path,
    ),
) -> None:
    """Remove an environment from cache."""
    cache = cache_dir if cache_dir is not None else get_cache_dir()
    dir_name = env_id.replace("/", "-")
    dest_dir = cache / dir_name
    if not dest_dir.is_dir():
        console.print(f"[red]Not found in cache: {env_id}[/red]")
        raise typer.Exit(1)
    if not yes:
        try:
            confirm = input(f"Delete {dest_dir}? [y/N]: ").strip().lower()
        except EOFError:
            confirm = "n"
        if confirm != "y":
            return
    shutil.rmtree(dest_dir)
    console.print(f"[green]Deleted {env_id}[/green]")


def main() -> int:
    app()
    return 0


if __name__ == "__main__":
    sys.exit(main())
