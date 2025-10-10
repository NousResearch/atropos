# Agent Guidelines for Atropos

## Commands
- **Run all tests**: `pytest`
- **Run single test**: `pytest atroposlib/tests/test_file.py::test_function_name`
- **Lint/format**: `pre-commit run --all-files` (runs black, ruff, flake8)
- **Manual formatting**: `black .` and `isort .`

## Code Style
- **Formatting**: Black (line length 120), enforced via pre-commit hooks
- **Linting**: Flake8 with `--max-line-length=120 --extend-ignore=E203,W503`
- **Import order**: Standard library → third-party → local, sorted by ruff/isort
- **Typing**: Use type hints from `typing` module (Dict, List, Optional, Tuple, etc.)
- **Classes**: Inherit from Pydantic BaseModel or TypedDict for data structures
- **Naming**: snake_case for functions/variables, PascalCase for classes
- **Docstrings**: Use for public methods; include description and parameter/return types
- **Error handling**: Use tenacity for retries, logging for debug info
- **Async**: Use async/await for I/O operations; asyncio for concurrency

## Project Structure
- Core library: `atroposlib/` - base classes, API server, utilities
- Environments: `environments/` - training environments; community contributions go in `environments/community/`
- New environments should use direct imports from their directory root

## Before Committing
Always run `pre-commit run --all-files` to ensure code passes black, isort, flake8, and other checks.
