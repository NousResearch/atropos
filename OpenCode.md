# OpenCode.md - Atropos Project Guide

## Build/Test/Lint Commands
- **Install dependencies**: `uv sync --all-extras` (includes dev tools)
- **Run all tests**: `uv run pytest`
- **Run single test**: `uv run pytest atroposlib/tests/test_specific.py::test_function_name`
- **Run tests with coverage**: `uv run pytest --cov=atroposlib`
- **Lint code**: `uv run flake8 atroposlib/` (check pyproject.toml for config)
- **Format code**: `uv run black atroposlib/` and `uv run isort atroposlib/`
- **Type check**: `uv run mypy atroposlib/`
- **Pre-commit hooks**: `uv run pre-commit run --all-files`

## Code Style Guidelines
- **Always use `uv run python`** instead of `python` directly (UV dependency management)
- **Imports**: 3 sections (stdlib, third-party, local) with alphabetical ordering within each
- **Type hints**: Required for all function parameters and returns (`def func(x: int) -> str:`)
- **Naming**: snake_case for functions/variables, PascalCase for classes, UPPER_SNAKE_CASE for constants
- **Strings**: Use f-strings for formatting (`f"Processing {count} items"`)
- **Error handling**: Specific exceptions with graceful degradation and logging
- **Docstrings**: Google-style with Args/Returns sections for public functions
- **Line length**: ~88-100 chars, use `# noqa: E501` for necessary exceptions
- **Async**: Use async/await consistently, don't mix with sync patterns
- **Pydantic**: Use for configuration classes with `Field()` descriptions
