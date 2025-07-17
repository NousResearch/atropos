# OpenCode.md - Atropos Project Helper

## Build, Lint, and Test Commands
- **Run Lint**: `uv run ruff .`
- **Run Tests**: `uv run pytest` (supports `-k <test_name>` for single tests)
- **Run Type Check**: `uv run mypy .`
- **Build Command**: Systems use dynamic runtime builds (`uv dependencies are always required`)

## Code Style Guidelines
### Imports
- **Grouping**: Standard library first, third-party libraries next, project-specific last.
- **Aliasing Rules**: Prefer short, clear aliases for long module names.

### Naming Conventions
- **Classes**: PascalCase (e.g., `TextWorldEnv`)
- **Methods/Functions**: snake_case (e.g., `generate_action()`)
- **Constants**: UPPERCASE (e.g., `MAX_MEMORY_ITEMS`)

### Formatting
- **Line Length**: Max 88 characters.
- **Spacing**: Use spaces around operators and after commas.
- **Newlines**: One newline at EOF, clear separation between methods.

### Types
- **Annotations**: Mandatory for all functions and methods using Python standard library typing.
- **Generics**: Use `List[]`, `Dict[]`, etc., for explicit type declarations.

### Error Handling
- **Exceptions**: Pointed classed exceptions (e.g., FAISS exceptions handled immediately).

---
## Additional Notes
- **Execution Environment Setup**: Always prefer `uv python` directly validates states; never `run/python bare`.
- **Prompt Editing XML Memory Widen Clarify is final Workflow Priority ensure JSON tool execution.