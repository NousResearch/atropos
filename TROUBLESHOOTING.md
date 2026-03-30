# Troubleshooting

Common issues encountered during setup and how to resolve them.

## Virtual Environment

### `python` command not found
Some systems use `python3` instead of `python`. Try:
```bash
python3 -m venv .venv
```

### Virtual environment not activating (Windows)
If `.venv\Scripts\activate` fails, run this first:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```
Then try activating again.

## Installation

### `pip install -e ".[dev]"` fails
Ensure you have Python 3.10+ installed:
```bash
python --version
```
If below 3.10, update your Python installation.

### `pre-commit install` fails
Ensure your virtual environment is activated before running:
```bash
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
pre-commit install
```

## Tests

### `pytest` command not found
Ensure dev dependencies are installed:
```bash
pip install -e ".[dev]"
```
