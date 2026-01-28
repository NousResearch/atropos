# Contributing to Atropos

First off, thank you for considering contributing to Atropos! It's people like you that make open source projects such great tools.

We welcome any type of contribution, not just code. You can help with:
* **Reporting a bug**
* **Discussing the current state of the code**
* **Submitting a fix**
* **Proposing new features**
* **Becoming a maintainer**

## We Develop with GitHub
We use GitHub to host the code, track issues and feature requests, and accept pull requests.

## We Use GitHub Flow
We follow the [GitHub Flow](https://docs.github.com) development workflow. All code changes happen through Pull Requests.

## Getting Started

### Project Setup

1.  **Fork the repository:** Click the "Fork" button on the top right of the [repository page](https://github.com/NousResearch/atropos). This creates your own copy of the project.
2.  **Clone your fork:**
    ```bash
    git clone [https://github.com/your-username/atropos.git](https://github.com/your-username/atropos.git)
    cd atropos
    ```
3.  **Set up the development environment:** This project uses standard Python `venv` for environment creation and `pip` for dependency management.
    ```bash
    # Ensure you have Python 3.10+ installed
    # Create and activate a virtual environment
    python -m venv .venv
    
    # Note: If you are on Ubuntu/WSL and get a "command not found" error, use python3:
    python3 -m venv .venv

    source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`

    # Install dependencies, including development dependencies
    pip install -e ".[dev]"
    ```
4.  **Install pre-commit hooks:** This project uses `pre-commit` for code quality checks. The hooks will run automatically when you commit changes.
    ```bash
    pre-commit install
    ```

### Running Tests

We use `pytest` for running tests. To run the test suite:

```bash
pytest