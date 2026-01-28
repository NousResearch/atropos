# Contributing to Atropos

First off, thank you for considering contributing to Atropos! It's people like you that make open source projects such great tools.

We welcome any type of contribution, not just code. You can help with:
* **Reporting a bug**
* **Discussing the current state of the code**
* **Submitting a fix**
* **Proposing new features**
* **Becoming a maintainer**

---

## We Develop with GitHub
We use GitHub to host the code, track issues and feature requests, and accept pull requests.

## We Use GitHub Flow
We follow the [GitHub Flow](https://docs.github.com) development workflow. All code changes happen through Pull Requests.

---

## Getting Started

### Project Setup

1. **Fork the repository**
   Click the **Fork** button on the top right of the [repository page](https://github.com/NousResearch/atropos). This creates your own copy of the project.

2. **Clone your fork**
   ```bash
   git clone https://github.com/your-username/atropos.git
   cd atropos
Set up the development environment
This project uses standard Python venv for environment creation and pip for dependency management.

# Ensure you have Python 3.10+ installed
python -m venv .venv
source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
⚠️ Important: After activating the virtual environment, make sure you see (.venv) in your terminal prompt.
If you don’t see it, the environment is not active and the following steps may fail.

Install dependencies (including development dependencies)

pip install -e ".[dev]"
⏱ This step may take 15–30 minutes depending on your system and network speed.
This is expected.

Install pre-commit hooks
This project uses pre-commit for code quality checks. Hooks will run automatically on commit.

pre-commit install
💡 If pre-commit is not found, install it first:

pip install pre-commit
GitHub Codespaces Notes
If you're using GitHub Codespaces:

Python is preinstalled, but virtual environments are not activated automatically

Always confirm that (.venv) appears in your terminal prompt

Authentication for git push is usually handled automatically

If prompted for credentials locally, use a Personal Access Token (PAT) — not your GitHub password

Running Tests
We use pytest for running tests. Before submitting a pull request, ensure all tests pass:

pytest
How to Contribute
Reporting Bugs
We use GitHub issues to track public bugs. Report a bug by opening a new issue on the repository.

When opening a bug report, please use the Bug Report issue template.

Great bug reports include:

A clear summary

Steps to reproduce (exact commands or minimal code)

Expected vs actual behavior

Error messages or logs

Environment details (OS, Python version, package versions)

Suggesting Enhancements
If you have an idea for a new feature or improvement, please open an issue first to discuss it.

Use the Feature Request issue template to help structure the discussion.

Submitting Changes (Pull Requests)
Pull requests are the best way to propose changes.

Create a branch from main

git checkout -b your-feature-or-fix-branch main
Make your changes

Add tests if applicable

Update documentation if behavior or APIs change

Run tests

pytest
Ensure formatting and linting

pre-commit run --all-files
Commit your changes

git add .
git commit -m "Clearly describe the changes made in this commit"
Use clear, descriptive commit messages. Vague messages may be rejected.

Push your branch

git push origin your-feature-or-fix-branch
Open a Pull Request

Provide a clear title and description

Link related issues (e.g. Closes #123)

Follow the appropriate PR template:

environment_pull_request_template.md for environment-related changes

non_environment_pull_request_template.md for all other changes

Code Style
This project follows standard Python style (PEP 8) enforced by:

black

flake8

isort

All checks are managed via pre-commit.

You can run them manually:

pre-commit run --all-files
License for Contributions
All contributions are made under the MIT License.
By submitting a pull request, you agree that your contribution will be licensed under the same terms.

Environment Contribution Guidelines
Since Atropos focuses on reinforcement learning environments:

Place new environments in environments/community/

Treat your environment directory as the import root

Do not submit environments involving illegal activity

Ensure compliance with GitHub policies

Explicit content must be clearly labeled and legally compliant

Avoid copyrighted or reverse-engineered commercial games

Consider ethical implications of the environment

If unsure, open an issue before investing significant development effort.

Code of Conduct
This project follows a Contributor Code of Conduct.
By participating, you agree to abide by its terms.

Thank you again for your contribution!
