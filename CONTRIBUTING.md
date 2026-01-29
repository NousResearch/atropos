## Contributing using GitHub Codespaces (Beginner-friendly)

If you are new to open source or do not want to set up a local development environment, you can contribute using **GitHub Codespaces**.

GitHub Codespaces provides a full cloud-based VSCode environment with terminal access, allowing you to edit files, run tests, and submit pull requests directly from your browser.

### Requirements
- A GitHub account
- A modern web browser (Chrome, Firefox, Edge)

No local installation is required.

---

### Step 1: Fork the repository

1. Visit the Atropos repository.
2. Click the **Fork** button at the top right to create your own copy of the repository.

---

### Step 2: Create a Codespace

1. Go to your forked repository.
2. Click the green **Code** button.
3. Open the **Codespaces** tab.
4. Click **Create codespace on main**.

GitHub will automatically create a cloud Ubuntu VM and open VSCode in your browser.

---

### Step 3: Create a new branch

Open the terminal in Codespaces and run:

```bash
git checkout -b docs/your-branch-name
Replace your-branch-name with a short description of your contribution.

### Step 4: Set up the Python environment

Create and activate a virtual environment:

python -m venv .venv
source .venv/bin/activate

Install dependencies:

pip install -e ".[dev]"


This step may take 15–30 minutes.

Install pre-commit hooks:

pre-commit install

### Step 5: Make your contribution

Most beginner-friendly contributions involve editing documentation files (.md).

Open the file you want to edit in VSCode.

Use Ctrl + Shift + V to preview Markdown changes.

Save your changes with Ctrl + S.

### Step 6: Commit your changes

Stage and commit your changes using a clear commit message:

git add .
git commit -m "docs: add contribution guide using GitHub Codespaces"

### Step 7: Push and open a Pull Request

Push your branch to your fork:

git push origin docs/your-branch-name


Then:

Open your forked repository in the browser.

Click Compare & pull request.

Provide a clear description of your changes.

Submit the pull request.

Notes

GitHub Codespaces provides 60 free hours per month for public repositories.
This is sufficient for documentation and beginner contributions.
No local setup is required.


---


1. **Ctrl + S** (save)
2. back to terminal
3. continue

```bash
git add .
git commit -m "docs: add beginner guide for contributing with GitHub Codespaces"
git push origin docs/my-first-contribution


creat  Pull Request
<!-- test contribution via GitHub Codespaces -->
<!-- save test -->
