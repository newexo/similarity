# Respondent Similarity

Python utilities for analyzing card-sort survey data. Card sorting is a research technique in which participants organize a fixed set of cards — typically labeled with terms drawn from a user interface or website — into categories. In an **open** card sort, participants invent their own category names; in a **closed** card sort, the categories are fixed in advance and participants assign cards to them.

This project provides tools to load such survey results from tabular sources, quantify how respondents grouped cards, and visualize the resulting structure.

## Goals

- Load card-sort survey responses from CSV or Excel into a normalized tabular form.
- For open sorts, surface and consolidate the categories that participants chose.
- Compute card-to-card similarity matrices based on co-occurrence within participant categories.
- Run hierarchical clustering and render dendrograms to expose latent groupings among the cards.

The project is in an early stage. The package currently contains scaffolding (path resolution helpers, version metadata, test harness); the analysis modules described above are planned.

## Expected Input Format

Raw survey data is expected to arrive in long-format tabular form, with one row per participant–card assignment. Typical columns include:

| Column            | Description                                                  |
|-------------------|--------------------------------------------------------------|
| `participant_id`  | Identifier for the respondent.                               |
| `card_id`         | Stable identifier for the card.                              |
| `card_label`      | Human-readable text on the card.                             |
| `category_label`  | Category the participant placed the card into.               |

Both CSV and Excel inputs are intended to be supported.

## Development Environment

This project pairs a [Conda](https://docs.conda.io/) environment (which provides the Python interpreter) with [Poetry](https://python-poetry.org/) (which manages project dependencies). The conda environment is also the interpreter configured in PyCharm and the target for local test runs, so all installs must go *into* that env rather than into a separate Poetry-managed virtualenv.

### One-time setup

Create the conda environment and install Poetry into it. The Poetry version should match the version pinned in `.github/workflows/python-package.yml` so that local behavior matches CI:

```bash
conda create -n respondent-similarity python=3.11
conda run -n respondent-similarity pip install 'poetry==2.3.4'
```

### Installing dependencies

Install the base package together with the development dependency group (testing and code quality tools — pytest, ruff, coverage, pytest-cov):

```bash
conda run -n respondent-similarity poetry install --with dev
```

Add the optional notebook group (Jupyter, JupyterLab, Matplotlib, Seaborn) when you need interactive exploration:

```bash
conda run -n respondent-similarity poetry install --with dev,notebook
```

If you ever see Poetry print `Creating virtualenv …`, stop and re-run with the `conda run -n respondent-similarity` wrapper — without it, Poetry installs into a separate cached virtualenv and the conda environment silently goes stale.

## Makefile-Based Workflow

The Makefile provides a simple interface for common development tasks and runs all tools inside the Poetry environment. After you create and activate a development environment and install dependencies, issue the following commands from the project root:

| Command              | Description                          |
|----------------------|--------------------------------------|
| `make test`          | Run the test suite.                  |
| `make format`        | Format the code with Ruff.           |
| `make lint`          | Run Ruff lint checks.                |
| `make check`         | Run formatting, linting and tests.   |
| `make coverage`      | Run tests with coverage enforcement. |
| `make coverage-html` | Create an HTML coverage report.      |

These commands support routine quality checks and keep the workflow consistent across local development and continuous integration.

The coverage threshold is defined in the Makefile. Projects should adjust this value to reflect their own testing standards.

## Directory Resolution Utilities

The package includes a `directories` module that centralizes resolution of absolute paths to project locations (the package directory, the project root, the `data/` directory, the test directories). Each function returns either the directory itself or a fully qualified path when given a filename, so analysis code can reference data files without hard-coded paths or assumptions about the current working directory.

```python
from respondent_similarity import directories

filepath = directories.data("example.csv")

with open(filepath, "r") as f:
    contents = f.read()
```

## Continuous Integration

GitHub Actions runs the lint, test, and coverage targets against Python 3.11 through 3.14 on every push to `main` and on every pull request. The workflow is defined in `.github/workflows/python-package.yml` and pins the Poetry version used in CI; keep your local Poetry installation aligned with that pin.
