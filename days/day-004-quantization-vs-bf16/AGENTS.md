# Repository Guidelines

This journal is a working lab notebook. Keep changes incremental, traceable, and easy to reproduce.

## Project Structure & Module Organization

- `days/`: daily logs, experiments, and scratch code (e.g. `days/day-004-quantization-vs-bf16`).
- `topics/`: cleaned-up notes and reusable code (e.g. `topics/vllm/example_server.py`).
- `benchmarks/`: source-of-truth benchmark CSVs and plots.
- `scripts/`: small utilities like `run_benchmark.sh` and `plot_benchmarks.py` (both start as stubs).
- `assets/` and `books/`: figures and reading; edit only when necessary.

Start work in `days/`; promote only stable patterns into `topics/` and `scripts/`.

## Build, Test, and Development Commands

- Create a Python environment (example): `python -m venv .venv && source .venv/bin/activate`.
- Prototype benchmarks: `./scripts/run_benchmark.sh --model ... --output benchmarks/run.csv` (currently a TODO stub).
- Prototype plotting: `python scripts/plot_benchmarks.py benchmarks/run.csv` (raises `NotImplementedError` until implemented).

Document any extra dependencies near the code that needs them.

## Coding Style & Naming Conventions

- Python: 4-space indentation, `snake_case` for functions/modules, `CamelCase` for classes, type hints where useful.
- Day folders: `day-XYZ-short-topic` (zero-padded index).
- Keep modules small; avoid committing large raw outputs or notebooks.

Introduce formatters or linters only when scoped and mention them in your PR.

## Testing Guidelines

There is no shared test suite yet. For utilities in `topics/` or `scripts/`:

- Add a minimal CLI or `main()` that exercises core paths.
- Include small example inputs/outputs in docstrings or nearby Markdown.
- Validate benchmark scripts with a quick, low-cost configuration.

## Commit & Pull Request Guidelines

- Use short, descriptive commit titles tied to a day or task, e.g. `day03 write up`, `task 04 - labs prepared`.
- PR descriptions should state motivation, touched areas (`days/`, `topics/`, etc.), commands run, and any benchmark impact.
- Attach plots or key metrics when changing `benchmarks/` or performance-critical code.

## Agent-Specific Instructions

- Do not reorganize directories or rename day folders unless explicitly requested.
- Prefer updating existing notes or templates over creating new top-level files.
- Modify `benchmarks/` data only when asked to add or refresh results.

