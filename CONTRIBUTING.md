# Contributing to Bio-WM

Thank you for your interest in contributing! Here's how to get started.

## Getting Started

1. **Fork** the repository and clone it locally.
2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Linux / macOS
   .venv\Scripts\activate      # Windows
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the evaluation harness to confirm your setup:
   ```bash
   python eval_harness.py
   ```

## Development Guidelines

- **Python 3.10+**: All code must be compatible with Python 3.10 or later.
- **Type Hints**: Use type annotations for function signatures and class attributes.
- **Configuration**: All hyperparameters belong in `bio-jepa-lewm/config/bio_lewm_config.yaml` — do not hard-code values.
- **Commits**: Use conventional commits (`feat:`, `fix:`, `docs:`, `refactor:`, `perf:`).

## Submitting a Pull Request

1. Create a branch from `main`: `git checkout -b feat/your-feature`.
2. Make your changes and verify the evaluation harness passes.
3. Open a PR with a clear description of what you changed and why.
4. Reference any related issues.

## Reporting Issues

Use the [GitHub Issues](../../issues) tab. Please include:
- A clear description of the bug or feature request.
- Steps to reproduce (for bugs).
- Expected vs. actual behavior.
- Relevant metrics output if applicable.

---

All contributions are subject to the [License](LICENSE).
