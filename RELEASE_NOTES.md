# Release v0.1.0

Draft release notes for `v0.1.0`.

Changelog
- Add `build_model` and `evaluate_model` helpers; allow `train_and_evaluate(df, **model_kwargs)` to pass model params.
- Add tests (including serialization and determinism checks) and CI to run them.
- Add dev tooling: `pre-commit`, `black`, `ruff`, `isort`, `pyproject.toml`, and `dev-requirements.txt`.
- Pin exact runtime dependencies in `requirements.txt`.
- Add GitHub Actions workflows for CI and release; add `Makefile` and `README.md`.

Notes
- The release workflow will build a distribution on tag push; to publish to PyPI set the `PYPI_API_TOKEN` repository secret.
