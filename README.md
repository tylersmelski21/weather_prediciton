# weather_prediciton

Simple script that generates synthetic NYC daily weather data and trains a RandomForestRegressor to predict next-day temperature.

Quick start

1. Create and activate the virtualenv (optional but recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the script:

```bash
python weather_ml.py
```

API notes

- `train_and_evaluate(df, **model_kwargs)` — convenience wrapper. Returns a metrics dict and accepts any `RandomForestRegressor` constructor kwargs (e.g., `n_estimators`, `max_depth`).
- `build_model(X_train, y_train, **model_kwargs)` — builds and fits a RandomForestRegressor.
- `evaluate_model(model, X_test, y_test, plot=True)` — returns `{'mae', 'r2', 'y_pred'}` and optionally plots results.

The `requirements.txt` lists runtime dependencies. Use `pytest` to run tests in `tests/`.
