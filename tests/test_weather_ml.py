import numpy as np

from weather_ml import (
    build_model,
    evaluate_model,
    generate_mock_weather_data,
    train_and_evaluate,
)


def test_build_and_evaluate_smoke():
    df = generate_mock_weather_data(n_days=60, seed=0)
    features = ["avg_temp", "humidity", "wind_speed", "pressure"]
    X = df[features]
    y = df["next_day_temp"]

    # split a tiny train/test
    X_train = X.iloc[:-10]
    y_train = y.iloc[:-10]
    X_test = X.iloc[-10:]
    y_test = y.iloc[-10:]

    model = build_model(X_train, y_train, n_estimators=10, random_state=0)
    assert hasattr(model, "predict")

    metrics = evaluate_model(model, X_test, y_test, plot=False)
    assert "mae" in metrics and "r2" in metrics
    assert isinstance(metrics["mae"], float)
    assert isinstance(metrics["r2"], float)


def test_train_and_evaluate_returns_metrics():
    df = generate_mock_weather_data(n_days=60, seed=1)
    metrics = train_and_evaluate(df, n_estimators=10)
    assert "mae" in metrics and "r2" in metrics


def test_model_serialization(tmp_path):
    # Ensure model can be saved and loaded with joblib and predictions remain consistent
    df = generate_mock_weather_data(n_days=80, seed=2)
    features = ["avg_temp", "humidity", "wind_speed", "pressure"]
    X = df[features]
    y = df["next_day_temp"]

    X_train = X.iloc[:-20]
    y_train = y.iloc[:-20]
    X_test = X.iloc[-20:]

    model = build_model(X_train, y_train, n_estimators=10, random_state=0)
    preds_before = model.predict(X_test)

    import joblib

    p = tmp_path / "model.joblib"
    joblib.dump(model, p)
    loaded = joblib.load(p)
    preds_after = loaded.predict(X_test)

    # Predictions should be identical (same model state)
    assert np.allclose(preds_before, preds_after)


def test_deterministic_with_random_state():
    # Ensure same seed and model params produce identical metrics
    df1 = generate_mock_weather_data(n_days=80, seed=42)
    df2 = generate_mock_weather_data(n_days=80, seed=42)

    m1 = train_and_evaluate(df1, n_estimators=20, random_state=0)
    m2 = train_and_evaluate(df2, n_estimators=20, random_state=0)

    assert abs(m1["mae"] - m2["mae"]) < 1e-8
    assert abs(m1["r2"] - m2["r2"]) < 1e-8
