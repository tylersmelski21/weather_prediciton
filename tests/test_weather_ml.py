import numpy as np
from weather_ml import (
    generate_mock_weather_data,
    build_model,
    evaluate_model,
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
