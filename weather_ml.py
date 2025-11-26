from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


def generate_mock_weather_data(n_days: int = 365 * 3, seed: int = 42) -> pd.DataFrame:
    """Generate mock daily weather data for NYC over n_days.

    Features (simplified & synthetic):
      - date
      - avg_temp (°C)
      - humidity (% )
      - wind_speed (m/s)
      - pressure (hPa)

    Target:
      - next_day_temp (°C)
    """
    rng = np.random.default_rng(seed)

    start_date = datetime(2020, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(n_days)]

    # Create a seasonal temperature pattern to mimic NYC
    days = np.arange(n_days)
    # Base sinusoidal pattern for yearly seasonality
    seasonal = 10 * np.sin(2 * np.pi * days / 365)  # ~[-10, 10]

    # Base temperature around 12°C with some noise
    base_temp = 12 + seasonal + rng.normal(0, 3, size=n_days)

    # Humidity between 40% and 90%, slightly higher in warmer days
    humidity = 60 + 0.3 * seasonal + rng.normal(0, 5, size=n_days)
    humidity = np.clip(humidity, 40, 95)

    # Wind speed between 0 and 12 m/s
    wind_speed = np.clip(rng.normal(4, 2, size=n_days), 0, 12)

    # Pressure around standard 1013 hPa
    pressure = 1013 + rng.normal(0, 8, size=n_days)

    df = pd.DataFrame(
        {
            "date": dates,
            "avg_temp": base_temp,
            "humidity": humidity,
            "wind_speed": wind_speed,
            "pressure": pressure,
        }
    )

    # Target: next day's temperature (shifted)
    df["next_day_temp"] = df["avg_temp"].shift(-1)

    # Drop last row with NaN target
    df = df.dropna().reset_index(drop=True)
    return df


def build_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    **model_kwargs,
) -> RandomForestRegressor:
    """Create and fit a RandomForestRegressor.

    Extra kwargs are forwarded to the constructor.
    """
    params = {
        "n_estimators": 200,
        "random_state": 42,
        "n_jobs": -1,
    }
    params.update(model_kwargs)

    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)
    return model


def evaluate_model(
    model: RandomForestRegressor,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    plot: bool = True,
) -> dict:
    """Evaluate `model` on test data and optionally plot predictions.

    Returns a dict: {'mae': float, 'r2': float, 'y_pred': np.ndarray}
    """
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("RandomForestRegressor results (NYC mock weather):")
    print(f"  MAE: {mae:.2f} °C")
    print(f"  R^2: {r2:.3f}")

    if plot:
        sample_idx = np.arange(0, min(100, len(y_test)))
        plt.figure(figsize=(10, 5))
        plt.plot(
            sample_idx,
            y_test.iloc[sample_idx].values,
            label="True",
            marker="o",
            linestyle="-",
        )
        plt.plot(
            sample_idx,
            y_pred[sample_idx],
            label="Predicted",
            marker="x",
            linestyle="--",
        )
        plt.title("Next-Day Temperature Prediction (sample)")
        plt.xlabel("Sample index")
        plt.ylabel("Temperature (°C)")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return {"mae": mae, "r2": r2, "y_pred": y_pred}


def train_and_evaluate(df: pd.DataFrame, **model_kwargs) -> None:
    """Train a RandomForestRegressor on the provided dataframe and evaluate it.

    Any extra keyword arguments are forwarded to `RandomForestRegressor` constructor
    (they override the file's defaults when provided).
    """

    features = ["avg_temp", "humidity", "wind_speed", "pressure"]
    target = "next_day_temp"

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Build and evaluate using small helpers so callers can reuse them.
    model = build_model(X_train, y_train, **model_kwargs)
    metrics = evaluate_model(model, X_test, y_test, plot=True)
    # metrics dict contains 'mae' and 'r2' (returned for programmatic use)
    return metrics


def main() -> None:
    df = generate_mock_weather_data()
    print("First few rows of the generated NYC weather data:")
    print(df.head())
    print("\nTraining model...\n")
    train_and_evaluate(df)


if __name__ == "__main__":
    main()
