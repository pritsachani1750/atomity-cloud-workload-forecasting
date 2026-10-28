import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path("challenge_data")
OUTPUT_DIR = Path("reports")
OUTPUT_DIR.mkdir(exist_ok=True)

HORIZON = 24
TRAIN_SIZE = 110
WINDOW = 48
MOVING_AVERAGE_WINDOW = 12

targets = pd.read_csv(DATA_DIR / "aggregate_targets.csv")

required = [
    "timestamp",
    "active_vm_count",
    "aggregate_cpu_utilization",
]
if list(targets.columns) != required:
    raise ValueError(f"Unexpected target columns: {targets.columns.tolist()}")

if len(targets) != 134:
    raise ValueError(f"Expected 134 timestamps; got {len(targets)}")

train = targets.iloc[:TRAIN_SIZE].copy()
future_timestamps = targets.iloc[TRAIN_SIZE:]["timestamp"].to_numpy()

if len(train) != 110 or len(future_timestamps) != HORIZON:
    raise ValueError("Unexpected training or forecast horizon.")

def forecast_persistence(history, horizon):
    return np.repeat(history[-1], horizon)

def forecast_moving_average(history, horizon):
    return np.repeat(history[-MOVING_AVERAGE_WINDOW:].mean(), horizon)

def forecast_linear_trend(history, horizon):
    recent = history[-WINDOW:]
    x = np.arange(len(recent), dtype=float)
    future_x = np.arange(
        len(recent), len(recent) + horizon, dtype=float
    )
    slope, intercept = np.polyfit(x, recent, 1)
    return intercept + slope * future_x

models = {
    "persistence": forecast_persistence,
    "moving_average_12": forecast_moving_average,
    "linear_trend_48": forecast_linear_trend,
}

# All validation folds are strictly inside the first 110 observations.
# Each fold predicts 24 steps ahead; no random splitting is used.
origins = [62, 70, 78, 86]
validation_rows = []

cpu_train = train["aggregate_cpu_utilization"].to_numpy(dtype=float)

for model_name, forecast_fn in models.items():
    for origin in origins:
        history = cpu_train[:origin]
        actual = cpu_train[origin:origin + HORIZON]

        if len(actual) != HORIZON:
            raise ValueError("Validation fold exceeds training window.")

        prediction = forecast_fn(history, HORIZON)
        errors = prediction - actual

        validation_rows.append({
            "model": model_name,
            "origin_index": origin,
            "horizon": HORIZON,
            "mae": float(np.mean(np.abs(errors))),
            "rmse": float(np.sqrt(np.mean(errors ** 2))),
        })

validation = pd.DataFrame(validation_rows)
validation.to_csv(OUTPUT_DIR / "validation_results.csv", index=False)

summary = (
    validation.groupby("model")
    .agg(
        mean_fold_mae=("mae", "mean"),
        mean_fold_rmse=("rmse", "mean"),
    )
    .sort_values("mean_fold_mae")
)

print("\n=== ROLLING-ORIGIN VALIDATION ===")
print(validation.to_string(index=False))
print("\n=== MODEL SUMMARY (TRAINING DATA ONLY) ===")
print(summary.to_string())

best_model_name = summary.index[0]
best_model = models[best_model_name]

print(f"\nSelected CPU model: {best_model_name}")
print("Selection criterion: lowest mean fold MAE.")

# Refit the selected forecasting rule on the first 110 observations.
cpu_forecast = best_model(cpu_train, HORIZON)
cpu_forecast = np.clip(cpu_forecast, 0.0, 100.0)

# Active VM count is constant in the training window.
# Use persistence for this target.
active_train = train["active_vm_count"].to_numpy(dtype=float)
active_forecast = forecast_persistence(active_train, HORIZON)
active_forecast = np.maximum(active_forecast, 0.0)

predictions = pd.DataFrame({
    "timestamp": future_timestamps,
    "active_vm_count": active_forecast,
    "aggregate_cpu_utilization": cpu_forecast,
})

if list(predictions.columns) != required:
    raise ValueError("Prediction schema is incorrect.")
if len(predictions) != 24:
    raise ValueError("Predictions must contain exactly 24 rows.")
if predictions.isna().any().any():
    raise ValueError("Predictions contain missing values.")
if not np.isfinite(predictions.select_dtypes(include=[np.number])).all().all():
    raise ValueError("Predictions contain non-finite values.")
if not predictions["timestamp"].is_monotonic_increasing:
    raise ValueError("Prediction timestamps are not chronological.")
if predictions["timestamp"].duplicated().any():
    raise ValueError("Duplicate prediction timestamps.")
if not predictions["aggregate_cpu_utilization"].between(0, 100).all():
    raise ValueError("CPU forecasts must be between 0 and 100.")

predictions["active_vm_count"] = (
    predictions["active_vm_count"].round().astype(int)
)
predictions.to_csv("predictions.csv", index=False)

print("\n=== FINAL PREDICTIONS ===")
print(predictions.to_string(index=False))
print("\nSaved reports/validation_results.csv")
print("Saved predictions.csv")
print("Holdout target values were not used for model selection.")
