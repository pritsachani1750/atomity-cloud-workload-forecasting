import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_OUTPUT = [
    "timestamp",
    "active_vm_count",
    "aggregate_cpu_utilization",
]
HORIZON = 24
MOVING_AVERAGE_WINDOW = 12


def read_csv_flexible(data_dir, stem):
    """Read either a compressed or an uncompressed CSV."""
    for filename in (f"{stem}.csv.gz", f"{stem}.csv"):
        path = data_dir / filename
        if path.exists():
            return pd.read_csv(path)
    raise FileNotFoundError(
        f"Could not find {stem}.csv.gz or {stem}.csv in {data_dir}"
    )


def aggregate_targets(data_dir):
    """Create aggregate targets directly from the supplied CPU readings."""
    cpu = read_csv_flexible(data_dir, "vm_cpu_readings")

    required = {"timestamp", "vm_id", "avg_cpu"}
    missing = required - set(cpu.columns)
    if missing:
        raise ValueError(f"CPU data is missing columns: {sorted(missing)}")

    cpu["timestamp"] = pd.to_numeric(cpu["timestamp"], errors="coerce")
    cpu["avg_cpu"] = pd.to_numeric(cpu["avg_cpu"], errors="coerce")

    valid = (
        cpu["timestamp"].notna()
        & cpu["vm_id"].notna()
        & np.isfinite(cpu["avg_cpu"])
        & cpu["avg_cpu"].between(0, 100)
    )
    cpu = cpu.loc[valid, ["timestamp", "vm_id", "avg_cpu"]].copy()

    if cpu.empty:
        raise ValueError("No valid CPU readings were found.")

    if cpu.duplicated(["timestamp", "vm_id"]).any():
        raise ValueError("Duplicate timestamp/VM readings found.")

    targets = (
        cpu.groupby("timestamp", as_index=False)
        .agg(
            active_vm_count=("vm_id", "nunique"),
            aggregate_cpu_utilization=("avg_cpu", "mean"),
        )
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    if targets["timestamp"].duplicated().any():
        raise ValueError("Duplicate aggregate timestamps found.")

    return targets


def persistence(history, horizon):
    return np.repeat(float(history[-1]), horizon)


def moving_average(history, horizon):
    recent = history[-MOVING_AVERAGE_WINDOW:]
    return np.repeat(float(np.mean(recent)), horizon)


def linear_trend(history, horizon):
    recent = np.asarray(history[-48:], dtype=float)

    if len(recent) < 2:
        return persistence(recent, horizon)

    x = np.arange(len(recent), dtype=float)
    future_x = np.arange(
        len(recent), len(recent) + horizon, dtype=float
    )
    slope, intercept = np.polyfit(x, recent, 1)
    return intercept + slope * future_x


MODELS = {
    "persistence": persistence,
    "moving_average_12": moving_average,
    "linear_trend_48": linear_trend,
}


def select_model(history):
    """Use rolling-origin validation when enough data is available."""
    n = len(history)
    rows = []

    # Preserve the four original validation folds for the official release.
    if n >= 110:
        origins = [62, 70, 78, 86]
    else:
        # Smaller datasets use only folds that fit completely in history.
        first_origin = max(2, min(12, n // 2))
        last_origin = n - HORIZON

        if last_origin < first_origin:
            return "moving_average_12", pd.DataFrame(
                columns=["model", "origin_index", "horizon", "mae", "rmse"]
            )

        origins = np.unique(
            np.linspace(
                first_origin,
                last_origin,
                num=min(4, last_origin - first_origin + 1),
                dtype=int,
            )
        ).tolist()

    for name, forecast_fn in MODELS.items():
        for origin in origins:
            fold_history = history[:origin]
            actual = history[origin:origin + HORIZON]

            if len(actual) != HORIZON or len(fold_history) == 0:
                continue

            predicted = forecast_fn(fold_history, HORIZON)
            errors = predicted - actual

            rows.append({
                "model": name,
                "origin_index": int(origin),
                "horizon": HORIZON,
                "mae": float(np.mean(np.abs(errors))),
                "rmse": float(np.sqrt(np.mean(errors ** 2))),
            })

    results = pd.DataFrame(rows)

    if results.empty:
        return "moving_average_12", results

    summary = results.groupby("model")["mae"].mean().sort_values()
    return str(summary.index[0]), results


def make_future_timestamps(targets):
    """Use the official holdout timestamps, or extend tiny datasets."""
    timestamps = targets["timestamp"].to_numpy()

    if len(targets) > HORIZON:
        # Official release: these are the final 24 timestamps.
        return timestamps[-HORIZON:], targets.iloc[:-HORIZON].copy()

    # Tiny smoke dataset: use all available observations as history
    # and predict the next 24 timestamps rather than fitting to them.
    if len(timestamps) >= 2:
        differences = np.diff(timestamps.astype(float))
        positive_differences = differences[differences > 0]
        cadence = (
            float(np.median(positive_differences))
            if len(positive_differences)
            else 300.0
        )
    else:
        cadence = 300.0

    future = timestamps[-1] + cadence * np.arange(1, HORIZON + 1)
    return future, targets.copy()


def main():
    parser = argparse.ArgumentParser(
        description="Forecast aggregate cloud VM CPU workload."
    )
    parser.add_argument("--data", type=Path, default=Path("challenge_data"))
    parser.add_argument("--output", type=Path, default=Path("predictions.csv"))
    args = parser.parse_args()

    targets = aggregate_targets(args.data)

    if not targets["timestamp"].is_monotonic_increasing:
        raise ValueError("Aggregate timestamps must be chronological.")

    future_timestamps, training = make_future_timestamps(targets)

    cpu_history = training["aggregate_cpu_utilization"].to_numpy(dtype=float)
    active_history = training["active_vm_count"].to_numpy(dtype=float)

    if len(cpu_history) == 0:
        raise ValueError("At least one historical observation is required.")

    selected_name, validation = select_model(cpu_history)
    cpu_forecast = MODELS[selected_name](cpu_history, HORIZON)
    cpu_forecast = np.clip(cpu_forecast, 0.0, 100.0)

    # Active VM count uses persistence, consistent with the original model.
    active_forecast = np.maximum(
        persistence(active_history, HORIZON), 0.0
    )

    predictions = pd.DataFrame({
        "timestamp": future_timestamps,
        "active_vm_count": np.rint(active_forecast).astype(int),
        "aggregate_cpu_utilization": cpu_forecast,
    })

    if list(predictions.columns) != REQUIRED_OUTPUT:
        raise ValueError("Incorrect prediction columns.")
    if len(predictions) != HORIZON:
        raise ValueError(f"Expected exactly {HORIZON} predictions.")
    if predictions.isna().any().any():
        raise ValueError("Predictions contain missing values.")
    if not np.isfinite(
        predictions.select_dtypes(include=[np.number]).to_numpy()
    ).all():
        raise ValueError("Predictions contain non-finite values.")
    if not predictions["timestamp"].is_monotonic_increasing:
        raise ValueError("Prediction timestamps are not chronological.")
    if predictions["timestamp"].duplicated().any():
        raise ValueError("Duplicate prediction timestamps.")
    if (predictions["active_vm_count"] < 0).any():
        raise ValueError("Active VM counts must be non-negative.")
    if not predictions["aggregate_cpu_utilization"].between(0, 100).all():
        raise ValueError("CPU forecasts must be within [0, 100].")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output, index=False)

    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    if not validation.empty:
        validation.to_csv(reports_dir / "validation_results.csv", index=False)

    print(f"Training observations: {len(training)}")
    print(f"Selected CPU model: {selected_name}")
    print(f"Forecast rows: {len(predictions)}")
    print(f"Saved predictions to: {args.output}")
    print(predictions.to_string(index=False))


if __name__ == "__main__":
    main()
