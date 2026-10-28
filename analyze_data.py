import pandas as pd
import numpy as np
from pathlib import Path

data_dir = Path("challenge_data")
cpu_path = data_dir / "vm_cpu_readings.csv.gz"
metadata_path = data_dir / "vm_metadata.csv.gz"

print("Loading CPU readings...")
cpu = pd.read_csv(cpu_path)
metadata = pd.read_csv(metadata_path)

print("\n=== CPU DATA QUALITY ===")
print("Shape:", cpu.shape)
print("\nMissing values:\n", cpu.isna().sum())
print("\nDuplicate timestamp/VM pairs:",
      cpu.duplicated(["timestamp", "vm_id"]).sum())
print("Unique VMs:", cpu["vm_id"].nunique())
print("Unique timestamps:", cpu["timestamp"].nunique())

timestamps = np.sort(cpu["timestamp"].dropna().unique())
print("First timestamps:", timestamps[:5])
print("Last timestamps:", timestamps[-5:])

if len(timestamps) > 1:
    diffs = np.diff(timestamps)
    print("Timestamp increments:", pd.Series(diffs).value_counts().to_dict())
    print("Five-minute cadence (300 seconds):",
          bool(np.all(diffs == 300)))

print("\n=== CPU STATISTICS ===")
print(cpu[["min_cpu", "max_cpu", "avg_cpu"]].describe())
for col in ["min_cpu", "max_cpu", "avg_cpu"]:
    values = pd.to_numeric(cpu[col], errors="coerce")
    print(f"{col}: outside [0, 100] =", int(
        ((values < 0) | (values > 100)).sum()
    ))

print("\n=== METADATA QUALITY ===")
print("Shape:", metadata.shape)
print("Unique metadata VMs:", metadata["vm_id"].nunique())
print("Missing values:\n", metadata.isna().sum())
print("Reading VMs without metadata:",
      len(set(cpu["vm_id"].dropna()) - set(metadata["vm_id"].dropna())))

# A valid reading has a VM identifier and finite, in-range avg_cpu.
avg_cpu = pd.to_numeric(cpu["avg_cpu"], errors="coerce")
valid = (
    cpu["vm_id"].notna()
    & avg_cpu.notna()
    & np.isfinite(avg_cpu)
    & avg_cpu.between(0, 100)
)
valid_cpu = cpu.loc[valid, ["timestamp", "vm_id"]].copy()
valid_cpu["avg_cpu"] = avg_cpu.loc[valid]

print("\n=== AGGREGATE TARGETS ===")
print("Valid CPU observations:", len(valid_cpu))
print("Excluded invalid CPU observations:", int((~valid).sum()))

duplicates = valid_cpu.duplicated(["timestamp", "vm_id"]).sum()
if duplicates:
    raise ValueError(
        f"Found {duplicates} duplicate valid timestamp/VM pairs. "
        "Review them before aggregation; no rows were silently discarded."
    )

targets = (
    valid_cpu.groupby("timestamp", as_index=False)
    .agg(
        active_vm_count=("vm_id", "nunique"),
        aggregate_cpu_utilization=("avg_cpu", "mean"),
    )
    .sort_values("timestamp")
    .reset_index(drop=True)
)

if len(targets) != 134:
    raise ValueError(
        f"Expected 134 aggregate timestamps, found {len(targets)}. "
        "Review the timestamp coverage before proceeding."
    )

diffs = targets["timestamp"].diff().dropna()
if not diffs.eq(300).all():
    raise ValueError(
        "Aggregate timestamps are not all 300 seconds apart. "
        "Investigate gaps or irregularities before forecasting."
    )

train = targets.iloc[:110].copy()
holdout = targets.iloc[110:].copy()

targets.to_csv(data_dir / "aggregate_targets.csv", index=False)

print("Aggregate timestamps:", len(targets))
print("Training observations:", len(train))
print("Holdout observations:", len(holdout))
print("Training timestamp range:",
      train["timestamp"].iloc[0], "to", train["timestamp"].iloc[-1])
print("Holdout timestamp range:",
      holdout["timestamp"].iloc[0], "to", holdout["timestamp"].iloc[-1])
print("\nTarget summary:\n", targets[
    ["active_vm_count", "aggregate_cpu_utilization"]
].describe())
print("\nSaved: challenge_data/aggregate_targets.csv")
print("Data analysis completed.")
