from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("challenge_data/aggregate_targets.csv")
validation = pd.read_csv("reports/validation_results.csv")

figures = Path("reports/figures")
figures.mkdir(parents=True, exist_ok=True)

# Figure 1: CPU utilization over time
hours = data["timestamp"] / 3600

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(hours, data["aggregate_cpu_utilization"],
        label="Mean CPU utilization (%)")
ax.set_xlabel("Hours from trace start")
ax.set_ylabel("Mean CPU utilization (%)")
ax.grid(True, alpha=0.3)

ax2 = ax.twinx()
ax2.plot(hours, data["active_vm_count"],
         linestyle="--", label="Active VM count")
ax2.set_ylabel("Active VM count")

lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc="best")
ax.set_title("Aggregate Cloud Workload")
fig.tight_layout()
fig.savefig(figures / "cpu_workload.png", dpi=160)
plt.close(fig)

# Figure 2: validation error comparison
summary = validation.groupby("model")["mae"].mean().sort_values()

fig, ax = plt.subplots(figsize=(8, 5))
summary.plot(kind="bar", ax=ax)
ax.set_title("Mean Rolling-Origin Validation MAE")
ax.set_xlabel("Forecasting model")
ax.set_ylabel("MAE (CPU percentage points)")
ax.tick_params(axis="x", rotation=15)
ax.grid(True, axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(figures / "validation_comparison.png", dpi=160)
plt.close(fig)

print("Charts saved:")
print(figures / "cpu_workload.png")
print(figures / "validation_comparison.png")
print("\nValidation summary:")
print(validation.groupby("model")[["mae", "rmse"]].mean().sort_values("mae"))
