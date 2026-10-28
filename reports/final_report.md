# Cloud Workload Forecasting and Analysis

## 1. Executive Summary

This project analyzes a curated subset of Microsoft Azure Public Dataset V2 and forecasts aggregate virtual-machine (VM) CPU utilization and active VM count over a two-hour horizon.

The dataset contains 1,340,000 CPU observations from 10,000 VMs at five-minute intervals. After data validation, 134 aggregate timestamps were constructed. The first 110 observations were used for model development, while the final 24 timestamps were reserved for forecasting.

Three CPU forecasting methods were compared using rolling-origin validation: persistence, a 12-observation moving average, and a linear trend fitted to the most recent 48 observations. The moving average achieved the lowest mean validation MAE of 0.239952 CPU percentage points, compared with 0.248082 for persistence and 0.381578 for the linear trend.

The moving average was selected for the final CPU forecasts. The active VM count was forecast as 10,000, consistent with the constant count in the training observations. The modest improvement over persistence suggests that more extensive validation would be needed before claiming a meaningful operational advantage.

## 2. Data Understanding and Preprocessing

The challenge dataset was supplied as the fixed `data-v1` release. SHA-256 checksum verification succeeded for the CPU readings, VM metadata, and data card.

The CPU file contains the fields `timestamp`, `vm_id`, `min_cpu`, `max_cpu`, and `avg_cpu`. The metadata file contains 10,000 VM records and 11 fields describing VM identifiers, lifetime timestamps, workload categories, and resource buckets.

The CPU dataset contains 1,340,000 rows and five columns. No missing values or duplicate `(timestamp, vm_id)` pairs were found. All CPU readings were within the interval [0, 100]. All 10,000 reading VMs had corresponding metadata records.

The timestamps range from 0 to 39,900 seconds in increments of 300 seconds, giving 134 regularly spaced observations over approximately 11.1 hours.

Two aggregate targets were constructed:

- `active_vm_count`: distinct VM identifiers with valid CPU readings at each timestamp.
- `aggregate_cpu_utilization`: arithmetic mean of VM-level `avg_cpu` readings at each timestamp.

All 1,340,000 readings were valid under the implemented checks, so no observations were excluded. The resulting active VM count was 10,000 at every timestamp.

## 3. Exploratory Workload Analysis

The mean of all VM-level average CPU readings was approximately 9.388%. Across aggregate timestamps, mean CPU utilization ranged from approximately 8.984% to 9.912%, with a standard deviation of approximately 0.195 percentage points.

At the VM-observation level, average CPU utilization had a median of approximately 5.105% and a 75th percentile of approximately 9.617%. These values indicate that the distribution of individual VM CPU readings differs from the distribution of the aggregate mean.

The active VM count was constant across the supplied trace. Therefore, this dataset does not provide evidence of changing aggregate VM count during the observed period. This does not establish that VM churn is unimportant in other workloads.

Figure 1, `figures/cpu_workload.png`, shows aggregate CPU utilization and active VM count across the observed trace. Further analysis of workload categories, resource buckets, and VM lifetimes could provide useful operational context, but is not required to reproduce the current forecasts.

## 4. Forecasting Approach

The aggregate series was split chronologically. Observations 1–110 formed the model-development window; observations 111–134 formed the final 24-step, two-hour forecast horizon.

Three methods were evaluated:

1. **Persistence:** repeat the most recent observed value for all forecast steps.
2. **Moving average:** repeat the mean of the most recent 12 observed values.
3. **Linear trend:** fit a least-squares linear trend to the most recent 48 observations and extrapolate it across the horizon.

The active VM count was forecast using persistence because it remained constant at 10,000 throughout the training window.

## 5. Validation Methodology

The CPU forecasting methods were evaluated using four chronological forecast origins at training indices 62, 70, 78, and 86. Each origin forecast the following 24 observations. All validation observations were within the first 110 aggregate observations.

This is a time-aware validation approach and avoids random shuffling. However, the validation windows overlap, so their errors are not independent. The results should be treated as a preliminary comparison rather than a definitive estimate of production performance.

The final 24 target observations were not used to fit, tune, or select the model. The model was selected using mean validation MAE.

## 6. Results and Error Analysis

The following are the measured mean metrics across the four validation folds:

| Model | Mean MAE | Mean RMSE |
|---|---:|---:|
| Moving average (12 observations) | 0.239952 | 0.265398 |
| Persistence | 0.248082 | 0.276567 |
| Linear trend (48 observations) | 0.381578 | 0.409507 |

MAE and RMSE are measured in CPU utilization percentage points. The moving average reduced mean MAE by approximately 3.3% relative to persistence. The linear trend performed worse than both baselines.

The moving average was selected because it had the lowest mean validation MAE. Its modest advantage may not be stable across other time periods. Fold-level errors varied, illustrating that model performance depends on the forecast origin.

Figure 2, `figures/validation_comparison.png`, compares mean validation MAE. The current evaluation does not establish peak-period performance separately, and no final holdout accuracy is reported because the holdout targets were excluded from model selection and evaluation.

## 7. Operational Recommendation

Use this forecast as a short-horizon aggregate demand signal, not as the sole trigger for automatic capacity changes.

A production system should combine aggregate forecasts with per-VM CPU distributions, workload criticality, available capacity, and forecast uncertainty. A conservative policy would alert operators when the forecast plus a validated safety margin approaches a defined capacity limit, while requiring additional evidence before scaling resources automatically.

The trade-off is between under-provisioning, which can degrade service performance, and over-provisioning, which can increase infrastructure cost. The appropriate safety margin and thresholds must be calibrated using longer traces, service-level objectives, and the actual capacity model. The current aggregate mean alone is insufficient to derive a defensible numeric scaling threshold.

## 8. Limitations

- The trace spans only approximately 11.1 hours, so daily and weekly seasonality cannot be established.
- Active VM count is constant in this curated subset, limiting its usefulness for evaluating VM churn forecasting.
- The validation folds overlap and provide only a small number of evaluation origins.
- The moving average improves mean validation MAE over persistence by only approximately 3.3%.
- Peak-load error, prediction intervals, and segment-level model performance have not yet been quantified.
- Aggregate mean CPU can conceal overloaded individual VMs.

## 9. Reproducibility and Deliverables

The project generates `challenge_data/aggregate_targets.csv` from the supplied compressed readings, creates rolling-origin validation results in `reports/validation_results.csv`, and writes the required `predictions.csv`.

The prediction file contains exactly 24 rows with the required columns:
`timestamp`, `active_vm_count`, and `aggregate_cpu_utilization`.

To reproduce the current analysis and forecasts from the repository root in the configured Python environment, run:

```powershell
python .\analyze_data.py
python .\forecast_pipeline.py
python .\create_report_assets.py
python -m pytest -q
```

The dataset release should be downloaded separately according to the challenge instructions. The generated report and charts summarize the current implementation; additional validation and peak-period analysis would strengthen a future version.
