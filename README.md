# Cloud Workload Forecasting

An AI/ML take-home project forecasting aggregate cloud VM CPU utilization using the Microsoft Azure Public Dataset V2 challenge release.

## Project Results

- 10,000 VMs and 1,340,000 CPU observations.
- 134 aggregate timestamps at five-minute intervals.
- First 110 observations used for model development.
- Final 24 timestamps forecast as a two-hour horizon.
- Moving average selected using rolling-origin validation.

### Validation Results

Mean metrics across four chronological validation folds:

| Model | MAE | RMSE |
|---|---:|---:|
| Moving average (12 observations) | 0.239952 | 0.265398 |
| Persistence | 0.248082 | 0.276567 |
| Linear trend (48 observations) | 0.381578 | 0.409507 |

The moving-average model achieved approximately 3.3% lower mean validation MAE than persistence. This is a modest improvement and does not establish production performance.

## Requirements

- Python 3.11 or newer
- NumPy
- pandas
- Matplotlib (for report figures)
- pytest (for tests)

## Setup on Windows PowerShell

Clone this repository and navigate to the project directory. Create and activate a virtual environment:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install numpy pandas matplotlib pytest
```

## Dataset

Download `data-v1.tar.gz` from the [official challenge release](https://github.com/atomityhq/forecasting-challenge/releases/tag/data-v1).

Extract the dataset into the repository root so that the required CPU readings are available in the `challenge_data/` directory.

The dataset is intentionally excluded from this submission repository. Do not commit the upstream dataset, credentials, virtual environments, generated caches, or large derived datasets.

## Reproduce Predictions

After installing the dependencies and extracting the dataset, run this single command from the repository root:

```powershell
python .\predict.py --data challenge_data --output predictions.csv
```

This command generates `predictions.csv` with exactly these columns:

```text
timestamp,active_vm_count,aggregate_cpu_utilization
```

For the official challenge release, the output contains exactly 24 predictions corresponding to the final 24 timestamps, in chronological order.

The pipeline derives aggregate targets directly from the supplied CPU readings. It compares forecasting approaches using chronological validation within the historical training data and selects the CPU forecasting model using mean validation MAE.

The final 24 observed target values are not used for model fitting, feature selection, hyperparameter tuning, or model selection.

## Prediction Validation

The generated predictions are checked for:

- Exactly 24 rows.
- Exactly three required columns, in the specified order.
- No missing or non-finite numeric values.
- Chronological, unique timestamps.
- Non-negative `active_vm_count`.
- `aggregate_cpu_utilization` within the range 0 to 100.
- No additional CSV index column.

## Additional Analysis

The repository includes scripts and outputs for additional data analysis:

- `predict.py` — single-command prediction pipeline.
- `analyze_data.py` — data quality checks and aggregate target generation.
- `forecast_pipeline.py` — original forecasting and validation analysis.
- `create_report_assets.py` — generates workload and validation figures.
- `inspect_data.py` — additional dataset inspection.
- `reports/validation_results.csv` — validation-fold metrics.
- `reports/figures/cpu_workload.png` — workload overview.
- `reports/figures/validation_comparison.png` — validation comparison.
- `reports/final_report.md` — final report, findings, limitations, and operational recommendations.

## Forecasting Methodology

The forecasting targets are:

- `active_vm_count`: the number of distinct VMs with valid readings at each timestamp.
- `aggregate_cpu_utilization`: the mean VM-level `avg_cpu` across valid readings at each timestamp.

The CPU forecasting approaches compared are:

1. Persistence — repeats the latest observed value.
2. Moving average — repeats the mean of the most recent 12 observations.
3. Linear trend — fits a linear trend to the most recent 48 observations.

The official release uses four chronological validation origins within the first 110 observations, with a 24-observation forecast horizon for each fold.

The selected CPU model is the model with the lowest mean validation MAE. The active VM count uses persistence because it is constant in the official training window.

Validation folds overlap, so their errors are not independent. The short trace does not establish daily or weekly seasonality, and the modest improvement over persistence should not be interpreted as evidence of production readiness.

## Run Tests

Run the automated tests from the repository root:

```powershell
python -m pytest -q
```

## Dataset Source

Microsoft Azure Public Dataset V2:

https://github.com/Azure/AzurePublicDataset