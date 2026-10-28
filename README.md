# Cloud Workload Forecasting

AI/ML take-home project using the fixed `data-v1` release of the Microsoft Azure Public Dataset V2.

## Project results

- 10,000 VMs and 1,340,000 CPU observations.
- 134 aggregate timestamps at five-minute intervals.
- First 110 observations used for model development.
- Final 24 timestamps forecast as a two-hour horizon.
- Moving average selected using rolling-origin validation.

Validation results (mean across four folds):

| Model | MAE | RMSE |
|---|---:|---:|
| Moving average (12 observations) | 0.239952 | 0.265398 |
| Persistence | 0.248082 | 0.276567 |
| Linear trend (48 observations) | 0.381578 | 0.409507 |

The moving-average model achieved approximately 3.3% lower mean validation MAE than persistence. This is a modest improvement and does not establish production performance.

## Requirements

Python 3.11 or newer is recommended by the challenge.

## Setup on Windows PowerShell

Clone this repository and enter its directory. Create and activate a virtual environment:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pip install pandas numpy matplotlib
```

## Dataset

Download `data-v1.tar.gz` from the [official challenge release](https://github.com/atomityhq/forecasting-challenge/releases/tag/data-v1).

Extract it into the repository root so the files are located in `challenge_data/`. The two compressed CSV files and the checksum manifest are required.

Do not substitute a different dataset version. The dataset is intentionally excluded from this submission repository.

## Reproduce the analysis and forecasts

Run these commands from the repository root after installing dependencies and extracting the dataset:

```powershell
python .\analyze_data.py
python .\forecast_pipeline.py
python .\create_report_assets.py
python -m pytest -q
```

The scripts produce:

- `challenge_data/aggregate_targets.csv` — derived aggregate targets.
- `reports/validation_results.csv` — validation-fold metrics.
- `reports/figures/cpu_workload.png` — workload overview.
- `reports/figures/validation_comparison.png` — validation comparison.
- `predictions.csv` — required 24-row forecast file.

`inspect_data.py` provides additional dataset inspection.

## Forecasting methodology

The required targets are:

- `active_vm_count`: distinct VMs with valid readings at each timestamp.
- `aggregate_cpu_utilization`: mean VM-level `avg_cpu` at each timestamp.

CPU forecasts compare persistence, a 12-observation moving average, and a linear trend fitted to the most recent 48 observations. Four chronological validation origins are used within the first 110 observations, with a 24-observation horizon at each origin.

The selected CPU model is the one with the lowest mean validation MAE. The active VM count uses persistence because it is constant in the training window.

The final 24 target values are not used for model selection. Validation folds overlap, so their errors are not independent. The short trace does not establish daily or weekly seasonality.

## Report

See [`reports/final_report.md`](reports/final_report.md) for data quality, workload analysis, validation results, limitations, and operational recommendations.

## Tests

Run:

```powershell
python -m pytest -q
```

## Dataset source

Microsoft Azure Public Dataset V2: https://github.com/Azure/AzurePublicDataset
