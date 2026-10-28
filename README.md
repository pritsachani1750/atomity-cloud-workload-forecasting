@'
# Cloud Workload Forecasting

An AI/ML take-home project forecasting aggregate cloud VM CPU utilization using the Microsoft Azure Public Dataset V2 challenge release.

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

Python 3.11 or newer, with `numpy` and `pandas`.

## Setup on Windows PowerShell

Clone this repository and enter its directory. Create and activate a virtual environment:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install numpy pandas matplotlib pytest