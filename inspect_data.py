
import gzip
import pandas as pd
from pathlib import Path

data_dir = Path("challenge_data")

for filename in [
    "vm_cpu_readings.csv.gz",
    "vm_metadata.csv.gz",
]:
    path = data_dir / filename

    print(f"\n{'=' * 60}")
    print(f"File: {filename}")

    with gzip.open(path, "rt") as file:
        df = pd.read_csv(file, nrows=5)

    print("Columns:", df.columns.tolist())
    print(df.head())
