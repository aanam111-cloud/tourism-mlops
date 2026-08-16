#!/usr/bin/env python3
"""
Data Registration Script
Reads the tourism CSV from the repo, checks expected columns, and prints a short summary.
Dataset stays inside the GitHub repo – no external store needed.
"""

import sys
from pathlib import Path
import pandas as pd

EXPECTED_COLUMNS = [
    "CustomerID",
    "ProdTaken",
    "Age",
    "TypeofContact",
    "CityTier",
    "DurationOfPitch",
    "Occupation",
    "Gender",
    "NumberOfPersonVisiting",
    "NumberOfFollowups",
    "ProductPitched",
    "PreferredPropertyStar",
    "MaritalStatus",
    "NumberOfTrips",
    "Passport",
    "PitchSatisfactionScore",
    "OwnCar",
    "NumberOfChildrenVisiting",
    "Designation",
    "MonthlyIncome",
]


def register_dataset(csv_path: Path) -> bool:
    if not csv_path.exists():
        print(f"ERROR: Dataset not found at {csv_path}")
        return False

    print("=" * 60)
    print("DATA REGISTRATION – Tourism Dataset")
    print("=" * 60)
    print(f"Source (inside repo): {csv_path}")

    df = pd.read_csv(csv_path)

    # Drop accidental index column if present
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
        print("Note: dropped 'Unnamed: 0' (index column).")

    # Column validation
    actual = set(df.columns)
    expected = set(EXPECTED_COLUMNS)
    missing = expected - actual
    extra = actual - expected

    print("\nColumn validation:")
    if not missing and not extra:
        print("  ✓ All expected columns are present.")
    else:
        if missing:
            print(f"  ✗ Missing columns : {sorted(missing)}")
        if extra:
            print(f"  ✗ Unexpected columns: {sorted(extra)}")
        return False

    # Short summary
    print("\nDataset summary:")
    print(f"  Rows          : {len(df):,}")
    print(f"  Columns       : {df.shape[1]}")
    print(f"  Memory usage  : {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    if "ProdTaken" in df.columns:
        counts = df["ProdTaken"].value_counts().sort_index()
        print(f"\n  Target (ProdTaken):")
        print(f"    0 (No)  : {counts.get(0, 0):,}")
        print(f"    1 (Yes) : {counts.get(1, 0):,}")
        print(f"    Positive rate: {df['ProdTaken'].mean():.1%}")

    n_missing = df.isnull().sum().sum()
    print(f"\n  Missing values: {n_missing}")

    print("\n  Key dtypes:")
    for col in ["Age", "MonthlyIncome", "TypeofContact", "Occupation", "Gender"]:
        if col in df.columns:
            print(f"    {col:20s} → {df[col].dtype}")

    print("\n" + "=" * 60)
    print("Registration successful – dataset is ready for the pipeline.")
    print("=" * 60)
    return True


if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    default_csv = project_root / "data" / "tourism.csv"

    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else default_csv
    ok = register_dataset(csv_path)
    sys.exit(0 if ok else 1)
