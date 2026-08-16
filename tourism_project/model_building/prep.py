#!/usr/bin/env python3
"""
Data Preparation Script
- Loads the dataset from the repository data folder
- Removes unnecessary columns
- Splits into train/test sets and saves them locally as CSV files
"""

import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Remove unnecessary columns and fix basic data issues."""
    df = df.copy()

    # Drop ID / index columns not useful for modelling
    drop_cols = [c for c in ["Unnamed: 0", "CustomerID"] if c in df.columns]
    df = df.drop(columns=drop_cols)

    # Fix Gender typo
    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})

    # Standardise categorical strings
    cat_cols = [
        "TypeofContact", "Occupation", "Gender",
        "ProductPitched", "MaritalStatus", "Designation"
    ]
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # Fill any remaining missing values
    for col in df.columns:
        if df[col].isnull().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode()[0])

    return df


def prepare_and_split(
    data_path: Path,
    output_dir: Path,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Load → clean → stratified split → save train/test CSVs."""
    print("=" * 60)
    print("DATA PREPARATION")
    print("=" * 60)

    print(f"Loading: {data_path}")
    df = pd.read_csv(data_path)
    print(f"Original shape: {df.shape}")

    df_clean = clean_data(df)
    print(f"Shape after cleaning: {df_clean.shape}")
    print(f"Columns kept: {df_clean.columns.tolist()}")

    target = "ProdTaken"
    X = df_clean.drop(columns=[target])
    y = df_clean[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # Re-attach target so downstream jobs can load a single CSV
    train_df = X_train.copy()
    train_df[target] = y_train
    test_df = X_test.copy()
    test_df[target] = y_test

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.csv"
    test_path = output_dir / "test.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"\nTrain set : {train_df.shape} → {train_path}")
    print(f"Test set  : {test_df.shape} → {test_path}")
    print(f"Train positive rate: {y_train.mean():.2%}")
    print(f"Test  positive rate: {y_test.mean():.2%}")

    print("\n" + "=" * 60)
    print("DATA PREPARATION COMPLETED")
    print("=" * 60)

    return train_path, test_path


if __name__ == "__main__":
    # Default paths relative to this script
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    data_path = project_root / "data" / "tourism.csv"
    output_dir = project_root / "data" / "processed"

    if len(sys.argv) > 1:
        data_path = Path(sys.argv[1])
    if len(sys.argv) > 2:
        output_dir = Path(sys.argv[2])

    prepare_and_split(data_path, output_dir)
