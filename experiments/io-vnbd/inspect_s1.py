import os
import pandas as pd
import numpy as np


FILES = [
    "data/io-vnbd/raw/S1/S-S1.csv",
    "data/io-vnbd/raw/S1/V-S1.csv"
]


def inspect_file(path):
    print("\n" + "=" * 70)
    print("FILE:", path)
    print("=" * 70)

    if not os.path.exists(path):
        print("ERROR: File not found")
        return

    # IO-VNBD file encoding
    df = pd.read_csv(
        path,
        encoding="latin1"
    )

    print("\nShape:")
    print(df.shape)

    print("\nColumns:")
    for column in df.columns:
        print(" -", column)

    print("\nFirst 3 rows:")
    print(df.head(3).to_string(index=False))

    # --------------------------------------------------
    # Find likely time columns
    # --------------------------------------------------

    time_candidates = []

    for column in df.columns:
        name = column.lower()

        if (
            name == "time"
            or "timestamp" in name
            or "time" in name
        ):
            time_candidates.append(column)

    print("\nPossible time columns:")
    print(time_candidates)

    # --------------------------------------------------
    # Analyze time
    # --------------------------------------------------

    for column in time_candidates:

        values = pd.to_numeric(
            df[column],
            errors="coerce"
        ).dropna().values

        if len(values) < 2:
            continue

        differences = np.diff(values)

        median_dt = np.median(differences)
        mean_dt = np.mean(differences)

        print("\nTime analysis for:", column)

        print("First value :", values[0])
        print("Last value  :", values[-1])
        print("Median dt   :", median_dt, "seconds")
        print("Mean dt     :", mean_dt, "seconds")

        if median_dt > 0:
            print(
                "Sampling rate:",
                1 / median_dt,
                "Hz"
            )

    # --------------------------------------------------
    # Missing values
    # --------------------------------------------------

    missing = df.isna().sum()

    missing_columns = missing[missing > 0]

    print("\nMissing values:")

    if len(missing_columns) == 0:
        print(" None")
    else:
        print(missing_columns.to_string())


# ======================================================
# Inspect both files
# ======================================================

for file_path in FILES:
    inspect_file(file_path)

print("\n")
print("=" * 70)
print("IO-VNBD S1 INSPECTION COMPLETE")
print("=" * 70)