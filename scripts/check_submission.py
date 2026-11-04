"""Check a submission file against the competition's sample before you upload it.

    python scripts/check_submission.py submissions/reference.csv [--sample data/raw/sample_submission.csv]

A rejected upload still costs one of the day's submissions. The file is
checked as it was written to disk: the header as text, the number of rows,
the ids in the sample's order, and whether the predictions look like
predictions (a constant column usually means the model was never
loaded, a spread far below the training labels' means it learned little).
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd


def check(path: str, sample_path: str) -> list[str]:
    problems = []
    with open(path) as f:
        header = f.readline().strip()
    if header != "image_id,usability":
        problems.append(f"header is {header!r}, must be exactly 'image_id,usability' (no index column)")
    sub, sample = pd.read_csv(path), pd.read_csv(sample_path)
    if len(sub) != len(sample):
        problems.append(f"{len(sub)} rows, the sample has {len(sample)}")
    if "image_id" in sub and "image_id" in sample:
        if sub["image_id"].duplicated().any():
            problems.append(f"{int(sub['image_id'].duplicated().sum())} duplicated image_id")
        missing = set(sample["image_id"]) - set(sub["image_id"])
        extra = set(sub["image_id"]) - set(sample["image_id"])
        if missing or extra:
            problems.append(f"{len(missing)} ids missing, {len(extra)} ids not in the sample")
        elif not (sub["image_id"].to_numpy() == sample["image_id"].to_numpy()).all():
            problems.append("the ids are not in the sample's order")
    if "usability" in sub:
        u = pd.to_numeric(sub["usability"], errors="coerce")
        if u.isna().any():
            problems.append(f"{int(u.isna().sum())} values are missing or not numbers")
        if ((u < 0) | (u > 1)).any():
            problems.append(f"{int(((u < 0) | (u > 1)).sum())} values outside [0, 1]")
        if u.nunique() == 1:
            problems.append(f"every prediction is {u.iloc[0]}: was the model loaded?")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description="Check a Kaggle submission file")
    ap.add_argument("submission")
    ap.add_argument("--sample", default="data/raw/sample_submission.csv")
    a = ap.parse_args()
    problems = check(a.submission, a.sample)
    u = pd.read_csv(a.submission)["usability"]
    print(f"{a.submission}: {len(u)} rows, predictions mean {u.mean():.3f}, sd {u.std():.3f}, "
          f"min {u.min():.3f}, max {u.max():.3f}, below 0.5: {np.mean(u < 0.5):.1%}")
    for p in problems:
        print("PROBLEM:", p)
    print("OK to upload" if not problems else "DO NOT UPLOAD")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
