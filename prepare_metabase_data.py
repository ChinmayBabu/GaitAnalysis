"""Prepare Metabase-ready data exports for gait recovery analysis.

This script creates dashboard-friendly CSV exports and summary tables that can
be loaded into Metabase or another BI tool.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


DATA_PATH = Path("features_engineered.csv")
OUT_DIR = Path("metabase_exports")


def build_subject_summary(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        "recovery_score",
        "Velocity_UGS_measured",
        "FGI",
        "StabilityIndex",
        "MES",
        "PCI",
        "PRI",
        "Step_UGS_measured",
        "Stride_UGS_measured",
        "Cadence_UGS_measured",
        "MonoSP_UGS",
        "BiSP_UGS",
        "Age",
        "BMI",
        "HR_Final",
    ]
    numeric_cols = [c for c in numeric_cols if c in df.columns]

    summary = (
        df.groupby("subject_id")
        .agg(
            session_count=("session", "nunique"),
            recovery_type=("recovery_type", "first") if "recovery_type" in df.columns else ("subject_id", "first"),
            **{f"avg_{col}": (col, "mean") for col in numeric_cols},
            **{f"min_{col}": (col, "min") for col in ["Velocity_UGS_measured"] if col in df.columns},
            **{f"max_{col}": (col, "max") for col in ["Velocity_UGS_measured"] if col in df.columns},
        )
        .reset_index()
    )
    return summary


def build_session_summary(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["session"]
    if "recovery_phase" in df.columns:
        group_cols.append("recovery_phase")
    elif "recovery_stage" in df.columns:
        group_cols.append("recovery_stage")

    numeric_cols = [
        "recovery_score",
        "Velocity_UGS_measured",
        "FGI",
        "StabilityIndex",
        "MES",
        "PCI",
        "PRI",
        "Step_UGS_measured",
        "Stride_UGS_measured",
        "Cadence_UGS_measured",
    ]
    numeric_cols = [c for c in numeric_cols if c in df.columns]

    summary = (
        df.groupby(group_cols)
        .agg(**{f"mean_{col}": (col, "mean") for col in numeric_cols})
        .reset_index()
    )
    return summary


def build_feature_summary(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df.select_dtypes(include="number")
    summary = pd.DataFrame(
        {
            "feature": numeric.columns,
            "count": numeric.count().values,
            "mean": numeric.mean().values,
            "std": numeric.std().values,
            "min": numeric.min().values,
            "q25": numeric.quantile(0.25).values,
            "median": numeric.median().values,
            "q75": numeric.quantile(0.75).values,
            "max": numeric.max().values,
        }
    )
    return summary.sort_values("feature").reset_index(drop=True)


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing data file: {DATA_PATH}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    subject_summary = build_subject_summary(df)
    session_summary = build_session_summary(df)
    feature_summary = build_feature_summary(df)

    # Export CSVs for easy review / BI tools.
    df.to_csv(OUT_DIR / "gait_observations.csv", index=False)
    subject_summary.to_csv(OUT_DIR / "subject_summary.csv", index=False)
    session_summary.to_csv(OUT_DIR / "session_summary.csv", index=False)
    feature_summary.to_csv(OUT_DIR / "feature_summary.csv", index=False)

    print(f"Created Metabase exports in: {OUT_DIR}")
    print("Files written:")
    print("- gait_observations.csv")
    print("- subject_summary.csv")
    print("- session_summary.csv")
    print("- feature_summary.csv")


if __name__ == "__main__":
    main()
