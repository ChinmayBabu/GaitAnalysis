"""Train a non-leaky ExtraTreesRegressor for velocity prediction.

This script predicts `Velocity_UGS_measured` using subject-grouped splits,
encoded session information, and raw biomechanical / anthropometric features.
It avoids the velocity-derived composite features (`FGI`, `MES`, `PRI`) so the
evaluation stays honest.
"""

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DATA_PATH = "features_engineered.csv"
TARGET = "Velocity_UGS_measured"
MODEL_PATH = "velocity_extra_trees_regressor.pkl"
PLOT_PATH = "velocity_extra_trees_actual_vs_predicted.png"


def build_feature_set(df: pd.DataFrame) -> list[str]:
    """Return a non-leaky feature set for velocity prediction."""

    candidate_features = [
        "session",
        "Step_UGS_measured",
        "Stride_UGS_measured",
        "Cadence_UGS_measured",
        "MonoSP_UGS",
        "BiSP_UGS",
        "Sex",
        "Age",
        "PA_level",
        "Height",
        "Weight",
        "BMI",
        "WaistC",
        "HipC",
        "NeckC",
        "Percentage_fat_mass",
        "Lean_mass",
        "HR_Final",
        "StabilityIndex",
        "PostureProxy",
        "JointHealthProxy",
        "PCI",
    ]

    return [col for col in candidate_features if col in df.columns]


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    if TARGET not in df.columns:
        raise ValueError(f"{TARGET} not found in {DATA_PATH}")
    if "subject_id" not in df.columns:
        raise ValueError("subject_id not found in dataset")

    features = build_feature_set(df)
    X = df[features].copy()
    y = df[TARGET]
    groups = df["subject_id"]

    categorical_features = [c for c in ["session", "Sex", "PA_level"] if c in X.columns]
    numeric_features = [c for c in X.columns if c not in categorical_features]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
            ("num", SimpleImputer(strategy="median"), numeric_features),
        ],
        verbose_feature_names_out=False,
    )

    model = Pipeline(
        steps=[
            ("prep", preprocessor),
            (
                "model",
                ExtraTreesRegressor(
                    n_estimators=500,
                    random_state=42,
                    n_jobs=1,
                ),
            ),
        ]
    )

    print("Dataset shape:", df.shape)
    print("Target:", TARGET)
    print("Features used:")
    print(features)

    # Grouped cross-validation gives a better estimate than a single split.
    cv = GroupKFold(n_splits=5)
    cv_scores = cross_validate(
        model,
        X,
        y,
        groups=groups,
        cv=cv,
        scoring={
            "mae": "neg_mean_absolute_error",
            "rmse": "neg_root_mean_squared_error",
            "r2": "r2",
        },
        n_jobs=1,
    )

    print("\n5-Fold Grouped CV Metrics")
    print("-------------------------")
    print(f"MAE  : {-cv_scores['test_mae'].mean():.4f} ± {cv_scores['test_mae'].std():.4f}")
    print(f"RMSE : {-cv_scores['test_rmse'].mean():.4f} ± {cv_scores['test_rmse'].std():.4f}")
    print(f"R^2  : {cv_scores['test_r2'].mean():.4f} ± {cv_scores['test_r2'].std():.4f}")

    # Final holdout split for a simple actual-vs-predicted plot.
    gss = GroupShuffleSplit(test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("\nHoldout Metrics")
    print("---------------")
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R^2  : {r2:.4f}")

    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.8)
    min_v = min(y_test.min(), y_pred.min())
    max_v = max(y_test.max(), y_pred.max())
    plt.plot([min_v, max_v], [min_v, max_v], linestyle="--", color="black", linewidth=1)
    plt.xlabel("Actual Velocity")
    plt.ylabel("Predicted Velocity")
    plt.title("ExtraTrees: Actual vs Predicted Velocity")
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=200)
    plt.close()

    importances = pd.Series(model.named_steps["model"].feature_importances_)
    encoded_names = model.named_steps["prep"].get_feature_names_out()
    importances.index = encoded_names
    importances = importances.sort_values(ascending=False)

    print("\nTop Feature Importances")
    print("-----------------------")
    print(importances.head(10).to_string())

    joblib.dump(model, MODEL_PATH)
    print(f"\nSaved model to {MODEL_PATH}")
    print(f"Saved plot to {PLOT_PATH}")


if __name__ == "__main__":
    main()
