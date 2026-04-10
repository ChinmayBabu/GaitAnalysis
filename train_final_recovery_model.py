# ============================================================
# FINAL RECOVERY PROGRESSION MODEL
# Predict continuous recovery_score (longitudinal recovery)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ------------------------------------------------------------
# 1. LOAD ENGINEERED DATA
# ------------------------------------------------------------

df = pd.read_csv("features_engineered.csv")

print("Dataset Loaded")
print("Shape:", df.shape)

# ------------------------------------------------------------
# 2. FEATURE SELECTION
# (Exclude velocity to avoid circular dependency)
# ------------------------------------------------------------

features = [
    "FGI",
    "StabilityIndex",
    "MES",
    "PCI",
    "PRI",
    "mean_knee_angle",
    "mean_trunk_angle",
    "MonoSP_UGS",
    "BiSP_UGS"
]

features = [f for f in features if f in df.columns]

target = "recovery_score"

if target not in df.columns:
    raise ValueError("recovery_score not found.")

print("\nFeatures Used:")
print(features)
print("Target:", target)

# ------------------------------------------------------------
# 3. PREPARE DATA
# ------------------------------------------------------------

X = df[features]
y = df[target]
groups = df["subject_id"]

# ------------------------------------------------------------
# 4. GROUP-BASED TRAIN-TEST SPLIT
# ------------------------------------------------------------

gss = GroupShuffleSplit(test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# ------------------------------------------------------------
# 5. TRAIN MODEL (Gradient Boosting)
# ------------------------------------------------------------

model = GradientBoostingRegressor(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=4,
    random_state=42
)

model.fit(X_train, y_train)

print("\nModel Training Completed.")

# ------------------------------------------------------------
# 6. EVALUATION
# ------------------------------------------------------------

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n==============================")
print("Final Recovery Model Results")
print("==============================")
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")

# ------------------------------------------------------------
# 7. ACTUAL VS PREDICTED
# ------------------------------------------------------------

plt.figure()
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Recovery Score")
plt.ylabel("Predicted Recovery Score")
plt.title("Actual vs Predicted Recovery Score")
plt.tight_layout()
plt.savefig("final_recovery_actual_vs_predicted.png")
plt.show()

# ------------------------------------------------------------
# 8. FEATURE IMPORTANCE
# ------------------------------------------------------------

importances = pd.Series(
    model.feature_importances_,
    index=features
).sort_values(ascending=True)

plt.figure()
importances.plot(kind="barh")
plt.title("Final Model Feature Importance")
plt.tight_layout()
plt.savefig("final_recovery_feature_importance.png")
plt.show()

print("\nFeature Importance Ranking:")
print(importances.sort_values(ascending=False))

# ------------------------------------------------------------
# 9. SAVE MODEL
# ------------------------------------------------------------

joblib.dump(model, "final_recovery_model.pkl")

print("\nModel saved as final_recovery_model.pkl")
print("Final recovery progression model completed successfully.")
