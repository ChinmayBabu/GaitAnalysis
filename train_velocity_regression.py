# ============================================================
# VELOCITY PREDICTION REGRESSION MODEL
# Predict Velocity_UGS_measured from biomechanical features
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------

df = pd.read_csv("features_engineered.csv")

print("Dataset Loaded")
print("Shape:", df.shape)


# ------------------------------------------------------------
# 2. DEFINE FEATURES AND TARGET
# ------------------------------------------------------------

features = [
    "FGI",
    "StabilityIndex",
    "MES",
    "PCI",
    "PRI"
]

# Ensure features exist
features = [f for f in features if f in df.columns]

target = "Velocity_UGS_measured"

if target not in df.columns:
    raise ValueError("Velocity_UGS_measured not found in dataset.")

print("\nFeatures Used:")
print(features)
print("Target:", target)


# ------------------------------------------------------------
# 3. PREPARE DATA
# ------------------------------------------------------------

X = df[features]
y = df[target]
groups = df["subject_id"]  # Prevent subject leakage


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
# 5. TRAIN RANDOM FOREST REGRESSOR
# ------------------------------------------------------------

model = RandomForestRegressor(
    n_estimators=400,
    max_depth=12,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
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
print("Velocity Regression Results")
print("==============================")
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")


# ------------------------------------------------------------
# 7. ACTUAL VS PREDICTED PLOT
# ------------------------------------------------------------

plt.figure()
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Velocity")
plt.ylabel("Predicted Velocity")
plt.title("Actual vs Predicted Velocity")
plt.tight_layout()
plt.savefig("velocity_actual_vs_predicted.png")
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
plt.title("Feature Importance")
plt.tight_layout()
plt.savefig("velocity_feature_importance.png")
plt.show()

print("\nFeature Importance Ranking:")
print(importances.sort_values(ascending=False))


# ------------------------------------------------------------
# 9. SAVE MODEL
# ------------------------------------------------------------

joblib.dump(model, "velocity_regressor.pkl")

print("\nModel saved as velocity_regressor.pkl")
print("Velocity regression pipeline completed successfully.")
