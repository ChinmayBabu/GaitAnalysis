# ============================================================
# TRAINING RECOVERY INDEX REGRESSION MODEL
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------

df = pd.read_csv("features_engineered.csv")

print("Dataset Loaded")
print("Shape:", df.shape)


# ------------------------------------------------------------
# 2. SELECT FEATURES
# ------------------------------------------------------------

features = [
    "FGI",
    "StabilityIndex",
    "MES",
    "Velocity_UGS_measured",
    "PCI",
    "PRI"
]

features = [f for f in features if f in df.columns]

print("\nFeatures Used:")
print(features)


# ------------------------------------------------------------
# 3. CREATE RECOVERY INDEX (STANDARDIZED)
# ------------------------------------------------------------

scaler = StandardScaler()
scaled_values = scaler.fit_transform(df[features])

scaled_df = pd.DataFrame(scaled_values, columns=features)

df["Recovery_Index"] = scaled_df.mean(axis=1)

print("\nRecovery Index Created.")


# ------------------------------------------------------------
# 4. PREPARE DATA
# ------------------------------------------------------------

X = df[features]
y = df["Recovery_Index"]
groups = df["subject_id"]

# ------------------------------------------------------------
# 5. GROUP-BASED SPLIT (NO LEAKAGE)
# ------------------------------------------------------------

gss = GroupShuffleSplit(test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# ------------------------------------------------------------
# 6. TRAIN RANDOM FOREST REGRESSOR
# ------------------------------------------------------------

model = RandomForestRegressor(
    n_estimators=400,
    max_depth=12,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("\nModel Training Completed.")


# ------------------------------------------------------------
# 7. EVALUATION
# ------------------------------------------------------------

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n==============================")
print("Regression Evaluation Results")
print("==============================")
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")


# ------------------------------------------------------------
# 8. PREDICTED VS ACTUAL PLOT
# ------------------------------------------------------------

plt.figure()
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Recovery Index")
plt.ylabel("Predicted Recovery Index")
plt.title("Actual vs Predicted Recovery Index")
plt.tight_layout()
plt.savefig("regression_actual_vs_predicted.png")
plt.show()


# ------------------------------------------------------------
# 9. FEATURE IMPORTANCE
# ------------------------------------------------------------

importances = pd.Series(
    model.feature_importances_,
    index=features
).sort_values(ascending=True)

plt.figure()
importances.plot(kind="barh")
plt.title("Feature Importance")
plt.tight_layout()
plt.savefig("regression_feature_importance.png")
plt.show()

print("\nFeature Importance Ranking:")
print(importances.sort_values(ascending=False))


# ------------------------------------------------------------
# 10. SAVE MODEL
# ------------------------------------------------------------

joblib.dump(model, "recovery_index_regressor.pkl")

print("\nModel saved as recovery_index_regressor.pkl")
print("Regression pipeline completed successfully.")
