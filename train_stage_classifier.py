# ============================================================
# TRAINING RECOVERY STAGE CLASSIFIER
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)
from sklearn.model_selection import GroupShuffleSplit

# ------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------

df = pd.read_csv("features_engineered.csv")

print("Dataset Loaded")
print("Shape:", df.shape)


# ------------------------------------------------------------
# 2. DEFINE TARGET VARIABLE
# ------------------------------------------------------------

if "recovery_phase" in df.columns:
    df["recovery_stage"] = df["recovery_phase"].map({
        "early": "Early",
        "mid": "Moderate",
        "late": "Fully"
    })

elif "recovery_stage" not in df.columns:
    raise ValueError("Recovery stage column not found.")

print("\nClass Distribution:")
print(df["recovery_stage"].value_counts())


# ------------------------------------------------------------
# 3. LOAD SELECTED FEATURES FROM EDA
# ------------------------------------------------------------

try:
    selected_features = pd.read_csv("selected_features.txt", header=None)[0].tolist()
    print("\nUsing EDA-selected features:")
    print(selected_features)
except:
    # fallback
    print("\nselected_features.txt not found. Using default features.")
    selected_features = [
        "FGI", "StabilityIndex", "MES",
        "Velocity_UGS_measured",
        "PCI", "PRI"
    ]

# Ensure features exist
selected_features = [f for f in selected_features if f in df.columns]


# ------------------------------------------------------------
# 4. PREPARE DATA
# ------------------------------------------------------------

X = df[selected_features]
y = df["recovery_stage"]
groups = df["subject_id"]  # Prevent leakage

print("\nFinal Feature Set Used:")
print(selected_features)


# ------------------------------------------------------------
# 5. GROUP-BASED TRAIN-TEST SPLIT
# ------------------------------------------------------------

gss = GroupShuffleSplit(test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# ------------------------------------------------------------
# 6. TRAIN RANDOM FOREST CLASSIFIER
# ------------------------------------------------------------

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("\nModel Training Completed.")


# ------------------------------------------------------------
# 7. EVALUATION
# ------------------------------------------------------------

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n==============================")
print("Model Evaluation Results")
print("==============================")
print(f"Accuracy: {accuracy:.4f}\n")

print("Classification Report:")
print(classification_report(y_test, y_pred))


# ------------------------------------------------------------
# 8. CONFUSION MATRIX
# ------------------------------------------------------------

cm = confusion_matrix(y_test, y_pred)

plt.figure()
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=model.classes_,
    yticklabels=model.classes_
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()


# ------------------------------------------------------------
# 9. FEATURE IMPORTANCE
# ------------------------------------------------------------

importances = pd.Series(
    model.feature_importances_,
    index=selected_features
).sort_values(ascending=True)

plt.figure()
importances.plot(kind="barh")
plt.title("Feature Importance")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()

print("\nFeature Importance Ranking:")
print(importances.sort_values(ascending=False))


# ------------------------------------------------------------
# 10. SAVE MODEL
# ------------------------------------------------------------

joblib.dump(model, "recovery_stage_classifier.pkl")

print("\nModel saved as recovery_stage_classifier.pkl")
print("Training pipeline completed successfully.")
