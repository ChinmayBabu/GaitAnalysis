import pandas as pd
import numpy as np

df = pd.read_csv("features_synthetic.csv")

# ---- Drop weak features ----
drop_cols = [
    "Step_FGS_measured","Stride_FGS_measured","Cadence_FGS_measured",
    "Velocity_FGS_estimated","Step_FGS_estimated","Stride_FGS_estimated"
]
df.drop(columns=drop_cols, errors="ignore", inplace=True)

# ---- Create derived features ----
df["FGI"] = (
    0.4*df["Velocity_UGS_measured"] +
    0.3*df["Stride_UGS_measured"] +
    0.3*df["Cadence_UGS_measured"]
)

df["StabilityIndex"] = df["MonoSP_UGS"] / (df["BiSP_UGS"] + 0.01)
df["MES"] = df["Velocity_UGS_measured"] / (df["BMI"] + 0.1)

# ---- Posture handling (robust) ----
df["PostureProxy"] = df["BiSP_UGS"] / (df["MonoSP_UGS"] + 0.01)
df["JointHealthProxy"] = df["Stride_UGS_measured"] / (df["Cadence_UGS_measured"] + 0.01)

# If true posture exists, use it. Otherwise fall back to proxies.
df["PCI"] = np.where(
    df["mean_knee_angle"].notna() & df["mean_trunk_angle"].notna(),
    df["mean_trunk_angle"] / (df["mean_knee_angle"] + 0.01),
    df["PostureProxy"]
)

# ---- Recompute PRI ----
df["PRI"] = (df["FGI"] * df["StabilityIndex"] * df["MES"]) / (df["PCI"] + 0.01)

# ---- Cleaning ----
df = df.sort_values(["subject_id","session"])

df.fillna(method="ffill", inplace=True)
df.fillna(df.median(numeric_only=True), inplace=True)

# Outlier clipping
for col in df.select_dtypes(include=np.number):
    lo, hi = df[col].quantile([0.01, 0.99])
    df[col] = df[col].clip(lo, hi)

df.to_csv("features_engineered.csv", index=False)

print("\n✅ features_engineered.csv created successfully\n")
