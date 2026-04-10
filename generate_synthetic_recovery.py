import pandas as pd
import numpy as np

# ---------------------------
# Load real features
# ---------------------------
df = pd.read_csv("features_real.csv")

# ---------------------------
# Recovery simulation config
# ---------------------------
timepoints = ["week1", "week3", "week6", "week10"]
recovery_rate = {
    "fast": 0.35,
    "medium": 0.22,
    "slow": 0.12
}

synthetic_rows = []

# ---------------------------
# Simulation
# ---------------------------
for _, row in df.iterrows():
    
    recovery_type = np.random.choice(["fast", "medium", "slow"], p=[0.3, 0.5, 0.2])
    k = recovery_rate[recovery_type]
    
    baseline = row.copy()

    for i, t in enumerate(timepoints):
        factor = np.exp(-k * i)

        new_row = row.copy()
        new_row["session"] = t
        new_row["recovery_type"] = recovery_type

        # Improvement rules
        new_row["Velocity_UGS_measured"] *= (1 + 0.25 * (1 - factor))
        new_row["Velocity_FGS_measured"] *= (1 + 0.3 * (1 - factor))
        new_row["Step_UGS_measured"]     *= (1 + 0.2 * (1 - factor))
        new_row["Stride_UGS_measured"]   *= (1 + 0.2 * (1 - factor))
        new_row["Cadence_UGS_measured"]  *= (1 + 0.1 * (1 - factor))

        new_row["mean_knee_angle"] *= (1 + 0.15 * (1 - factor))
        new_row["mean_trunk_angle"] *= factor

        # Stability improvement
        new_row["BiSP_UGS"] *= factor
        new_row["MonoSP_UGS"] *= (1 + 0.1 * (1 - factor))

        # Noise injection (numeric columns only)
        noise = np.random.normal(0, 0.02)
        numeric_cols = new_row.index[new_row.apply(lambda x: isinstance(x, (int, float, np.number)))]
        new_row[numeric_cols] = new_row[numeric_cols] * (1 + noise)


        synthetic_rows.append(new_row)

# ---------------------------
# Create synthetic dataset
# ---------------------------
synthetic_df = pd.DataFrame(synthetic_rows)

# ---------------------------
# Recovery labels
# ---------------------------
baseline_vel = synthetic_df.groupby("subject_id")["Velocity_UGS_measured"].transform("first")
synthetic_df["recovery_score"] = (synthetic_df["Velocity_UGS_measured"] / baseline_vel) * 100
synthetic_df["recovery_phase"] = pd.cut(
    synthetic_df["recovery_score"],
    bins=[0, 110, 125, 1000],
    labels=["early", "mid", "late"]
)

synthetic_df.to_csv("features_synthetic.csv", index=False)

print("\n✅ features_synthetic.csv created successfully.\n")
