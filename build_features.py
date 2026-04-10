import pandas as pd
import numpy as np
import os
import json

DATA_DIR = "data/healthgait_sample"

# -----------------------------
# 1. Load CSV files
# -----------------------------
gait_params = pd.read_csv(os.path.join(DATA_DIR, "gait_parameters.csv"))
gait_est = pd.read_csv(os.path.join(DATA_DIR, "gait_parameters_estimation.csv"))
patients = pd.read_csv(os.path.join(DATA_DIR, "patients_measures.csv"))

# -----------------------------
# 2. Standardize column names
# -----------------------------
gait_params = gait_params.rename(columns={"ID": "subject_id"})
gait_est = gait_est.rename(columns={"ID": "subject_id"})
patients = patients.rename(columns={"ID": "subject_id"})

# -----------------------------
# 3. Merge all tabular data
# -----------------------------
features = gait_params.merge(
    gait_est, 
    on="subject_id", 
    how="left", 
    suffixes=("_measured", "_estimated")
)

features = features.merge(patients, on="subject_id", how="left")

# -----------------------------
# 4. Posture Feature Extraction
# -----------------------------
POSE_DIR = os.path.join(DATA_DIR, "pose")

def compute_posture_features(pose_file):
    with open(pose_file) as f:
        frames = json.load(f)

    knee_angles = []
    trunk_angles = []

    for frame in frames:
        kp = frame["keypoints"]

        hip = np.array(kp[11])
        knee = np.array(kp[13])
        ankle = np.array(kp[15])
        shoulder = np.array(kp[5])

        # Knee angle
        v1 = hip - knee
        v2 = ankle - knee
        angle_knee = np.degrees(np.arccos(
            np.clip(
                np.dot(v1, v2) / (np.linalg.norm(v1)*np.linalg.norm(v2)), 
                -1.0, 1.0
            )
        ))
        knee_angles.append(angle_knee)

        # Trunk angle
        trunk_vec = shoulder - hip
        vertical = np.array([0, -1])
        angle_trunk = np.degrees(np.arccos(
            np.clip(
                np.dot(trunk_vec, vertical) / (np.linalg.norm(trunk_vec)), 
                -1.0, 1.0
            )
        ))
        trunk_angles.append(angle_trunk)

    return np.mean(knee_angles), np.mean(trunk_angles)

# -----------------------------
# 5. Apply posture extraction
# -----------------------------
knee_vals, trunk_vals = [], []

for sid in features["subject_id"]:
    pose_path = os.path.join(POSE_DIR, f"{sid}.json")

    if os.path.exists(pose_path):
        knee, trunk = compute_posture_features(pose_path)
    else:
        knee, trunk = np.nan, np.nan

    knee_vals.append(knee)
    trunk_vals.append(trunk)

features["mean_knee_angle"] = knee_vals
features["mean_trunk_angle"] = trunk_vals

# -----------------------------
# 6. Save output
# -----------------------------
features.to_csv("features_real.csv", index=False)

print("\n✅ features_real.csv created successfully.\n")
