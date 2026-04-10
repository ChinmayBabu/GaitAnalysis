# ============================================================
# EDA ANALYSIS FOR GAIT RECOVERY STAGE CLASSIFICATION
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway

# ------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------

df = pd.read_csv("features_engineered.csv")

print("\nDataset Loaded Successfully")
print("Shape:", df.shape)


# ------------------------------------------------------------
# 2. BASIC DATA OVERVIEW
# ------------------------------------------------------------

print("\n--- Dataset Info ---")
print(df.info())

print("\n--- Summary Statistics ---")
print(df.describe())

print("\n--- Missing Values ---")
print(df.isnull().sum())


# ------------------------------------------------------------
# 3. DEFINE TARGET VARIABLE (Recovery Stage)
# ------------------------------------------------------------

if "recovery_phase" in df.columns:
    df["recovery_stage"] = df["recovery_phase"].map({
        "early": "Early",
        "mid": "Moderate",
        "late": "Fully"
    })

elif "recovery_stage" not in df.columns:
    raise ValueError("No recovery phase column found.")

print("\n--- Recovery Stage Distribution ---")
print(df["recovery_stage"].value_counts())


# ------------------------------------------------------------
# 4. CLASS DISTRIBUTION PLOT
# ------------------------------------------------------------

plt.figure()
df["recovery_stage"].value_counts().plot(kind="bar")
plt.title("Recovery Stage Distribution")
plt.xlabel("Recovery Stage")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("eda_class_distribution.png")
plt.show()


# ------------------------------------------------------------
# 5. NUMERIC FEATURE SELECTION
# ------------------------------------------------------------

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

# Remove non-feature numeric columns
drop_cols = ["subject_id", "session"]
numeric_features = [col for col in numeric_cols if col not in drop_cols]

print("\nNumeric Features Considered:")
print(numeric_features)


# ------------------------------------------------------------
# 6. CORRELATION MATRIX
# ------------------------------------------------------------

corr = df[numeric_features].corr()

plt.figure(figsize=(12, 8))
sns.heatmap(corr, cmap="coolwarm", center=0)
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig("eda_correlation_matrix.png")
plt.show()

# Identify highly correlated features (> 0.90)
high_corr_pairs = []

for i in range(len(corr.columns)):
    for j in range(i):
        if abs(corr.iloc[i, j]) > 0.90:
            high_corr_pairs.append((corr.columns[i], corr.columns[j]))

print("\nHighly Correlated Feature Pairs (>0.90):")
print(high_corr_pairs)


# ------------------------------------------------------------
# 7. BOX PLOTS (Feature vs Recovery Stage)
# ------------------------------------------------------------

important_features = [
    col for col in numeric_features
    if col in ["FGI", "StabilityIndex", "MES",
               "Velocity_UGS_measured",
               "Cadence_measured",
               "PCI", "PRI"]
]

for feature in important_features:
    if feature in df.columns:
        plt.figure()
        sns.boxplot(x="recovery_stage", y=feature, data=df)
        plt.title(f"{feature} vs Recovery Stage")
        plt.tight_layout()
        plt.savefig(f"eda_boxplot_{feature}.png")
        plt.show()


# ------------------------------------------------------------
# 8. ANOVA TEST (Statistical Significance)
# ------------------------------------------------------------

print("\n--- ANOVA Results ---")

anova_results = {}

for feature in important_features:
    if feature in df.columns:

        early = df[df["recovery_stage"] == "Early"][feature]
        moderate = df[df["recovery_stage"] == "Moderate"][feature]
        fully = df[df["recovery_stage"] == "Fully"][feature]

        if len(early) > 1 and len(moderate) > 1 and len(fully) > 1:
            f_stat, p_value = f_oneway(early, moderate, fully)
            anova_results[feature] = p_value
            print(f"{feature} → p-value: {p_value:.6f}")

print("\nSignificant Features (p < 0.05):")
significant_features = [k for k, v in anova_results.items() if v < 0.05]
print(significant_features)


# ------------------------------------------------------------
# 9. OUTLIER ANALYSIS (IQR METHOD)
# ------------------------------------------------------------

print("\n--- Outlier Detection (IQR) ---")

for feature in important_features:
    if feature in df.columns:
        Q1 = df[feature].quantile(0.25)
        Q3 = df[feature].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        outliers = df[(df[feature] < lower) | (df[feature] > upper)]
        print(f"{feature} → {len(outliers)} outliers")


# ------------------------------------------------------------
# 10. SAVE SELECTED FEATURES FOR MODELING
# ------------------------------------------------------------

final_features = list(set(significant_features))

if not final_features:
    final_features = important_features  # fallback

print("\nFinal Selected Features for Modeling:")
print(final_features)

pd.Series(final_features).to_csv("selected_features.txt", index=False)

print("\nEDA Completed Successfully.")
print("Plots saved as PNG files.")
