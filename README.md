# Gait Analysis and Recovery Modeling

This repository contains a full gait-recovery modeling pipeline built from raw gait measurements, pose-derived posture features, engineered biomechanical indices, exploratory analysis, and model evaluation.

The final recommended model is a non-leaky `ExtraTreesRegressor` that predicts `Velocity_UGS_measured` using subject-grouped validation.

## Project Goals

- Build clinically meaningful gait and recovery features
- Explore recovery progression using EDA and plots
- Compare stage classification, recovery regression, and velocity regression
- Avoid subject leakage and target leakage
- Present the data in Metabase for visualization and reporting

## Main Outcome

The strongest and most defensible final model in this repo is:

- Target: `Velocity_UGS_measured`
- Model: `ExtraTreesRegressor`
- Validation: subject-grouped cross-validation
- Final metrics:
  - `R^2 = 0.8493` in 5-fold grouped CV
  - `MAE = 0.0713`
  - `RMSE = 0.0978`

## Repository Structure

- `build_features.py` - merges raw CSVs and extracts posture features from pose JSON
- `feature_engineering.py` - creates engineered gait and recovery indices
- `generate_synthetic_recovery.py` - simulates longitudinal recovery data
- `eda_analysis.py` - performs exploratory data analysis and feature selection
- `train_stage_classifier.py` - recovery stage classification baseline
- `train_recovery_regression.py` - composite recovery index regression baseline
- `train_final_recovery_model.py` - recovery score regression attempt
- `train_velocity_regression.py` - baseline velocity regression
- `train_velocity_extra_trees.py` - final ExtraTrees velocity model
- `prepare_metabase_data.py` - creates dashboard-ready CSV exports
- `load_metabase_postgres.py` - loads exports into PostgreSQL
- `docker-compose.yml` - starts PostgreSQL and Metabase containers
- `PROJECT_DOCUMENTATION.md` - full project write-up
- `METABASE_SETUP.md` - Metabase setup guide

## Data Pipeline

1. Raw data is loaded from `data/healthgait_sample/`
2. Gait and patient tables are merged
3. Posture features are extracted from pose JSON files
4. Engineered features are created
5. EDA is run to inspect correlations, class balance, and significance
6. Multiple models are trained and compared
7. Final velocity model is trained with non-leaky features
8. Metabase-ready exports are created for dashboarding

## Key Features

Engineered features include:

- `FGI`
- `StabilityIndex`
- `MES`
- `PCI`
- `PRI`
- `PostureProxy`
- `JointHealthProxy`

## Important Modeling Note

Some engineered features are derived from `Velocity_UGS_measured`, so they are useful for interpretation but should not always be used as predictors when velocity is the target.

The final model avoids that leakage by using a non-leaky feature set.

## How to Run

### 1. Build the engineered dataset

```powershell
py .\build_features.py
py .\feature_engineering.py
```

### 2. Run exploratory analysis

```powershell
py .\eda_analysis.py
```

### 3. Train the final model

```powershell
py .\train_velocity_extra_trees.py
```

### 4. Prepare Metabase exports

```powershell
py .\prepare_metabase_data.py
```

## Metabase Visualization

This repo includes a Metabase workflow for visualization and reporting.

Options:

- Use the local PostgreSQL setup
- Or use the Docker-based PostgreSQL + Metabase setup in `docker-compose.yml`

Recommended dashboard cards:

- recovery stage distribution
- average velocity by session
- average recovery score by session
- velocity by recovery type
- feature summary table
- subject-level drill-down

## Docker Setup

If you want to use the bundled container setup:

```powershell
docker compose up -d
```

Then import the exported CSVs into PostgreSQL with:

```powershell
py .\load_metabase_postgres.py
```

See `METABASE_SETUP.md` for the full step-by-step process.

## Plots and Outputs

Important plots produced by the repo include:

- `eda_class_distribution.png`
- `eda_correlation_matrix.png`
- `confusion_matrix.png`
- `regression_actual_vs_predicted.png`
- `final_recovery_actual_vs_predicted.png`
- `final_recovery_feature_importance.png`
- `velocity_extra_trees_actual_vs_predicted.png`

## Final Recommendation

If your goal is the best balance of accuracy, interpretability, and statistical honesty, use:

- `ExtraTreesRegressor`
- `Velocity_UGS_measured`
- subject-grouped validation
- non-leaky predictors only

## Documentation

- `PROJECT_DOCUMENTATION.md` contains the full project narrative from raw data to final model selection
- `METABASE_SETUP.md` explains the visualization workflow

