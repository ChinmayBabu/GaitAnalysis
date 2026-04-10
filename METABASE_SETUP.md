# Metabase Setup Guide

This repo now includes dashboard-ready CSV exports in `metabase_exports/`, a PostgreSQL loader script, and a Docker stack for PostgreSQL + Metabase.

## Files

- `gait_observations.csv`
- `subject_summary.csv`
- `session_summary.csv`
- `feature_summary.csv`

## PostgreSQL loader

Use `load_metabase_postgres.py` to push the CSV exports into PostgreSQL tables.

It expects these environment variables:

- `PGHOST`
- `PGPORT`
- `PGDATABASE`
- `PGUSER`
- `PGPASSWORD`
- `PGSSLMODE` optional

Example PowerShell session:

```powershell
$env:PGHOST="localhost"
$env:PGPORT="5432"
$env:PGDATABASE="gait_recovery"
$env:PGUSER="postgres"
$env:PGPASSWORD="your_password"
py .\load_metabase_postgres.py
```

## Docker route

If you want to keep everything local and isolated, use the Docker stack in `docker-compose.yml`.

### Set your password

Create a `.env` file in the project root with:

```env
POSTGRES_PASSWORD=your_actual_password_here
```

You can copy `.env.example` as a starting point.

If you already started the Postgres container with a different password, stop the stack and recreate the Postgres volume so the new password takes effect.

### Start the stack

```powershell
docker compose up -d
```

This launches:

- PostgreSQL on host port `5433`
- Metabase on `http://localhost:3000`

### Load the data into the Docker PostgreSQL container

```powershell
$env:PGHOST="localhost"
$env:PGPORT="5433"
$env:PGDATABASE="gait_recovery"
$env:PGUSER="postgres"
$env:PGPASSWORD="your_actual_password_here"
py .\load_metabase_postgres.py
```

### Connect Metabase to PostgreSQL

In the Metabase database setup screen, use:

- `Host`: `postgres`
- `Port`: `5432`
- `Database name`: `gait_recovery`
- `Username`: `postgres`
- `Password`: `your_actual_password_here`

## Suggested dashboard cards

- recovery stage distribution
- session-wise average recovery score
- session-wise average velocity
- correlation or feature summary view
- top features by importance
- actual vs predicted velocity
- subject-level drill-down

## How to use

1. Choose either the local PostgreSQL route or the Docker route.
2. Run `load_metabase_postgres.py` to import the exports into PostgreSQL.
3. Open Metabase.
4. Connect Metabase to the PostgreSQL database.
5. Build charts from the summary tables first.
6. Use `gait_observations` for drill-down and subject-level filtering.

## Why these tables exist

- `gait_observations` keeps the full row-level dataset.
- `subject_summary` makes per-subject analysis easy.
- `session_summary` supports time/progression charts.
- `feature_summary` supports quick descriptive overview cards.

## Good first charts in Metabase

- `session_summary.mean_Velocity_UGS_measured` by `session`
- `session_summary.mean_recovery_score` by `session`
- `subject_summary.avg_Velocity_UGS_measured` by `recovery_type`
- `feature_summary.mean` for selected numeric features
- `gait_observations` filtered by `subject_id`
