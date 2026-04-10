"""Load Metabase-ready exports into PostgreSQL.

Expected environment variables:
- PGHOST
- PGPORT
- PGDATABASE
- PGUSER
- PGPASSWORD
- PGSSLMODE (optional)

This script loads the CSV exports created in `metabase_exports/` into a
PostgreSQL database so Metabase can connect to it directly.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text


EXPORT_DIR = Path("metabase_exports")


def build_engine():
    host = os.getenv("PGHOST", "localhost")
    port = os.getenv("PGPORT", "5432")
    database = os.getenv("PGDATABASE")
    user = os.getenv("PGUSER")
    password = os.getenv("PGPASSWORD")
    sslmode = os.getenv("PGSSLMODE")

    if not database or not user:
        raise ValueError("PGDATABASE and PGUSER must be set.")

    auth = f"{user}:{password or ''}"
    if password is None:
        auth = user

    url = f"postgresql+psycopg2://{auth}@{host}:{port}/{database}"
    if sslmode:
        url += f"?sslmode={sslmode}"

    return create_engine(url, future=True)


def load_table(engine, csv_path: Path, table_name: str) -> None:
    df = pd.read_csv(csv_path)
    with engine.begin() as conn:
        df.to_sql(table_name, conn, if_exists="replace", index=False, method="multi")


def main() -> None:
    if not EXPORT_DIR.exists():
        raise FileNotFoundError(
            f"{EXPORT_DIR} not found. Run prepare_metabase_data.py first."
        )

    engine = build_engine()

    table_map = {
        "gait_observations": EXPORT_DIR / "gait_observations.csv",
        "subject_summary": EXPORT_DIR / "subject_summary.csv",
        "session_summary": EXPORT_DIR / "session_summary.csv",
        "feature_summary": EXPORT_DIR / "feature_summary.csv",
    }

    with engine.begin() as conn:
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS public;"))

    for table_name, csv_path in table_map.items():
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing CSV export: {csv_path}")
        print(f"Loading {csv_path.name} -> {table_name}")
        load_table(engine, csv_path, table_name)

    print("\nLoad complete.")
    print("Tables available for Metabase:")
    for table_name in table_map:
        print(f"- {table_name}")


if __name__ == "__main__":
    main()
