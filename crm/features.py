"""Feature definitions, dbt runner, and the shared training/serving feature path."""

from __future__ import annotations

import os
from pathlib import Path

import duckdb
import pandas as pd

DBT_DIR = Path(__file__).resolve().parent.parent / "warehouse"
FEATURE_SQL = DBT_DIR / "models" / "marts" / "fct_credit_features.sql"
TARGET = "default_next_month"

FEATURE_GROUPS: dict[str, list[str]] = {
    "credit_line": ["log_limit_bal"],
    "payment_history": ["delay_m1", "max_delay_6m", "n_months_delayed_6m", "delay_trend_3m", "inactive_m1"],
    "utilization": ["utilization_m1", "avg_utilization_6m", "utilization_trend_3m"],
    "repayment": ["repay_ratio_m1", "avg_repay_ratio_5m", "n_full_payments_5m", "log_total_paid_6m"],
}
BEHAVIOURAL = [f for group in FEATURE_GROUPS.values() for f in group]
# Protected / sensitive attributes: audited, never fed to the production model.
DEMOGRAPHIC = ["sex", "education", "marriage", "age"]

STAGING_COLS = (
    ["limit_bal"]
    + [f"pay_status_m{i}" for i in range(1, 7)]
    + [f"bill_amt_m{i}" for i in range(1, 7)]
    + [f"pay_amt_m{i}" for i in range(1, 7)]
)


def run_dbt(warehouse: Path, command: str = "build") -> None:
    """Run `dbt <command>` against `warehouse` and raise if any model or test fails."""
    from dbt.cli.main import dbtRunner

    os.environ["CRM_WAREHOUSE"] = str(Path(warehouse).resolve())
    result = dbtRunner().invoke(
        [command, "--project-dir", str(DBT_DIR), "--profiles-dir", str(DBT_DIR), "--quiet"]
    )
    if not result.success:
        raise RuntimeError(f"dbt {command} failed: {result.exception or 'see dbt output'}")


def load_table(warehouse: Path, table: str) -> pd.DataFrame:
    with duckdb.connect(str(warehouse)) as con:
        return con.execute(f"select * from main.{table} order by client_id").df()


def render_feature_sql(source_table: str) -> str:
    """The dbt feature model with its `ref()` pointed at an arbitrary table."""
    sql = FEATURE_SQL.read_text().replace("{{ ref('stg_credit_clients') }}", source_table)
    if "{{" in sql:
        raise ValueError("feature SQL has Jinja the serving path cannot render")
    return sql


def compute_features(staging_rows: pd.DataFrame) -> pd.DataFrame:
    """Run the *same* SQL dbt uses on staging-shaped rows (e.g. one API request).

    Using one definition for both paths removes training/serving skew by construction;
    tests/test_features.py checks the two paths agree row-for-row.
    """
    rows = staging_rows.copy()
    if "client_id" not in rows:
        rows["client_id"] = range(len(rows))
    for col in DEMOGRAPHIC + [TARGET]:
        if col not in rows:
            rows[col] = None
    numeric = [c for c in STAGING_COLS if c in rows]
    rows[numeric] = rows[numeric].astype(float)
    with duckdb.connect() as con:
        con.register("request_rows", rows)
        return con.execute(render_feature_sql("request_rows")).df()
