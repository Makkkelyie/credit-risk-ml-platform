"""Download the UCI credit-card default data, validate it, and load it into DuckDB.

The raw table lands in `raw.credit_clients`; everything downstream (cleaning, features,
tests) is done by the dbt project in `warehouse/`.
"""

from __future__ import annotations

import io
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import duckdb
import pandas as pd

UCI_URL = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"
RAW_DIR = Path("data/raw")
RAW_FILE = RAW_DIR / "default_of_credit_card_clients.xls"

PAY_COLS = ["pay_0", "pay_2", "pay_3", "pay_4", "pay_5", "pay_6"]
BILL_COLS = [f"bill_amt{i}" for i in range(1, 7)]
PAYAMT_COLS = [f"pay_amt{i}" for i in range(1, 7)]
EXPECTED_COLS = (
    ["id", "limit_bal", "sex", "education", "marriage", "age"]
    + PAY_COLS + BILL_COLS + PAYAMT_COLS + ["default_next_month"]
)


class DataQualityError(ValueError):
    """Raised when the raw data fails a hard check and must not be loaded."""


@dataclass
class DQReport:
    rows: int
    errors: list[str] = field(default_factory=list)    # hard failures - block the load
    warnings: list[str] = field(default_factory=list)  # documented quirks - cleaned in dbt

    def to_markdown(self) -> str:
        lines = [f"- Rows: {self.rows:,}"]
        lines += [f"- ERROR: {e}" for e in self.errors]
        lines += [f"- Note: {w}" for w in self.warnings]
        return "\n".join(lines)


def download(dest: Path = RAW_FILE) -> Path:
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(UCI_URL, timeout=60) as resp:
        payload = resp.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        xls_name = next(n for n in zf.namelist() if n.endswith(".xls"))
        dest.write_bytes(zf.read(xls_name))
    return dest


def read_raw(path: Path = RAW_FILE) -> pd.DataFrame:
    # Row 0 of the sheet is X1..X23 placeholder names; row 1 holds the real headers.
    df = pd.read_excel(path, header=1)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df.rename(columns={"default_payment_next_month": "default_next_month"})


def validate(df: pd.DataFrame) -> DQReport:
    report = DQReport(rows=len(df))
    missing = [c for c in EXPECTED_COLS if c not in df.columns]
    if missing:
        report.errors.append(f"missing columns: {missing}")
        return report

    if df["id"].duplicated().any():
        report.errors.append(f"{df['id'].duplicated().sum()} duplicate client ids")
    nulls = int(df[EXPECTED_COLS].isna().sum().sum())
    if nulls:
        report.errors.append(f"{nulls} null values")
    if (df["limit_bal"] <= 0).any():
        report.errors.append("non-positive credit limits")
    if not df["age"].between(18, 100).all():
        report.errors.append("ages outside 18-100")
    if not df["default_next_month"].isin([0, 1]).all():
        report.errors.append("target is not binary")
    if (df[PAYAMT_COLS] < 0).any().any():
        report.errors.append("negative payment amounts")

    # Known quirks of this dataset: codes that the data dictionary does not define.
    bad_edu = (~df["education"].isin([1, 2, 3, 4])).sum()
    bad_mar = (~df["marriage"].isin([1, 2, 3])).sum()
    undocumented_pay = df[PAY_COLS].isin([-2, 0]).any(axis=1).sum()
    neg_bills = (df[BILL_COLS] < 0).any(axis=1).sum()
    if bad_edu:
        report.warnings.append(f"{bad_edu} rows with undocumented education codes (0/5/6) -> mapped to 'other'")
    if bad_mar:
        report.warnings.append(f"{bad_mar} rows with undocumented marriage code 0 -> mapped to 'other'")
    if undocumented_pay:
        report.warnings.append(
            f"{undocumented_pay} rows use repayment-status codes -2/0 (not in the data dictionary; "
            "treated as 'no delay')"
        )
    if neg_bills:
        report.warnings.append(f"{neg_bills} rows with a negative bill (credit balance) in some month")
    report.warnings.append(f"default rate {df['default_next_month'].mean():.1%}")
    return report


def load_to_warehouse(df: pd.DataFrame, warehouse: Path) -> None:
    warehouse.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(warehouse)) as con:
        con.execute("create schema if not exists raw")
        con.register("incoming", df[EXPECTED_COLS])
        con.execute("create or replace table raw.credit_clients as select * from incoming")


def ingest(warehouse: Path, df: pd.DataFrame | None = None) -> DQReport:
    """Validate and load. Pass `df` to skip the download (tests use synthetic data)."""
    if df is None:
        df = read_raw(download())
    report = validate(df)
    if report.errors:
        raise DataQualityError("; ".join(report.errors))
    load_to_warehouse(df, warehouse)
    return report
