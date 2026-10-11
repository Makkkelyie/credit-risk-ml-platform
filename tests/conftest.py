"""Synthetic raw data shaped like the UCI file, so tests never touch the network."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from crm.features import run_dbt
from crm.ingest import ingest


def synthetic_raw(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    limit = rng.choice([20_000, 50_000, 100_000, 200_000], n).astype(float)
    pay = rng.choice([-2, -1, 0, 1, 2, 3], size=(n, 6), p=[0.15, 0.2, 0.45, 0.1, 0.07, 0.03])
    bills = (limit[:, None] * rng.uniform(-0.05, 0.9, (n, 6))).round()
    pays = (np.clip(bills, 0, None) * rng.uniform(0, 1.1, (n, 6))).round()
    risk = 0.6 * np.clip(pay[:, 0], 0, None) + bills[:, 0] / limit - 1.5
    df = pd.DataFrame({
        "id": np.arange(1, n + 1),
        "limit_bal": limit,
        "sex": rng.choice([1, 2], n),
        "education": rng.choice([0, 1, 2, 3, 4, 5, 6], n),
        "marriage": rng.choice([0, 1, 2, 3], n),
        "age": rng.integers(21, 70, n),
    })
    for i, c in enumerate(["pay_0", "pay_2", "pay_3", "pay_4", "pay_5", "pay_6"]):
        df[c] = pay[:, i]
    for i in range(6):
        df[f"bill_amt{i + 1}"] = bills[:, i]
        df[f"pay_amt{i + 1}"] = pays[:, i]
    df["default_next_month"] = (rng.random(n) < 1 / (1 + np.exp(-risk))).astype(int)
    return df


@pytest.fixture
def raw() -> pd.DataFrame:
    return synthetic_raw()


@pytest.fixture(scope="session")
def warehouse(tmp_path_factory):
    """A DuckDB warehouse built by the real ingest + dbt build from synthetic data."""
    path = tmp_path_factory.mktemp("wh") / "warehouse.duckdb"
    ingest(path, synthetic_raw())
    run_dbt(path)
    return path
