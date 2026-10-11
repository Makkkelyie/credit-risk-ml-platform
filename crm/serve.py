"""Scoring API. Loads the `champion` model from the MLflow registry and scores raw
account history through the same feature SQL that dbt uses for training.

    uvicorn crm.serve:app --port 8000
"""

from __future__ import annotations

import os

from fastapi import FastAPI
from pydantic import BaseModel, Field

from crm.features import compute_features

MLFLOW_DB = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")


class AccountHistory(BaseModel):
    """Six months of statement history; m1 is the most recent month."""

    limit_bal: float = Field(gt=0)
    pay_status: list[int] = Field(min_length=6, max_length=6, description="-2..9, m1 first")
    bill_amt: list[float] = Field(min_length=6, max_length=6)
    pay_amt: list[float] = Field(min_length=6, max_length=6)

    def to_staging_row(self) -> dict:
        row = {"limit_bal": self.limit_bal}
        for i in range(6):
            row[f"pay_status_m{i + 1}"] = self.pay_status[i]
            row[f"bill_amt_m{i + 1}"] = self.bill_amt[i]
            row[f"pay_amt_m{i + 1}"] = max(self.pay_amt[i], 0.0)
        return row


class Score(BaseModel):
    probability_of_default: float
    risk_band: str
    model_version: str


def risk_band(p: float) -> str:
    return "high" if p >= 0.5 else "medium" if p >= 0.2 else "low"


def create_app(model=None, version: str | None = None) -> FastAPI:
    """`model`/`version` can be injected (tests); otherwise the champion is loaded."""
    api = FastAPI(title="Credit default scoring")
    state = {"model": model, "version": version}

    def get_model():
        if state["model"] is None:
            import mlflow

            from crm.registry import load_champion

            mlflow.set_tracking_uri(MLFLOW_DB)
            state["model"], state["version"] = load_champion()
        return state["model"], state["version"]

    @api.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @api.post("/score", response_model=Score)
    def score(account: AccountHistory) -> Score:
        import pandas as pd

        mdl, ver = get_model()
        features = compute_features(pd.DataFrame([account.to_staging_row()]))
        p = float(mdl.predict_proba(features)[:, 1][0])
        return Score(probability_of_default=round(p, 4), risk_band=risk_band(p), model_version=str(ver))

    return api


app = create_app()
