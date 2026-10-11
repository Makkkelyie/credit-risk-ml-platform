"""Post-deployment monitoring: population stability (PSI) of features and scores, and a
fairness audit of the champion across sensitive groups."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from crm.features import BEHAVIOURAL, TARGET

PSI_WATCH, PSI_ALERT = 0.10, 0.25  # common industry rule-of-thumb thresholds


def psi(expected, actual, n_bins: int = 10) -> float:
    """Population Stability Index of `actual` against the `expected` (training) distribution.

    Bins are expected-distribution quantiles; discrete features collapse to fewer bins.
    """
    expected, actual = np.asarray(expected, float), np.asarray(actual, float)
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 2:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-4, None), np.clip(a, 1e-4, None)
    return float(np.sum((a - e) * np.log(a / e)))


def status(value: float) -> str:
    return "ALERT" if value >= PSI_ALERT else "WATCH" if value >= PSI_WATCH else "OK"


def drift_report(reference: pd.DataFrame, batch: pd.DataFrame, model) -> pd.DataFrame:
    rows = [{"signal": f, "psi": psi(reference[f], batch[f])} for f in BEHAVIOURAL]
    rows.append({"signal": "model_score (PD)",
                 "psi": psi(model.predict_proba(reference)[:, 1], model.predict_proba(batch)[:, 1])})
    out = pd.DataFrame(rows)
    out["status"] = out["psi"].map(status)
    return out.sort_values("psi", ascending=False).reset_index(drop=True)


def stress_scenario(staging: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Simulate a downturn on staging-shaped rows: balances up 20%, repayments down 30%,
    and 25% of clients slip one more month behind on their latest payment."""
    rng = np.random.default_rng(seed)
    out = staging.copy()
    bills = [f"bill_amt_m{i}" for i in range(1, 4)]
    pays = [f"pay_amt_m{i}" for i in range(1, 4)]
    out[bills] = out[bills] * 1.2
    out[pays] = out[pays] * 0.7
    slip = rng.random(len(out)) < 0.25
    out.loc[slip, "pay_status_m1"] = out.loc[slip, "pay_status_m1"].clip(lower=0) + 1
    return out


def fairness_audit(df: pd.DataFrame, model, threshold: float = 0.5) -> pd.DataFrame:
    """Per-group AUC, observed default rate, mean PD and share flagged high-risk."""
    scored = df.assign(
        pd_score=model.predict_proba(df)[:, 1],
        sex_group=df["sex"].map({1: "male", 2: "female"}),
        age_band=pd.cut(df["age"], [0, 29, 44, 200], labels=["<30", "30-44", "45+"]),
    )
    rows = []
    for attr in ["sex_group", "age_band"]:
        for group, g in scored.groupby(attr, observed=True):
            rows.append({
                "attribute": attr.replace("_group", "").replace("_band", " band"),
                "group": str(group), "n": len(g),
                "auc": roc_auc_score(g[TARGET], g["pd_score"]),
                "default_rate": g[TARGET].mean(),
                "mean_pd": g["pd_score"].mean(),
                "flag_rate": (g["pd_score"] >= threshold).mean(),
            })
    return pd.DataFrame(rows)
