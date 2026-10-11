"""Model registry + promotion gate.

A challenger only replaces the champion when it is *measurably* better on the same
holdout rows - not just higher on one number.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlflow
from mlflow import MlflowClient

from crm.metrics import bootstrap_auc_diff, evaluate

MODEL_NAME = "credit-default"
ALIAS = "champion"
MAX_BRIER_INCREASE = 0.002  # calibration must not get meaningfully worse


@dataclass
class GateDecision:
    promote: bool
    reason: str
    auc_diff: float = float("nan")
    auc_diff_ci: tuple[float, float] = (float("nan"), float("nan"))
    brier_diff: float = float("nan")


def gate(y_true, p_challenger, p_champion=None) -> GateDecision:
    """Promote if the 95% CI of the paired AUC gain is above 0 and Brier doesn't regress."""
    if p_champion is None:
        return GateDecision(True, "no champion yet")
    mean, lo, hi = bootstrap_auc_diff(y_true, p_challenger, p_champion)
    brier_diff = evaluate(y_true, p_challenger)["brier"] - evaluate(y_true, p_champion)["brier"]
    common = dict(auc_diff=mean, auc_diff_ci=(lo, hi), brier_diff=brier_diff)
    if lo <= 0:
        return GateDecision(False, f"AUC gain not significant (95% CI {lo:+.4f} to {hi:+.4f})", **common)
    if brier_diff > MAX_BRIER_INCREASE:
        return GateDecision(False, f"calibration regressed (Brier +{brier_diff:.4f})", **common)
    return GateDecision(True, f"AUC +{mean:.4f} (95% CI {lo:+.4f} to {hi:+.4f}), Brier {brier_diff:+.4f}", **common)


def register(model_uri: str, tags: dict[str, str]) -> str:
    version = mlflow.register_model(model_uri, MODEL_NAME, tags=tags).version
    return str(version)


def champion_version() -> str | None:
    try:
        return str(MlflowClient().get_model_version_by_alias(MODEL_NAME, ALIAS).version)
    except mlflow.exceptions.MlflowException:
        return None


def load(version: str):
    return mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{version}")


def load_champion():
    return mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{ALIAS}"), champion_version()


def promote(version: str, decision: GateDecision) -> None:
    client = MlflowClient()
    client.set_model_version_tag(MODEL_NAME, version, "gate_decision", decision.reason)
    if decision.promote:
        client.set_registered_model_alias(MODEL_NAME, ALIAS, version)
