"""Structured experiments: baseline vs challenger, plus feature-group ablations.

Every experiment is tracked in MLflow (params, CV + holdout metrics, the fitted pipeline).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import mlflow
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from crm.features import BEHAVIOURAL, DEMOGRAPHIC, FEATURE_GROUPS, TARGET
from crm.metrics import evaluate

SEED = 42
EXPERIMENT = "credit-default"
# MLflow serializes with skops, which only loads allow-listed types; these are our own models.
TRUSTED_TYPES = ["collections.OrderedDict", "lightgbm.basic.Booster", "lightgbm.sklearn.LGBMClassifier"]


def logistic() -> ClassifierMixin:
    return LogisticRegression(max_iter=2000, C=1.0)


def lightgbm() -> ClassifierMixin:
    return LGBMClassifier(
        n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=100,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
        importance_type="gain", random_state=SEED, verbose=-1,
    )


def make_pipeline(features: list[str], estimator: Callable[[], ClassifierMixin]) -> Pipeline:
    # The pipeline selects its own columns, so callers can pass the full feature table.
    scale = estimator is logistic
    select = ColumnTransformer(
        [("x", StandardScaler() if scale else "passthrough", features)], remainder="drop"
    )
    return Pipeline([("select", select), ("model", estimator())])


@dataclass(frozen=True)
class Experiment:
    name: str
    role: str  # baseline | challenger | ablation | audit
    features: list[str]
    estimator: Callable[[], ClassifierMixin]
    note: str = ""


def experiments() -> list[Experiment]:
    exps = [
        Experiment("baseline_scorecard", "baseline", ["delay_m1", "utilization_m1"], logistic,
                   "2-feature logistic scorecard: last month's delay + utilization"),
        Experiment("logistic_all", "challenger", BEHAVIOURAL, logistic,
                   "logistic regression on all behavioural features"),
        Experiment("lightgbm_all", "challenger", BEHAVIOURAL, lightgbm,
                   "gradient-boosted trees on all behavioural features"),
    ]
    for group, cols in FEATURE_GROUPS.items():
        exps.append(Experiment(f"lightgbm_minus_{group}", "ablation",
                               [f for f in BEHAVIOURAL if f not in cols], lightgbm,
                               f"lightgbm_all without the {group} group"))
    exps.append(Experiment("lightgbm_plus_demographics", "audit", BEHAVIOURAL + DEMOGRAPHIC, lightgbm,
                           "measures what sensitive attributes would add; not eligible to ship"))
    return exps


def split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    return train_test_split(df, test_size=0.2, stratify=df[TARGET], random_state=SEED)


def run_experiments(df: pd.DataFrame, cv_folds: int = 5) -> tuple[pd.DataFrame, dict[str, dict]]:
    """Fit and log every experiment. Returns a results table and per-run details."""
    train, test = split(df)
    mlflow.set_experiment(EXPERIMENT)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=SEED)
    rows, runs = [], {}
    for exp in experiments():
        pipe = make_pipeline(exp.features, exp.estimator)
        cv_auc = cross_val_score(pipe, train, train[TARGET], cv=cv, scoring="roc_auc")
        pipe.fit(train, train[TARGET])
        holdout = evaluate(test[TARGET], pipe.predict_proba(test)[:, 1])

        with mlflow.start_run(run_name=exp.name) as run:
            mlflow.set_tags({"role": exp.role, "note": exp.note})
            mlflow.log_params({
                "estimator": type(pipe.named_steps["model"]).__name__,
                "n_features": len(exp.features),
                "features": ",".join(exp.features),
                "train_rows": len(train), "test_rows": len(test),
            })
            mlflow.log_metrics({"cv_auc_mean": cv_auc.mean(), "cv_auc_std": cv_auc.std(),
                                **{f"holdout_{k}": v for k, v in holdout.items()}})
            info = mlflow.sklearn.log_model(pipe, name="model", input_example=test.head(3),
                                            skops_trusted_types=TRUSTED_TYPES)

        runs[exp.name] = {"experiment": exp, "pipeline": pipe, "model_uri": info.model_uri,
                          "run_id": run.info.run_id}
        rows.append({"experiment": exp.name, "role": exp.role, "n_features": len(exp.features),
                     "cv_auc": cv_auc.mean(), "cv_auc_std": cv_auc.std(),
                     **{f"holdout_{k}": v for k, v in holdout.items()}})
    return pd.DataFrame(rows), runs


def pick_challenger(results: pd.DataFrame) -> str:
    """Best CV AUC among shippable models (ablations and the demographic audit excluded)."""
    eligible = results[results["role"].isin(["baseline", "challenger"])]
    return str(eligible.sort_values("cv_auc", ascending=False).iloc[0]["experiment"])


def feature_importance(pipe: Pipeline, features: list[str]) -> pd.Series:
    model = pipe.named_steps["model"]
    values = model.feature_importances_ if hasattr(model, "feature_importances_") else np.abs(model.coef_[0])
    return pd.Series(values, index=features).sort_values(ascending=False)
