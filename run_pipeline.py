"""End-to-end run: ingest -> dbt build -> experiments -> registry gate -> monitoring.

    python run_pipeline.py            # writes outputs/results.md and outputs/*.png
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
import matplotlib.pyplot as plt
import mlflow
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve

from crm import registry
from crm.features import compute_features, load_table, run_dbt, TARGET
from crm.ingest import ingest
from crm.monitor import drift_report, fairness_audit, stress_scenario
from crm.train import feature_importance, pick_challenger, run_experiments, split

OUT = Path("outputs")


def fmt_table(df, floats: str = "{:.3f}") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = [floats.format(v) if isinstance(v, float) else str(v) for v in r]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def plot(test, models: dict) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    for name, m in models.items():
        p = m.predict_proba(test)[:, 1]
        fpr, tpr, _ = roc_curve(test[TARGET], p)
        ax1.plot(fpr, tpr, label=name)
        frac, mean_pred = calibration_curve(test[TARGET], p, n_bins=10, strategy="quantile")
        ax2.plot(mean_pred, frac, marker="o", label=name)
    ax1.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax1.set(title="ROC (holdout)", xlabel="False positive rate", ylabel="True positive rate")
    ax2.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax2.set(title="Calibration (holdout)", xlabel="Mean predicted PD", ylabel="Observed default rate")
    for ax in (ax1, ax2):
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "model_comparison.png", dpi=130)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--warehouse", default="data/warehouse.duckdb")
    ap.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    args = ap.parse_args()
    warehouse = Path(args.warehouse)
    OUT.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(args.tracking_uri)

    print("1/5 ingest + validate")
    dq = ingest(warehouse)
    print("2/5 dbt build (staging, features, data tests)")
    run_dbt(warehouse)
    df = load_table(warehouse, "fct_credit_features")

    print("3/5 experiments")
    results, runs = run_experiments(df)
    _, test = split(df)

    print("4/5 registry + promotion gate")
    decisions = []
    if registry.champion_version() is None:
        # Seed the registry with the simple scorecard so every later model must beat it.
        base = runs["baseline_scorecard"]
        v = registry.register(base["model_uri"], {"experiment": "baseline_scorecard"})
        d = registry.gate(test[TARGET], base["pipeline"].predict_proba(test)[:, 1])
        registry.promote(v, d)
        decisions.append(("baseline_scorecard", v, d))

    challenger = pick_challenger(results)
    champ_model, champ_v = registry.load_champion()
    cand = runs[challenger]
    v = registry.register(cand["model_uri"], {"experiment": challenger})
    d = registry.gate(test[TARGET], cand["pipeline"].predict_proba(test)[:, 1],
                      champ_model.predict_proba(test)[:, 1])
    registry.promote(v, d)
    decisions.append((challenger, v, d))
    champion, champion_v = registry.load_champion()

    print("5/5 monitoring + fairness")
    train, _ = split(df)
    staging = load_table(warehouse, "stg_credit_clients")
    test_staging = staging[staging["client_id"].isin(test["client_id"])]
    stable = drift_report(train, test, champion)
    stressed_features = compute_features(stress_scenario(test_staging))
    stressed = drift_report(train, stressed_features, champion)
    fairness = fairness_audit(test, champion)

    plot(test, {k: runs[k]["pipeline"] for k in ["baseline_scorecard", "logistic_all", "lightgbm_all"]})
    importance = feature_importance(cand["pipeline"], cand["experiment"].features)

    full = results.set_index("experiment")["cv_auc"]
    ablation = results[results["role"].isin(["ablation", "audit"])][["experiment", "cv_auc"]].copy()
    ablation["delta_vs_lightgbm_all"] = ablation["cv_auc"] - full["lightgbm_all"]

    mean_pd_ref = champion.predict_proba(test)[:, 1].mean()
    mean_pd_stress = champion.predict_proba(stressed_features)[:, 1].mean()
    gate_rows = "\n".join(
        f"- **{name}** -> registry v{ver}: {'PROMOTED to champion' if dec.promote else 'kept as candidate'} ({dec.reason})"
        for name, ver, dec in decisions
    )
    report = f"""# Results

Data: UCI *Default of Credit Card Clients* (30,000 accounts, Apr-Sep 2005). Target: default
in the following month. Stratified 80/20 split (seed 42); CV = 5-fold on the training set.

## Data quality (crm.ingest)

{dq.to_markdown()}

## Experiments (tracked in MLflow)

{fmt_table(results.drop(columns=["cv_auc_std"]))}

## Ablations: CV AUC change when a feature group is removed (or demographics added)

{fmt_table(ablation)}

## Promotion gate

{gate_rows}

Current champion: **v{champion_v}**

Top features of `{challenger}` (gain):

{fmt_table(importance.head(8).rename("importance").reset_index().rename(columns={"index": "feature"}), "{:.0f}")}

## Monitoring: PSI vs training population (OK < 0.10 <= WATCH < 0.25 <= ALERT)

Holdout batch (no drift expected) - worst 5 signals:

{fmt_table(stable.head(5))}

Simulated downturn batch (balances +20%, repayments -30%, 25% of clients one month later) - worst 8 signals:

{fmt_table(stressed.head(8))}

Mean predicted PD: {mean_pd_ref:.3f} on holdout -> {mean_pd_stress:.3f} under stress.

## Fairness audit of the champion (holdout; sensitive attributes are not model inputs)

{fmt_table(fairness)}
"""
    (OUT / "results.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
