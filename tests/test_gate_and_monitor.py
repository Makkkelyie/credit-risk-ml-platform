import numpy as np

from crm.metrics import evaluate, expected_calibration_error, ks_statistic
from crm.monitor import psi, status
from crm.registry import gate

rng = np.random.default_rng(0)
Y = rng.integers(0, 2, 3000)
WEAK = np.clip(0.5 + 0.10 * (Y - 0.5) + rng.normal(0, 0.2, 3000), 0.01, 0.99)
STRONG = np.clip(0.5 + 0.40 * (Y - 0.5) + rng.normal(0, 0.2, 3000), 0.01, 0.99)


def test_first_model_becomes_champion():
    assert gate(Y, WEAK).promote


def test_clearly_better_challenger_is_promoted():
    d = gate(Y, STRONG, WEAK)
    assert d.promote and d.auc_diff_ci[0] > 0


def test_identical_challenger_is_not_promoted():
    d = gate(Y, WEAK, WEAK)
    assert not d.promote and "not significant" in d.reason


def test_better_ranking_but_worse_calibration_is_blocked():
    miscalibrated = np.clip(STRONG + 0.45, 0, 1)  # ranks well, but PDs are far too high
    d = gate(Y, miscalibrated, WEAK)
    assert d.auc_diff_ci[0] > 0  # the AUC gain alone would have passed
    assert not d.promote and "calibration" in d.reason


def test_metric_sanity():
    m = evaluate(Y, Y.astype(float))
    assert m["auc"] == 1.0 and ks_statistic(Y, Y) == 1.0 and m["brier"] == 0.0
    assert expected_calibration_error(Y, np.full(len(Y), Y.mean())) < 1e-9


def test_psi_flags_shift_but_not_noise():
    ref = rng.normal(0, 1, 20_000)
    assert psi(ref, rng.normal(0, 1, 5_000)) < 0.02
    assert status(psi(ref, rng.normal(0.8, 1, 5_000))) == "ALERT"


def test_psi_handles_discrete_features():
    ref = rng.choice([0, 1, 2], 10_000, p=[0.8, 0.15, 0.05])
    assert psi(ref, ref.copy()) < 1e-9
