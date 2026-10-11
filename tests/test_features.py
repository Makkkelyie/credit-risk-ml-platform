import pandas as pd
import pytest

from crm.features import BEHAVIOURAL, compute_features, load_table


def test_dbt_build_produces_one_row_per_client(warehouse):
    fct = load_table(warehouse, "fct_credit_features")
    assert len(fct) == 400 and fct["client_id"].is_unique
    assert set(BEHAVIOURAL) <= set(fct.columns)


def test_undocumented_codes_are_folded(warehouse):
    stg = load_table(warehouse, "stg_credit_clients")
    assert set(stg["education"]) <= {1, 2, 3, 4}
    assert set(stg["marriage"]) <= {1, 2, 3}


def test_hand_computed_features():
    row = {
        "limit_bal": 10_000,
        "pay_status_m1": 2, "pay_status_m2": -1, "pay_status_m3": 0,
        "pay_status_m4": 1, "pay_status_m5": -2, "pay_status_m6": 3,
        "bill_amt_m1": 5_000, "bill_amt_m2": 4_000, "bill_amt_m3": 0,
        "bill_amt_m4": 2_000, "bill_amt_m5": 1_000, "bill_amt_m6": 0,
        "pay_amt_m1": 2_000, "pay_amt_m2": 0, "pay_amt_m3": 2_000,
        "pay_amt_m4": 1_000, "pay_amt_m5": 0, "pay_amt_m6": 0,
    }
    f = compute_features(pd.DataFrame([row])).iloc[0]
    assert f["delay_m1"] == 2
    assert f["max_delay_6m"] == 3
    assert f["n_months_delayed_6m"] == 3
    assert f["delay_trend_3m"] == 2
    assert f["utilization_m1"] == pytest.approx(0.5)
    assert f["repay_ratio_m1"] == pytest.approx(0.5)  # paid 2,000 of the 4,000 bill
    # ratios: 0.5, 1.0 (nothing owed), 1.0, 1.0, 1.0 (nothing owed)
    assert f["avg_repay_ratio_5m"] == pytest.approx(0.9)
    assert f["n_full_payments_5m"] == 4


def test_serving_path_matches_training_path(warehouse):
    """Training/serving skew check: the API's feature path reproduces the dbt table."""
    stg = load_table(warehouse, "stg_credit_clients")
    fct = load_table(warehouse, "fct_credit_features")
    served = compute_features(stg).sort_values("client_id").reset_index(drop=True)
    pd.testing.assert_frame_equal(served[BEHAVIOURAL], fct[BEHAVIOURAL], check_dtype=False)
