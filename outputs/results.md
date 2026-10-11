# Results

Data: UCI *Default of Credit Card Clients* (30,000 accounts, Apr-Sep 2005). Target: default
in the following month. Stratified 80/20 split (seed 42); CV = 5-fold on the training set.

## Data quality (crm.ingest)

- Rows: 30,000
- Note: 345 rows with undocumented education codes (0/5/6) -> mapped to 'other'
- Note: 54 rows with undocumented marriage code 0 -> mapped to 'other'
- Note: 25939 rows use repayment-status codes -2/0 (not in the data dictionary; treated as 'no delay')
- Note: 1930 rows with a negative bill (credit balance) in some month
- Note: default rate 22.1%

## Experiments (tracked in MLflow)

| experiment | role | n_features | cv_auc | holdout_auc | holdout_ks | holdout_pr_auc | holdout_brier | holdout_ece |
|---|---|---|---|---|---|---|---|---|
| baseline_scorecard | baseline | 2 | 0.704 | 0.680 | 0.357 | 0.467 | 0.144 | 0.010 |
| logistic_all | challenger | 13 | 0.770 | 0.756 | 0.399 | 0.520 | 0.140 | 0.019 |
| lightgbm_all | challenger | 13 | 0.784 | 0.777 | 0.433 | 0.560 | 0.135 | 0.017 |
| lightgbm_minus_credit_line | ablation | 12 | 0.780 | 0.775 | 0.426 | 0.557 | 0.135 | 0.013 |
| lightgbm_minus_payment_history | ablation | 8 | 0.721 | 0.711 | 0.310 | 0.432 | 0.153 | 0.014 |
| lightgbm_minus_utilization | ablation | 10 | 0.779 | 0.769 | 0.425 | 0.552 | 0.136 | 0.011 |
| lightgbm_minus_repayment | ablation | 9 | 0.783 | 0.777 | 0.417 | 0.562 | 0.135 | 0.012 |
| lightgbm_plus_demographics | audit | 17 | 0.787 | 0.780 | 0.435 | 0.565 | 0.134 | 0.012 |

## Ablations: CV AUC change when a feature group is removed (or demographics added)

| experiment | cv_auc | delta_vs_lightgbm_all |
|---|---|---|
| lightgbm_minus_credit_line | 0.780 | -0.004 |
| lightgbm_minus_payment_history | 0.721 | -0.063 |
| lightgbm_minus_utilization | 0.779 | -0.005 |
| lightgbm_minus_repayment | 0.783 | -0.001 |
| lightgbm_plus_demographics | 0.787 | 0.003 |

## Promotion gate

- **baseline_scorecard** -> registry v1: PROMOTED to champion (no champion yet)
- **lightgbm_all** -> registry v2: PROMOTED to champion (AUC +0.0970 (95% CI +0.0803 to +0.1132), Brier -0.0092)

Current champion: **v2**

Top features of `lightgbm_all` (gain):

| feature | importance |
|---|---|
| delay_m1 | 29535 |
| max_delay_6m | 19531 |
| log_total_paid_6m | 7606 |
| n_months_delayed_6m | 7375 |
| utilization_m1 | 5912 |
| avg_utilization_6m | 4625 |
| utilization_trend_3m | 4522 |
| log_limit_bal | 4211 |

## Monitoring: PSI vs training population (OK < 0.10 <= WATCH < 0.25 <= ALERT)

Holdout batch (no drift expected) - worst 5 signals:

| signal | psi | status |
|---|---|---|
| avg_utilization_6m | 0.002 | OK |
| log_total_paid_6m | 0.002 | OK |
| log_limit_bal | 0.002 | OK |
| avg_repay_ratio_5m | 0.002 | OK |
| repay_ratio_m1 | 0.001 | OK |

Simulated downturn batch (balances +20%, repayments -30%, 25% of clients one month later) - worst 8 signals:

| signal | psi | status |
|---|---|---|
| repay_ratio_m1 | 0.992 | ALERT |
| avg_repay_ratio_5m | 0.603 | ALERT |
| n_full_payments_5m | 0.289 | ALERT |
| utilization_m1 | 0.240 | WATCH |
| max_delay_6m | 0.226 | WATCH |
| delay_trend_3m | 0.199 | WATCH |
| delay_m1 | 0.179 | WATCH |
| model_score (PD) | 0.126 | WATCH |

Mean predicted PD: 0.221 on holdout -> 0.254 under stress.

## Fairness audit of the champion (holdout; sensitive attributes are not model inputs)

| attribute | group | n | auc | default_rate | mean_pd | flag_rate |
|---|---|---|---|---|---|---|
| sex | female | 3598 | 0.777 | 0.213 | 0.216 | 0.122 |
| sex | male | 2402 | 0.776 | 0.234 | 0.228 | 0.130 |
| age band | <30 | 1912 | 0.783 | 0.223 | 0.237 | 0.152 |
| age band | 30-44 | 3002 | 0.776 | 0.211 | 0.208 | 0.108 |
| age band | 45+ | 1086 | 0.769 | 0.245 | 0.228 | 0.127 |
