-- One row per client: behavioural features for the default model.
-- This file is also executed by the scoring API (crm.features.compute_features), so
-- training and serving share one definition of every feature.
with s as (
    select * from {{ ref('stg_credit_clients') }}
),

clipped as (
    -- Repayment status: -2/-1/0 all mean "no delay"; 1..9 = months behind.
    select
        *,
        greatest(pay_status_m1, 0) as d1,
        greatest(pay_status_m2, 0) as d2,
        greatest(pay_status_m3, 0) as d3,
        greatest(pay_status_m4, 0) as d4,
        greatest(pay_status_m5, 0) as d5,
        greatest(pay_status_m6, 0) as d6,
        -- Share of the previous month's bill that was paid (capped; 1.0 when nothing was owed).
        case when bill_amt_m2 > 0 then least(pay_amt_m1 / bill_amt_m2, 2.0) else 1.0 end as r1,
        case when bill_amt_m3 > 0 then least(pay_amt_m2 / bill_amt_m3, 2.0) else 1.0 end as r2,
        case when bill_amt_m4 > 0 then least(pay_amt_m3 / bill_amt_m4, 2.0) else 1.0 end as r3,
        case when bill_amt_m5 > 0 then least(pay_amt_m4 / bill_amt_m5, 2.0) else 1.0 end as r4,
        case when bill_amt_m6 > 0 then least(pay_amt_m5 / bill_amt_m6, 2.0) else 1.0 end as r5
    from s
)

select
    client_id,

    -- credit line
    limit_bal,
    ln(limit_bal) as log_limit_bal,

    -- payment history
    d1 as delay_m1,
    greatest(d1, d2, d3, d4, d5, d6) as max_delay_6m,
    (d1 > 0)::int + (d2 > 0)::int + (d3 > 0)::int
        + (d4 > 0)::int + (d5 > 0)::int + (d6 > 0)::int as n_months_delayed_6m,
    d1 - d3 as delay_trend_3m,
    (pay_status_m1 = -2)::int as inactive_m1,

    -- utilization
    bill_amt_m1 / limit_bal as utilization_m1,
    (bill_amt_m1 + bill_amt_m2 + bill_amt_m3 + bill_amt_m4 + bill_amt_m5 + bill_amt_m6)
        / 6.0 / limit_bal as avg_utilization_6m,
    (bill_amt_m1 - bill_amt_m3) / limit_bal as utilization_trend_3m,

    -- repayment behaviour
    r1 as repay_ratio_m1,
    (r1 + r2 + r3 + r4 + r5) / 5.0 as avg_repay_ratio_5m,
    (r1 >= 1)::int + (r2 >= 1)::int + (r3 >= 1)::int + (r4 >= 1)::int + (r5 >= 1)::int
        as n_full_payments_5m,
    ln(1 + pay_amt_m1 + pay_amt_m2 + pay_amt_m3 + pay_amt_m4 + pay_amt_m5 + pay_amt_m6)
        as log_total_paid_6m,

    -- demographics: kept for the fairness audit, never used as model inputs
    sex,
    education,
    marriage,
    age,

    default_next_month
from clipped
