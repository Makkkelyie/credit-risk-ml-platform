-- Fails (returns rows) if any engineered feature leaves its valid range.
select client_id
from {{ ref('fct_credit_features') }}
where repay_ratio_m1 not between 0 and 2
   or avg_repay_ratio_5m not between 0 and 2
   or n_months_delayed_6m not between 0 and 6
   or n_full_payments_5m not between 0 and 5
   or max_delay_6m < delay_m1
