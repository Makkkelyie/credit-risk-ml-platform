-- Rename to month-relative columns (m1 = September 2005, the most recent month) and
-- fold the undocumented category codes found by crm.ingest.validate into 'other'.
select
    id as client_id,
    cast(limit_bal as double) as limit_bal,
    sex,
    case when education in (1, 2, 3, 4) then education else 4 end as education,
    case when marriage in (1, 2, 3) then marriage else 3 end as marriage,
    age,

    pay_0 as pay_status_m1,
    pay_2 as pay_status_m2,
    pay_3 as pay_status_m3,
    pay_4 as pay_status_m4,
    pay_5 as pay_status_m5,
    pay_6 as pay_status_m6,

    cast(bill_amt1 as double) as bill_amt_m1,
    cast(bill_amt2 as double) as bill_amt_m2,
    cast(bill_amt3 as double) as bill_amt_m3,
    cast(bill_amt4 as double) as bill_amt_m4,
    cast(bill_amt5 as double) as bill_amt_m5,
    cast(bill_amt6 as double) as bill_amt_m6,

    -- pay_amt_mN is the amount paid in month N, i.e. towards the bill of month N+1.
    cast(pay_amt1 as double) as pay_amt_m1,
    cast(pay_amt2 as double) as pay_amt_m2,
    cast(pay_amt3 as double) as pay_amt_m3,
    cast(pay_amt4 as double) as pay_amt_m4,
    cast(pay_amt5 as double) as pay_amt_m5,
    cast(pay_amt6 as double) as pay_amt_m6,

    default_next_month
from {{ source('raw', 'credit_clients') }}
