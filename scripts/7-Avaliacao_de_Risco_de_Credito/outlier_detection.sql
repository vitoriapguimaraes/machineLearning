WITH all_data AS (
  SELECT
    user_id,
    CAST(more_90_days_overdue AS FLOAT64) AS more_90_days_overdue,
    CAST(using_lines_not_secured_personal_assets AS FLOAT64) AS using_lines_not_secured_personal_assets,
    CAST(number_times_delayed_payment_loan_30_59_days AS FLOAT64) AS number_times_delayed_payment_loan_30_59_days,
    CAST(debt_ratio AS FLOAT64) AS debt_ratio,
    CAST(number_times_delayed_payment_loan_60_89_days AS FLOAT64) AS number_times_delayed_payment_loan_60_89_days,
    CAST(age AS FLOAT64) AS age,
    CAST(last_month_salary AS FLOAT64) AS last_month_salary,
    CAST(number_dependents AS FLOAT64) AS number_dependents
  FROM `lab025-p003.new_data.dt_complete_ok`
),

quartiles AS (
  SELECT
    APPROX_QUANTILES(more_90_days_overdue, 4)[OFFSET(1)] AS q1_more_90_days_overdue,
    APPROX_QUANTILES(more_90_days_overdue, 4)[OFFSET(3)] AS q3_more_90_days_overdue,
    APPROX_QUANTILES(using_lines_not_secured_personal_assets, 4)[OFFSET(1)] AS q1_using_lines_not_secured_personal_assets,
    APPROX_QUANTILES(using_lines_not_secured_personal_assets, 4)[OFFSET(3)] AS q3_using_lines_not_secured_personal_assets,
    APPROX_QUANTILES(number_times_delayed_payment_loan_30_59_days, 4)[OFFSET(1)] AS q1_number_times_delayed_payment_loan_30_59_days,
    APPROX_QUANTILES(number_times_delayed_payment_loan_30_59_days, 4)[OFFSET(3)] AS q3_number_times_delayed_payment_loan_30_59_days,
    APPROX_QUANTILES(debt_ratio, 4)[OFFSET(1)] AS q1_debt_ratio,
    APPROX_QUANTILES(debt_ratio, 4)[OFFSET(3)] AS q3_debt_ratio,
    APPROX_QUANTILES(number_times_delayed_payment_loan_60_89_days, 4)[OFFSET(1)] AS q1_number_times_delayed_payment_loan_60_89_days,
    APPROX_QUANTILES(number_times_delayed_payment_loan_60_89_days, 4)[OFFSET(3)] AS q3_number_times_delayed_payment_loan_60_89_days,
    APPROX_QUANTILES(age, 4)[OFFSET(1)] AS q1_age,
    APPROX_QUANTILES(age, 4)[OFFSET(3)] AS q3_age,
    APPROX_QUANTILES(last_month_salary, 4)[OFFSET(1)] AS q1_last_month_salary,
    APPROX_QUANTILES(last_month_salary, 4)[OFFSET(3)] AS q3_last_month_salary,
    APPROX_QUANTILES(number_dependents, 4)[OFFSET(1)] AS q1_number_dependents,
    APPROX_QUANTILES(number_dependents, 4)[OFFSET(3)] AS q3_number_dependents
  FROM all_data
)

SELECT
  a.user_id,
  a.more_90_days_overdue,
  (a.more_90_days_overdue < q.q1_more_90_days_overdue - 1.5 * (q.q3_more_90_days_overdue - q.q1_more_90_days_overdue) OR
   a.more_90_days_overdue > q.q3_more_90_days_overdue + 1.5 * (q.q3_more_90_days_overdue - q.q1_more_90_days_overdue)) AS more_90_days_overdue_outlier,

  a.using_lines_not_secured_personal_assets,
  (a.using_lines_not_secured_personal_assets < q.q1_using_lines_not_secured_personal_assets - 1.5 * (q.q3_using_lines_not_secured_personal_assets - q.q1_using_lines_not_secured_personal_assets) OR
   a.using_lines_not_secured_personal_assets > q.q3_using_lines_not_secured_personal_assets + 1.5 * (q.q3_using_lines_not_secured_personal_assets - q.q1_using_lines_not_secured_personal_assets)) AS using_lines_not_secured_personal_assets_outlier,

  a.number_times_delayed_payment_loan_30_59_days,
  (a.number_times_delayed_payment_loan_30_59_days < q.q1_number_times_delayed_payment_loan_30_59_days - 1.5 * (q.q3_number_times_delayed_payment_loan_30_59_days - q.q1_number_times_delayed_payment_loan_30_59_days) OR
   a.number_times_delayed_payment_loan_30_59_days > q.q3_number_times_delayed_payment_loan_30_59_days + 1.5 * (q.q3_number_times_delayed_payment_loan_30_59_days - q.q1_number_times_delayed_payment_loan_30_59_days)) AS number_times_delayed_payment_loan_30_59_days_outlier,

  a.debt_ratio,
  (a.debt_ratio < q.q1_debt_ratio - 1.5 * (q.q3_debt_ratio - q.q1_debt_ratio) OR
   a.debt_ratio > q.q3_debt_ratio + 1.5 * (q.q3_debt_ratio - q.q1_debt_ratio)) AS debt_ratio_outlier,

  a.number_times_delayed_payment_loan_60_89_days,
  (a.number_times_delayed_payment_loan_60_89_days < q.q1_number_times_delayed_payment_loan_60_89_days - 1.5 * (q.q3_number_times_delayed_payment_loan_60_89_days - q.q1_number_times_delayed_payment_loan_60_89_days) OR
   a.number_times_delayed_payment_loan_60_89_days > q.q3_number_times_delayed_payment_loan_60_89_days + 1.5 * (q.q3_number_times_delayed_payment_loan_60_89_days - q.q1_number_times_delayed_payment_loan_60_89_days)) AS number_times_delayed_payment_loan_60_89_days_outlier,

  a.age,
  (a.age < q.q1_age - 1.5 * (q.q3_age - q.q1_age) OR
   a.age > q.q3_age + 1.5 * (q.q3_age - q.q1_age)) AS age_outlier,

  a.last_month_salary,
  (a.last_month_salary < q.q1_last_month_salary - 1.5 * (q.q3_last_month_salary - q.q1_last_month_salary) OR
   a.last_month_salary > q.q3_last_month_salary + 1.5 * (q.q3_last_month_salary - q.q1_last_month_salary)) AS last_month_salary_outlier,

  a.number_dependents,
  (a.number_dependents < q.q1_number_dependents - 1.5 * (q.q3_number_dependents - q.q1_number_dependents) OR
   a.number_dependents > q.q3_number_dependents + 1.5 * (q.q3_number_dependents - q.q1_number_dependents)) AS number_dependents_outlier

FROM all_data a
CROSS JOIN quartiles q
WHERE
   (a.more_90_days_overdue < q.q1_more_90_days_overdue - 1.5 * (q.q3_more_90_days_overdue - q.q1_more_90_days_overdue) OR
    a.more_90_days_overdue > q.q3_more_90_days_overdue + 1.5 * (q.q3_more_90_days_overdue - q.q1_more_90_days_overdue)) OR
   (a.using_lines_not_secured_personal_assets < q.q1_using_lines_not_secured_personal_assets - 1.5 * (q.q3_using_lines_not_secured_personal_assets - q.q1_using_lines_not_secured_personal_assets) OR
    a.using_lines_not_secured_personal_assets > q.q3_using_lines_not_secured_personal_assets + 1.5 * (q.q3_using_lines_not_secured_personal_assets - q.q1_using_lines_not_secured_personal_assets)) OR
   (a.number_times_delayed_payment_loan_30_59_days < q.q1_number_times_delayed_payment_loan_30_59_days - 1.5 * (q.q3_number_times_delayed_payment_loan_30_59_days - q.q1_number_times_delayed_payment_loan_30_59_days) OR
    a.number_times_delayed_payment_loan_30_59_days > q.q3_number_times_delayed_payment_loan_30_59_days + 1.5 * (q.q3_number_times_delayed_payment_loan_30_59_days - q.q1_number_times_delayed_payment_loan_30_59_days)) OR
   (a.debt_ratio < q.q1_debt_ratio - 1.5 * (q.q3_debt_ratio - q.q1_debt_ratio) OR
    a.debt_ratio > q.q3_debt_ratio + 1.5 * (q.q3_debt_ratio - q.q1_debt_ratio)) OR
   (a.number_times_delayed_payment_loan_60_89_days < q.q1_number_times_delayed_payment_loan_60_89_days - 1.5 * (q.q3_number_times_delayed_payment_loan_60_89_days - q.q1_number_times_delayed_payment_loan_60_89_days) OR
    a.number_times_delayed_payment_loan_60_89_days > q.q3_number_times_delayed_payment_loan_60_89_days + 1.5 * (q.q3_number_times_delayed_payment_loan_60_89_days - q.q1_number_times_delayed_payment_loan_60_89_days)) OR
   (a.age < q.q1_age - 1.5 * (q.q3_age - q.q1_age) OR
    a.age > q.q3_age + 1.5 * (q.q3_age - q.q1_age)) OR
   (a.last_month_salary < q.q1_last_month_salary - 1.5 * (q.q3_last_month_salary - q.q1_last_month_salary) OR
    a.last_month_salary > q.q3_last_month_salary + 1.5 * (q.q3_last_month_salary - q.q1_last_month_salary)) OR
   (a.number_dependents < q.q1_number_dependents - 1.5 * (q.q3_number_dependents - q.q1_number_dependents) OR
    a.number_dependents > q.q3_number_dependents + 1.5 * (q.q3_number_dependents - q.q1_number_dependents));


