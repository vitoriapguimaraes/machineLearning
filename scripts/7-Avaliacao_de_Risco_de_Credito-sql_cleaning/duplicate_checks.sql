SELECT
  'defaut' AS table_name,
  user_id, default_flag,
  COUNT (*) AS quant_freq
FROM `lab025-p003.dataset.defaut`
GROUP BY
  user_id, default_flag
HAVING COUNT(*) > 1

UNION ALL

SELECT
  'loans_detail' AS table_name,
  user_id, NULL, -- default_flag não existe em loans_detail
  COUNT(*) AS quant_freq
FROM `lab025-p003.dataset.loans_detail`
GROUP BY
  user_id,
  more_90_days_overdue,
  using_lines_not_secured_personal_assets,
  number_times_delayed_payment_loan_30_59_days,
  debt_ratio,
  number_times_delayed_payment_loan_60_89_days
HAVING COUNT(*) > 1

UNION ALL

SELECT
  'loans_outstanding' AS table_name,
  user_id, NULL, -- default_flag não existe em loans_outstanding
  COUNT(*) AS quant_freq
FROM `lab025-p003.dataset.loans_outstanding`
GROUP BY
  loan_id, user_id, loan_type
HAVING COUNT(*) > 1

UNION ALL

SELECT
  'user_info' AS table_name,
  user_id, NULL, -- default_flag não existe em user_info
  COUNT(*) AS quant_freq
FROM `lab025-p003.dataset.user_info`
GROUP BY
  user_id, age, sex, last_month_salary, number_dependents
HAVING COUNT(*) > 1;


