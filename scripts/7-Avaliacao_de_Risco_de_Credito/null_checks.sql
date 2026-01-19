SELECT
  'defaut' AS table_name,
  COUNTIF(user_id IS NULL) AS user_id_null,
  COUNTIF(default_flag IS NULL) AS default_flag_null
FROM `lab025-p003.dataset.defaut`

UNION ALL

SELECT
  'loans_detail' AS table_name,
  COUNTIF(user_id IS NULL) AS user_id_NULL,
  COUNTIF(more_90_days_overdue IS NULL) AS more_90_days_overdue_NULL,
  COUNTIF(using_lines_not_secured_personal_assets IS NULL) AS using_lines_not_secured_personal_assets_NULL,
  COUNTIF(number_times_delayed_payment_loan_30_59_days IS NULL) AS number_times_delayed_payment_loan_30_59_days_NULL,
  COUNTIF(debt_ratio IS NULL) AS debt_ratio_NULL,
  COUNTIF(number_times_delayed_payment_loan_60_89_days IS NULL) AS number_times_delayed_payment_loan_60_89_days_NULL,
  NULL, NULL -- Para alinhar com as colunas extras de outras tabelas
FROM `lab025-p003.dataset.loans_detail`

UNION ALL

SELECT
  'loans_outstanding' AS table_name,
  COUNTIF(loan_id IS NULL) AS loan_id_NULL,
  COUNTIF(user_id IS NULL) AS user_id_NULL,
  COUNTIF(loan_type IS NULL) AS loan_type_NULL,
  NULL, NULL, NULL, NULL, NULL -- Para alinhar com as colunas extras de outras tabelas
FROM `lab025-p003.dataset.loans_outstanding`

UNION ALL

SELECT
  'user_info' AS table_name,
  COUNTIF(user_id IS NULL) AS user_id_NULL,
  COUNTIF(age IS NULL) AS age_NULL,
  COUNTIF(sex IS NULL) AS sex_NULL,
  COUNTIF(last_month_salary IS NULL) AS last_month_salary_NULL,
  COUNTIF(number_dependents IS NULL) AS number_dependents_NULL,
  NULL, NULL, NULL -- Para alinhar com as colunas extras de outras tabelas
FROM `lab025-p003.dataset.user_info`;


