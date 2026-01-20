CREATE OR REPLACE TABLE `lab025-p003.new_data.stats_summary` AS
SELECT
  'more_90_days_overdue' AS var_name,
  MIN(CAST(more_90_days_overdue AS FLOAT64)) AS min_val,
  MAX(CAST(more_90_days_overdue AS FLOAT64)) AS max_val,
  AVG(CAST(more_90_days_overdue AS FLOAT64)) AS media,
  STDDEV(CAST(more_90_days_overdue AS FLOAT64)) AS desvio_padrao,
  VARIANCE(CAST(more_90_days_overdue AS FLOAT64)) AS variancia,
  APPROX_QUANTILES(CAST(more_90_days_overdue AS FLOAT64), 4)[OFFSET(1)] AS q1,
  APPROX_QUANTILES(CAST(more_90_days_overdue AS FLOAT64), 4)[OFFSET(2)] AS q2,
  APPROX_QUANTILES(CAST(more_90_days_overdue AS FLOAT64), 4)[OFFSET(3)] AS q3,
  APPROX_QUANTILES(CAST(more_90_days_overdue AS FLOAT64), 100)[OFFSET(50)] AS mediana,
  (SELECT t.more_90_days_overdue FROM `lab025-p003.new_data.dt_complete_ok` t GROUP BY t.more_90_days_overdue ORDER BY COUNT(*) DESC LIMIT 1) AS moda
FROM `lab025-p003.new_data.dt_complete_ok`

UNION ALL

SELECT
  'using_lines_not_secured_personal_assets' AS var_name,
  MIN(CAST(using_lines_not_secured_personal_assets AS FLOAT64)) AS min_val,
  MAX(CAST(using_lines_not_secured_personal_assets AS FLOAT64)) AS max_val,
  AVG(CAST(using_lines_not_secured_personal_assets AS FLOAT64)) AS media,
  STDDEV(CAST(using_lines_not_secured_personal_assets AS FLOAT64)) AS desvio_padrao,
  VARIANCE(CAST(using_lines_not_secured_personal_assets AS FLOAT64)) AS variancia,
  APPROX_QUANTILES(CAST(using_lines_not_secured_personal_assets AS FLOAT64), 4)[OFFSET(1)] AS q1,
  APPROX_QUANTILES(CAST(using_lines_not_secured_personal_assets AS FLOAT64), 4)[OFFSET(2)] AS q2,
  APPROX_QUANTILES(CAST(using_lines_not_secured_personal_assets AS FLOAT64), 4)[OFFSET(3)] AS q3,
  APPROX_QUANTILES(CAST(using_lines_not_secured_personal_assets AS FLOAT64), 100)[OFFSET(50)] AS mediana,
  (SELECT t.using_lines_not_secured_personal_assets FROM `lab025-p003.new_data.dt_complete_ok` t GROUP BY t.using_lines_not_secured_personal_assets ORDER BY COUNT(*) DESC LIMIT 1) AS moda
FROM `lab025-p003.new_data.dt_complete_ok`

UNION ALL

SELECT
  'number_times_delayed_payment_loan_30_59_days' AS var_name,
  MIN(CAST(number_times_delayed_payment_loan_30_59_days AS FLOAT64)) AS min_val,
  MAX(CAST(number_times_delayed_payment_loan_30_59_days AS FLOAT64)) AS max_val,
  AVG(CAST(number_times_delayed_payment_loan_30_59_days AS FLOAT64)) AS media,
  STDDEV(CAST(number_times_delayed_payment_loan_30_59_days AS FLOAT64)) AS desvio_padrao,
  VARIANCE(CAST(number_times_delayed_payment_loan_30_59_days AS FLOAT64)) AS variancia,
  APPROX_QUANTILES(CAST(number_times_delayed_payment_loan_30_59_days AS FLOAT64), 4)[OFFSET(1)] AS q1,
  APPROX_QUANTILES(CAST(number_times_delayed_payment_loan_30_59_days AS FLOAT64), 4)[OFFSET(2)] AS q2,
  APPROX_QUANTILES(CAST(number_times_delayed_payment_loan_30_59_days AS FLOAT64), 4)[OFFSET(3)] AS q3,
  APPROX_QUANTILES(CAST(number_times_delayed_payment_loan_30_59_days AS FLOAT64), 100)[OFFSET(50)] AS mediana,
  (SELECT t.number_times_delayed_payment_loan_30_59_days FROM `lab025-p003.new_data.dt_complete_ok` t GROUP BY t.number_times_delayed_payment_loan_30_59_days ORDER BY COUNT(*) DESC LIMIT 1) AS moda
FROM `lab025-p003.new_data.dt_complete_ok`

UNION ALL

SELECT
  'debt_ratio' AS var_name,
  MIN(CAST(debt_ratio AS FLOAT64)) AS min_val,
  MAX(CAST(debt_ratio AS FLOAT64)) AS max_val,
  AVG(CAST(debt_ratio AS FLOAT64)) AS media,
  STDDEV(CAST(debt_ratio AS FLOAT64)) AS desvio_padrao,
  VARIANCE(CAST(debt_ratio AS FLOAT64)) AS variancia,
  APPROX_QUANTILES(CAST(debt_ratio AS FLOAT64), 4)[OFFSET(1)] AS q1,
  APPROX_QUANTILES(CAST(debt_ratio AS FLOAT64), 4)[OFFSET(2)] AS q2,
  APPROX_QUANTILES(CAST(debt_ratio AS FLOAT64), 4)[OFFSET(3)] AS q3,
  APPROX_QUANTILES(CAST(debt_ratio AS FLOAT64), 100)[OFFSET(50)] AS mediana,
  (SELECT t.debt_ratio FROM `lab025-p003.new_data.dt_complete_ok` t GROUP BY t.debt_ratio ORDER BY COUNT(*) DESC LIMIT 1) AS moda
FROM `lab025-p003.new_data.dt_complete_ok`

UNION ALL

SELECT
  'number_times_delayed_payment_loan_60_89_days' AS var_name,
  MIN(CAST(number_times_delayed_payment_loan_60_89_days AS FLOAT64)) AS min_val,
  MAX(CAST(number_times_delayed_payment_loan_60_89_days AS FLOAT64)) AS max_val,
  AVG(CAST(number_times_delayed_payment_loan_60_89_days AS FLOAT64)) AS media,
  STDDEV(CAST(number_times_delayed_payment_loan_60_89_days AS FLOAT64)) AS desvio_padrao,
  VARIANCE(CAST(number_times_delayed_payment_loan_60_89_days AS FLOAT64)) AS variancia,
  APPROX_QUANTILES(CAST(number_times_delayed_payment_loan_60_89_days AS FLOAT64), 4)[OFFSET(1)] AS q1,
  APPROX_QUANTILES(CAST(number_times_delayed_payment_loan_60_89_days AS FLOAT64), 4)[OFFSET(2)] AS q2,
  APPROX_QUANTILES(CAST(number_times_delayed_payment_loan_60_89_days AS FLOAT64), 4)[OFFSET(3)] AS q3,
  APPROX_QUANTILES(CAST(number_times_delayed_payment_loan_60_89_days AS FLOAT64), 100)[OFFSET(50)] AS mediana,
  (SELECT t.number_times_delayed_payment_loan_60_89_days FROM `lab025-p003.new_data.dt_complete_ok` t GROUP BY t.number_times_delayed_payment_loan_60_89_days ORDER BY COUNT(*) DESC LIMIT 1) AS moda
FROM `lab025-p003.new_data.dt_complete_ok`

UNION ALL

SELECT
  'total_emprestimos' AS var_name,
  MIN(CAST(total_emprestimos AS FLOAT64)) AS min_val,
  MAX(CAST(total_emprestimos AS FLOAT64)) AS max_val,
  AVG(CAST(total_emprestimos AS FLOAT64)) AS media,
  STDDEV(CAST(total_emprestimos AS FLOAT64)) AS desvio_padrao,
  VARIANCE(CAST(total_emprestimos AS FLOAT64)) AS variancia,
  APPROX_QUANTILES(CAST(total_emprestimos AS FLOAT64), 4)[OFFSET(1)] AS q1,
  APPROX_QUANTILES(CAST(total_emprestimos AS FLOAT64), 4)[OFFSET(2)] AS q2,
  APPROX_QUANTILES(CAST(total_emprestimos AS FLOAT64), 4)[OFFSET(3)] AS q3,
  APPROX_QUANTILES(CAST(total_emprestimos AS FLOAT64), 100)[OFFSET(50)] AS mediana,
  (SELECT t.total_emprestimos FROM `lab025-p003.new_data.dt_complete_ok` t GROUP BY t.total_emprestimos ORDER BY COUNT(*) DESC LIMIT 1) AS moda
FROM `lab025-p003.new_data.dt_complete_ok`

UNION ALL

SELECT
  'qtd_real_estate' AS var_name,
  MIN(CAST(qtd_real_estate AS FLOAT64)) AS min_val,
  MAX(CAST(qtd_real_estate AS FLOAT64)) AS max_val,
  AVG(CAST(qtd_real_estate AS FLOAT64)) AS media,
  STDDEV(CAST(qtd_real_estate AS FLOAT64)) AS desvio_padrao,
  VARIANCE(CAST(qtd_real_estate AS FLOAT64)) AS variancia,
  APPROX_QUANTILES(CAST(qtd_real_estate AS FLOAT64), 4)[OFFSET(1)] AS q1,
  APPROX_QUANTILES(CAST(qtd_real_estate AS FLOAT64), 4)[OFFSET(2)] AS q2,
  APPROX_QUANTILES(CAST(qtd_real_estate AS FLOAT64), 4)[OFFSET(3)] AS q3,
  APPROX_QUANTILES(CAST(qtd_real_estate AS FLOAT64), 100)[OFFSET(50)] AS mediana,
  (SELECT t.qtd_real_estate FROM `lab025-p003.new_data.dt_complete_ok` t GROUP BY t.qtd_real_estate ORDER BY COUNT(*) DESC LIMIT 1) AS moda
FROM `lab025-p003.new_data.dt_complete_ok`

UNION ALL

SELECT
  'qtd_others' AS var_name,
  MIN(CAST(qtd_others AS FLOAT64)) AS min_val,
  MAX(CAST(qtd_others AS FLOAT64)) AS max_val,
  AVG(CAST(qtd_others AS FLOAT64)) AS media,
  STDDEV(CAST(qtd_others AS FLOAT64)) AS desvio_padrao,
  VARIANCE(CAST(qtd_others AS FLOAT64)) AS variancia,
  APPROX_QUANTILES(CAST(qtd_others AS FLOAT64), 4)[OFFSET(1)] AS q1,
  APPROX_QUANTILES(CAST(qtd_others AS FLOAT64), 4)[OFFSET(2)] AS q2,
  APPROX_QUANTILES(CAST(qtd_others AS FLOAT64), 4)[OFFSET(3)] AS q3,
  APPROX_QUANTILES(CAST(qtd_others AS FLOAT64), 100)[OFFSET(50)] AS mediana,
  (SELECT t.qtd_others FROM `lab025-p003.new_data.dt_complete_ok` t GROUP BY t.qtd_others ORDER BY COUNT(*) DESC LIMIT 1) AS moda
FROM `lab025-p003.new_data.dt_complete_ok`

UNION ALL

SELECT
  'perc_real_estate' AS var_name,
  MIN(CAST(perc_real_estate AS FLOAT64)) AS min_val,
  MAX(CAST(perc_real_estate AS FLOAT64)) AS max_val,
  AVG(CAST(perc_real_estate AS FLOAT64)) AS media,
  STDDEV(CAST(perc_real_estate AS FLOAT64)) AS desvio_padrao,
  VARIANCE(CAST(perc_real_estate AS FLOAT64)) AS variancia,
  APPROX_QUANTILES(CAST(perc_real_estate AS FLOAT64), 4)[OFFSET(1)] AS q1,
  APPROX_QUANTILES(CAST(perc_real_estate AS FLOAT64), 4)[OFFSET(2)] AS q2,
  APPROX_QUANTILES(CAST(perc_real_estate AS FLOAT64), 4)[OFFSET(3)] AS q3,
  APPROX_QUANTILES(CAST(perc_real_estate AS FLOAT64), 100)[OFFSET(50)] AS mediana,
  (SELECT t.perc_real_estate FROM `lab025-p003.new_data.dt_complete_ok` t GROUP BY t.perc_real_estate ORDER BY COUNT(*) DESC LIMIT 1) AS moda
FROM `lab025-p003.new_data.dt_complete_ok`

UNION ALL

SELECT
  'perc_others' AS var_name,
  MIN(CAST(perc_others AS FLOAT64)) AS min_val,
  MAX(CAST(perc_others AS FLOAT64)) AS max_val,
  AVG(CAST(perc_others AS FLOAT64)) AS media,
  STDDEV(CAST(perc_others AS FLOAT64)) AS desvio_padrao,
  VARIANCE(CAST(perc_others AS FLOAT64)) AS variancia,
  APPROX_QUANTILES(CAST(perc_others AS FLOAT64), 4)[OFFSET(1)] AS q1,
  APPROX_QUANTILES(CAST(perc_others AS FLOAT64), 4)[OFFSET(2)] AS q2,
  APPROX_QUANTILES(CAST(perc_others AS FLOAT64), 4)[OFFSET(3)] AS q3,
  APPROX_QUANTILES(CAST(perc_others AS FLOAT64), 100)[OFFSET(50)] AS mediana,
  (SELECT t.perc_others FROM `lab025-p003.new_data.dt_complete_ok` t GROUP BY t.perc_others ORDER BY COUNT(*) DESC LIMIT 1) AS moda
FROM `lab025-p003.new_data.dt_complete_ok`

UNION ALL

SELECT
  'age' AS var_name,
  MIN(CAST(age AS FLOAT64)) AS min_val,
  MAX(CAST(age AS FLOAT64)) AS max_val,
  AVG(CAST(age AS FLOAT64)) AS media,
  STDDEV(CAST(age AS FLOAT64)) AS desvio_padrao,
  VARIANCE(CAST(age AS FLOAT64)) AS variancia,
  APPROX_QUANTILES(CAST(age AS FLOAT64), 4)[OFFSET(1)] AS q1,
  APPROX_QUANTILES(CAST(age AS FLOAT64), 4)[OFFSET(2)] AS q2,
  APPROX_QUANTILES(CAST(age AS FLOAT64), 4)[OFFSET(3)] AS q3,
  APPROX_QUANTILES(CAST(age AS FLOAT64), 100)[OFFSET(50)] AS mediana,
  (SELECT t.age FROM `lab025-p003.new_data.dt_complete_ok` t GROUP BY t.age ORDER BY COUNT(*) DESC LIMIT 1) AS moda
FROM `lab025-p003.new_data.dt_complete_ok`

UNION ALL

SELECT
  'sex_num' AS var_name,
  MIN(CAST(sex_num AS FLOAT64)) AS min_val,
  MAX(CAST(sex_num AS FLOAT64)) AS max_val,
  AVG(CAST(sex_num AS FLOAT64)) AS media,
  STDDEV(CAST(sex_num AS FLOAT64)) AS desvio_padrao,
  VARIANCE(CAST(sex_num AS FLOAT64)) AS variancia,
  APPROX_QUANTILES(CAST(sex_num AS FLOAT64), 4)[OFFSET(1)] AS q1,
  APPROX_QUANTILES(CAST(sex_num AS FLOAT64), 4)[OFFSET(2)] AS q2,
  APPROX_QUANTILES(CAST(sex_num AS FLOAT64), 4)[OFFSET(3)] AS q3,
  APPROX_QUANTILES(CAST(sex_num AS FLOAT64), 100)[OFFSET(50)] AS mediana,
  (SELECT t.sex_num FROM `lab025-p003.new_data.dt_complete_ok` t GROUP BY t.sex_num ORDER BY COUNT(*) DESC LIMIT 1) AS moda
FROM `lab025-p003.new_data.dt_complete_ok`

UNION ALL

SELECT
  'last_month_salary' AS var_name,
  MIN(CAST(last_month_salary AS FLOAT64)) AS min_val,
  MAX(CAST(last_month_salary AS FLOAT64)) AS max_val,
  AVG(CAST(last_month_salary AS FLOAT64)) AS media,
  STDDEV(CAST(last_month_salary AS FLOAT64)) AS desvio_padrao,
  VARIANCE(CAST(last_month_salary AS FLOAT64)) AS variancia,
  APPROX_QUANTILES(CAST(last_month_salary AS FLOAT64), 4)[OFFSET(1)] AS q1,
  APPROX_QUANTILES(CAST(last_month_salary AS FLOAT64), 4)[OFFSET(2)] AS q2,
  APPROX_QUANTILES(CAST(last_month_salary AS FLOAT64), 4)[OFFSET(3)] AS q3,
  APPROX_QUANTILES(CAST(last_month_salary AS FLOAT64), 100)[OFFSET(50)] AS mediana,
  (SELECT t.last_month_salary FROM `lab025-p003.new_data.dt_complete_ok` t GROUP BY t.last_month_salary ORDER BY COUNT(*) DESC LIMIT 1) AS moda
FROM `lab025-p003.new_data.dt_complete_ok`

UNION ALL

SELECT
  'number_dependents' AS var_name,
  MIN(CAST(number_dependents AS FLOAT64)) AS min_val,
  MAX(CAST(number_dependents AS FLOAT64)) AS max_val,
  AVG(CAST(number_dependents AS FLOAT64)) AS media,
  STDDEV(CAST(number_dependents AS FLOAT64)) AS desvio_padrao,
  VARIANCE(CAST(number_dependents AS FLOAT64)) AS variancia,
  APPROX_QUANTILES(CAST(number_dependents AS FLOAT64), 4)[OFFSET(1)] AS q1,
  APPROX_QUANTILES(CAST(number_dependents AS FLOAT64), 4)[OFFSET(2)] AS q2,
  APPROX_QUANTILES(CAST(number_dependents AS FLOAT64), 4)[OFFSET(3)] AS q3,
  APPROX_QUANTILES(CAST(number_dependents AS FLOAT64), 100)[OFFSET(50)] AS mediana,
  (SELECT t.number_dependents FROM `lab025-p003.new_data.dt_complete_ok` t GROUP BY t.number_dependents ORDER BY COUNT(*) DESC LIMIT 1) AS moda
FROM `lab025-p003.new_data.dt_complete_ok`;


