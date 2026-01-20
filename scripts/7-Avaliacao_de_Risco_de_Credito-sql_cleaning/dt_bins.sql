CREATE OR REPLACE TABLE `lab025-p003.new_data.dt_bins_auto` AS
WITH stats AS (
  SELECT
    MIN(age) AS min_age, MAX(age) AS max_age,
    MIN(last_month_salary) AS min_salary, MAX(last_month_salary) AS max_salary,
    MIN(debt_ratio) AS min_debt_ratio, MAX(debt_ratio) AS max_debt_ratio,
    MIN(more_90_days_overdue) AS min_overdue90, MAX(more_90_days_overdue) AS max_overdue90,
    MIN(number_times_delayed_payment_loan_30_59_days) AS min_delay30_59, MAX(number_times_delayed_payment_loan_30_59_days) AS max_delay30_59,
    MIN(number_times_delayed_payment_loan_60_89_days) AS min_delay60_89, MAX(number_times_delayed_payment_loan_60_89_days) AS max_delay60_89,
    MIN(total_emprestimos) AS min_total, MAX(total_emprestimos) AS max_total,
    MIN(qtd_real_estate) AS min_real, MAX(qtd_real_estate) AS max_real,
    MIN(qtd_others) AS min_others, MAX(qtd_others) AS max_others
  FROM `lab025-p003.new_data.dt_complete_ok`
),
bin_sizes AS (
  SELECT
    (max_age - min_age) / 15 AS age_bin_size,
    (max_salary - min_salary) / 15 AS salary_bin_size,
    (max_debt_ratio - min_debt_ratio) / 15 AS debt_ratio_bin_size,
    (max_overdue90 - min_overdue90) / 15 AS overdue90_bin_size,
    (max_delay30_59 - min_delay30_59) / 15 AS delay30_59_bin_size,
    (max_delay60_89 - min_delay60_89) / 15 AS delay60_89_bin_size,
    (max_total - min_total) / 15 AS total_loans_bin_size,
    (max_real - min_real) / 15 AS real_estate_bin_size,
    (max_others - min_others) / 15 AS others_bin_size
  FROM stats
)
SELECT
  a.user_id,

  -- Idade
  CONCAT(
    CAST(FLOOR((a.age - s.min_age) / bs.age_bin_size) * bs.age_bin_size + s.min_age AS INT64),
    ' - ',
    CAST(FLOOR((a.age - s.min_age) / bs.age_bin_size) * bs.age_bin_size + s.min_age + bs.age_bin_size - 1 AS INT64)
  ) AS age_bin,

  -- Salário
  CONCAT(
    CAST(FLOOR((a.last_month_salary - s.min_salary) / bs.salary_bin_size) * bs.salary_bin_size + s.min_salary AS INT64),
    ' - ',
    CAST(FLOOR((a.last_month_salary - s.min_salary) / bs.salary_bin_size) * bs.salary_bin_size + s.min_salary + bs.salary_bin_size - 1 AS INT64)
  ) AS salary_bin,

  -- Debt Ratio
  CONCAT(
    CAST(FLOOR((a.debt_ratio - s.min_debt_ratio) / bs.debt_ratio_bin_size) * bs.debt_ratio_bin_size + s.min_debt_ratio AS FLOAT64),
    ' - ',
    CAST(FLOOR((a.debt_ratio - s.min_debt_ratio) / bs.debt_ratio_bin_size) * bs.debt_ratio_bin_size + s.min_debt_ratio + bs.debt_ratio_bin_size AS FLOAT64)
  ) AS debt_ratio_bin,

  -- Overdue 90
  CONCAT(
    CAST(FLOOR((a.more_90_days_overdue - s.min_overdue90) / bs.overdue90_bin_size) * bs.overdue90_bin_size + s.min_overdue90 AS INT64),
    ' - ',
    CAST(FLOOR((a.more_90_days_overdue - s.min_overdue90) / bs.overdue90_bin_size) * bs.overdue90_bin_size + s.min_overdue90 + bs.overdue90_bin_size - 1 AS INT64)
  ) AS overdue90_bin,

  -- Delay 30-59
  CONCAT(
    CAST(FLOOR((a.number_times_delayed_payment_loan_30_59_days - s.min_delay30_59) / bs.delay30_59_bin_size) * bs.delay30_59_bin_size + s.min_delay30_59 AS INT64),
    ' - ',
    CAST(FLOOR((a.number_times_delayed_payment_loan_30_59_days - s.min_delay30_59) / bs.delay30_59_bin_size) * bs.delay30_59_bin_size + s.min_delay30_59 + bs.delay30_59_bin_size - 1 AS INT64)
  ) AS delay30_59_bin,

  -- Delay 60-89
  CONCAT(
    CAST(FLOOR((a.number_times_delayed_payment_loan_60_89_days - s.min_delay60_89) / bs.delay60_89_bin_size) * bs.delay60_89_bin_size + s.min_delay60_89 AS INT64),
    ' - ',
    CAST(FLOOR((a.number_times_delayed_payment_loan_60_89_days - s.min_delay60_89) / bs.delay60_89_bin_size) * bs.delay60_89_bin_size + s.min_delay60_89 + bs.delay60_89_bin_size - 1 AS INT64)
  ) AS delay60_89_bin,

  -- Total empréstimos
  CONCAT(
    CAST(FLOOR((a.total_emprestimos - s.min_total) / bs.total_loans_bin_size) * bs.total_loans_bin_size + s.min_total AS INT64),
    ' - ',
    CAST(FLOOR((a.total_emprestimos - s.min_total) / bs.total_loans_bin_size) * bs.total_loans_bin_size + s.min_total + bs.total_loans_bin_size - 1 AS INT64)
  ) AS total_loans_bin,

  -- Empréstimos Real Estate
  CONCAT(
    CAST(FLOOR((a.qtd_real_estate - s.min_real) / bs.real_estate_bin_size) * bs.real_estate_bin_size + s.min_real AS INT64),
    ' - ',
    CAST(FLOOR((a.qtd_real_estate - s.min_real) / bs.real_estate_bin_size) * bs.real_estate_bin_size + s.min_real + bs.real_estate_bin_size - 1 AS INT64)
  ) AS real_estate_bin,

  -- Empréstimos Others
  CONCAT(
    CAST(FLOOR((a.qtd_others - s.min_others) / bs.others_bin_size) * bs.others_bin_size + s.min_others AS INT64),
    ' - ',
    CAST(FLOOR((a.qtd_others - s.min_others) / bs.others_bin_size) * bs.others_bin_size + s.min_others + bs.others_bin_size - 1 AS INT64)
  ) AS others_bin
FROM `lab025-p003.new_data.dt_complete_ok` a
CROSS JOIN stats s
CROSS JOIN bin_sizes bs;


