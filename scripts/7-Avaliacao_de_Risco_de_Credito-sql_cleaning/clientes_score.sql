CREATE OR REPLACE TABLE `lab025-p003.new_data.clientes_score` AS
WITH base AS (
  SELECT
    user_id,
    default_flag,
    age,
    last_month_salary,
    total_emprestimos,
    qtd_real_estate,
    qtd_others,
    number_times_delayed_payment_loan_30_59_days AS atraso_30_59,
    number_times_delayed_payment_loan_60_89_days AS atraso_60_89,
    more_90_days_overdue AS atraso_90_plus
  FROM `lab025-p003.new_data.dt_complete_ok`
),

quartis_joined AS (
  SELECT
    b.user_id,
    b.default_flag,
    b.age,
    b.last_month_salary,
    b.total_emprestimos,
    b.qtd_real_estate,
    b.qtd_others,
    b.atraso_30_59,
    b.atraso_60_89,
    b.atraso_90_plus,
    sq_age.q1 AS q1_age, sq_age.q2 AS q2_age, sq_age.q3 AS q3_age,
    sq_salary.q1 AS q1_salary, sq_salary.q2 AS q2_salary, sq_salary.q3 AS q3_salary,
    sq_emprestimos.q1 AS q1_emprestimos, sq_emprestimos.q2 AS q2_emprestimos, sq_emprestimos.q3 AS q3_emprestimos,
    sq_atraso30.q1 AS q1_atraso30, sq_atraso30.q2 AS q2_atraso30, sq_atraso30.q3 AS q3_atraso30,
    sq_atraso60.q1 AS q1_atraso60, sq_atraso60.q2 AS q2_atraso60, sq_atraso60.q3 AS q3_atraso60,
    sq_atraso90.q1 AS q1_atraso90, sq_atraso90.q2 AS q2_atraso90, sq_atraso90.q3 AS q3_atraso90
  FROM base b
  LEFT JOIN `lab025-p003.new_data.stats_summary` sq_age ON sq_age.var_name = 'age'
  LEFT JOIN `lab025-p003.new_data.stats_summary` sq_salary ON sq_salary.var_name = 'last_month_salary'
  LEFT JOIN `lab025-p003.new_data.stats_summary` sq_emprestimos ON sq_emprestimos.var_name = 'total_emprestimos'
  LEFT JOIN `lab025-p003.new_data.stats_summary` sq_atraso30 ON sq_atraso30.var_name = 'number_times_delayed_payment_loan_30_59_days'
  LEFT JOIN `lab025-p003.new_data.stats_summary` sq_atraso60 ON sq_atraso60.var_name = 'number_times_delayed_payment_loan_60_89_days'
  LEFT JOIN `lab025-p003.new_data.stats_summary` sq_atraso90 ON sq_atraso90.var_name = 'more_90_days_overdue'
)

SELECT
  user_id,
  default_flag,
  CASE WHEN age <= q1_age THEN 1 WHEN age <= q2_age THEN 2 WHEN age <= q3_age THEN 3 ELSE 4 END AS score_age,
  CASE WHEN last_month_salary <= q1_salary THEN 1 WHEN last_month_salary <= q2_salary THEN 2 WHEN last_month_salary <= q3_salary THEN 3 ELSE 4 END AS score_salary,
  CASE WHEN total_emprestimos <= q1_emprestimos THEN 1 WHEN total_emprestimos <= q2_emprestimos THEN 2 WHEN total_emprestimos <= q3_emprestimos THEN 3 ELSE 4 END AS score_emprestimos,
  CASE WHEN atraso_30_59 <= q1_atraso30 THEN 1 WHEN atraso_30_59 <= q2_atraso30 THEN 2 WHEN atraso_30_59 <= q3_atraso30 THEN 3 ELSE 4 END AS score_atraso_30_59,
  CASE WHEN atraso_60_89 <= q1_atraso60 THEN 1 WHEN atraso_60_89 <= q2_atraso60 THEN 2 WHEN atraso_60_89 <= q3_atraso60 THEN 3 ELSE 4 END AS score_atraso_60_89,
  CASE WHEN atraso_90_plus <= q1_atraso90 THEN 1 WHEN atraso_90_plus <= q2_atraso90 THEN 2 WHEN atraso_90_plus <= q3_atraso90 THEN 3 ELSE 4 END AS score_atraso_90_plus,
  (CASE WHEN age <= q1_age THEN 1 WHEN age <= q2_age THEN 2 WHEN age <= q3_age THEN 3 ELSE 4 END +
   CASE WHEN last_month_salary <= q1_salary THEN 1 WHEN last_month_salary <= q2_salary THEN 2 WHEN last_month_salary <= q3_salary THEN 3 ELSE 4 END +
   CASE WHEN total_emprestimos <= q1_emprestimos THEN 1 WHEN total_emprestimos <= q2_emprestimos THEN 2 WHEN total_emprestimos <= q3_emprestimos THEN 3 ELSE 4 END +
   CASE WHEN atraso_30_59 <= q1_atraso30 THEN 1 WHEN atraso_30_59 <= q2_atraso30 THEN 2 WHEN atraso_30_59 <= q3_atraso30 THEN 3 ELSE 4 END +
   CASE WHEN atraso_60_89 <= q1_atraso60 THEN 1 WHEN atraso_60_89 <= q2_atraso60 THEN 2 WHEN atraso_60_89 <= q3_atraso60 THEN 3 ELSE 4 END +
   CASE WHEN atraso_90_plus <= q1_atraso90 THEN 1 WHEN atraso_90_plus <= q2_atraso90 THEN 2 WHEN atraso_90_plus <= q3_atraso90 THEN 3 ELSE 4 END) AS score_risco
FROM quartis_joined;


