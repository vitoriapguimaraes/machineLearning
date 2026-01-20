CREATE OR REPLACE TABLE `lab025-p003.new_data.dt_full_analysis` AS
SELECT
  user_id,
  age,
  sex_num,
  last_month_salary,
  number_dependents,
  total_emprestimos,
  qtd_real_estate,
  qtd_others,
  perc_real_estate,
  perc_others,
  more_90_days_overdue,
  using_lines_not_secured_personal_assets,
  number_times_delayed_payment_loan_30_59_days,
  debt_ratio,
  number_times_delayed_payment_loan_60_89_days,
  default_flag,

  -- Categoria idade
  CASE
    WHEN age < 20 THEN '-20'
    WHEN age BETWEEN 20 AND 34 THEN '20-34'
    WHEN age BETWEEN 35 AND 49 THEN '35-49'
    WHEN age BETWEEN 50 AND 64 THEN '50-64'
    ELSE '65+'
  END AS faixa_etaria,

  -- Categoria salário (faixas podem ser ajustadas conforme seus dados)
  CASE
    WHEN last_month_salary < 2000 THEN 'D) Baixa renda (<2k)'
    WHEN last_month_salary BETWEEN 2000 AND 5000 THEN 'C) Média renda (2k-5k)'
    WHEN last_month_salary BETWEEN 5001 AND 10000 THEN 'B) Boa renda (5k-10k)'
    ELSE 'A) Alta renda (10k+)'
  END AS faixa_salarial,

  -- Categoria dependentes
  CASE
    WHEN number_dependents = 0 THEN '0: Sem dependentes'
    WHEN number_dependents BETWEEN 1 AND 2 THEN '1-2: Poucos'
    WHEN number_dependents BETWEEN 3 AND 5 THEN '3-5: Alguns'
    ELSE '6+: Muitos'
  END AS faixa_dependentes,

  -- Categoria tipo empréstimo
  CASE
    WHEN perc_real_estate >= 0.8 THEN 'Predominantemente Real Estate'
    WHEN perc_others >= 0.8 THEN 'Predominantemente Outros'
    WHEN perc_real_estate BETWEEN 0.4 AND 0.6 THEN 'Equilibrado'
    ELSE 'Misto'
  END AS loan_profile
FROM `lab025-p003.new_data.dt_complete_ok`;


