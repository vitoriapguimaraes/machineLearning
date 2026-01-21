CREATE OR REPLACE TABLE `lab025-p003.new_data.dataset_complete` AS
SELECT
  u.user_id,
  CASE WHEN u.age > 90 THEN NULL ELSE u.age END AS age,
  CASE WHEN u.sex = 'F' THEN 1 WHEN u.sex = 'M' THEN 0 END AS sex_num,
  u.last_month_salary,
  IFNULL(u.number_dependents, 0) AS number_dependents,
  IFNULL(f.total_emprestimos, 0) AS total_emprestimos,
  IFNULL(f.qtd_real_estate, 0) AS qtd_real_estate,
  IFNULL(f.qtd_others, 0) AS qtd_others,
  IFNULL(f.perc_real_estate, 0) AS perc_real_estate,
  IFNULL(f.perc_others, 0) AS perc_others,
  d.more_90_days_overdue,
  d.using_lines_not_secured_personal_assets,
  d.number_times_delayed_payment_loan_30_59_days,
  d.debt_ratio,
  d.number_times_delayed_payment_loan_60_89_days,
  df.default_flag
FROM `lab025-p003.dataset.user_info` u
LEFT JOIN `lab025-p003.new_data.loans_outstanding_features` f ON u.user_id = f.user_id
LEFT JOIN `lab025-p003.dataset.loans_detail` d ON u.user_id = d.user_id
LEFT JOIN `lab025-p003.dataset.defaut` df ON u.user_id = df.user_id
WHERE u.age <= 90 AND u.user_id != 21096;


