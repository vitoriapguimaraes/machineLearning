CREATE OR REPLACE TABLE `lab025-p003.new_data.loans_outstanding_features` AS
SELECT
  user_id,
  COUNT(DISTINCT loan_id) AS total_emprestimos,
  COUNTIF(UPPER(TRIM(loan_type)) = 'REAL ESTATE') AS qtd_real_estate,
  COUNTIF(UPPER(TRIM(loan_type)) IN ('OTHER', 'OTHERS')) AS qtd_others,
  SAFE_DIVIDE(COUNTIF(UPPER(TRIM(loan_type)) = 'REAL ESTATE'), COUNT(DISTINCT loan_id)) AS perc_real_estate,
  SAFE_DIVIDE(COUNTIF(UPPER(TRIM(loan_type)) IN ('OTHER', 'OTHERS')), COUNT(DISTINCT loan_id)) AS perc_others
FROM `lab025-p003.dataset.loans_outstanding`
GROUP BY user_id;


