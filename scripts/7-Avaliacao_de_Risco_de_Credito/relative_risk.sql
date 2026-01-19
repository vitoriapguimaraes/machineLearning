WITH quartis_sal AS (
  SELECT q1, q2, q3
  FROM `lab025-p003.new_data.stats_summary`
  WHERE var_name = 'last_month_salary'
),

categorias AS (
  SELECT
    user_id,
    default_flag,
    CASE
      WHEN last_month_salary <= (SELECT q1 FROM quartis_sal) THEN 'Q1'
      WHEN last_month_salary <= (SELECT q2 FROM quartis_sal) THEN 'Q2'
      WHEN last_month_salary <= (SELECT q3 FROM quartis_sal) THEN 'Q3'
      ELSE 'Q4'
    END AS faixa_salario
  FROM `lab025-p003.new_data.dt_complete_ok`
),

contagens AS (
  SELECT
    faixa_salario,
    COUNT(*) AS total,
    SUM(CASE WHEN default_flag = 1 THEN 1 ELSE 0 END) AS inadimplentes
  FROM categorias
  GROUP BY faixa_salario
),

taxas AS (
  SELECT
    faixa_salario,
    inadimplentes,
    total,
    SAFE_DIVIDE(inadimplentes, total) AS taxa_default
  FROM contagens
),

risco_relativo AS (
  SELECT
    t1.faixa_salario,
    t1.taxa_default,
    SAFE_DIVIDE(
      t1.taxa_default,
      (SELECT SAFE_DIVIDE(SUM(inadimplentes), SUM(total)) FROM taxas t2 WHERE t2.faixa_salario != t1.faixa_salario)
    ) AS risco_relativo
  FROM taxas t1
)

SELECT * FROM risco_relativo;


