import streamlit as st
from utils.load_file import load_dataset
from utils.ui import setup_sidebar, add_back_to_top
from utils.visualizations import (
    show_univariate_grid,
    plot_regression,
)
from utils.models import train_salary_model, predict_salary

st.set_page_config(
    page_title="Previsão de Salário por Estudos", page_icon="🎓", layout="wide"
)

setup_sidebar()
add_back_to_top()

st.title("🎓 Previsão de Salário por Estudos")

# Data Loading
try:
    df = load_dataset("estudo_salario.csv")
except Exception as e:
    st.error(f"Erro ao carregar dados: {e}")
    st.stop()


# Model Training
@st.cache_resource
def get_trained_model(data):
    return train_salary_model(data)


model_data = get_trained_model(df)
model = model_data["model"]

# Tabs
tab_overview, tab_analysis, tab_prediction = st.tabs(
    ["Visão Geral", "Análise", "Previsão"]
)

with tab_overview:
    st.markdown(
        """
        ### Entendendo o Problema
        Este projeto tem como objetivo prever o salário mensal com base no número de horas de estudo dedicadas por mês. É utilizado um modelo de **Regressão Linear Simples** para identificar a relação entre o esforço de estudo e a recompensa financeira.
        - **Variável Independente (X):** Horas de estudo por mês.
        - **Variável Dependente (y):** Salário mensal.
        """
    )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Amostra dos Dados")
        st.dataframe(df.head(), use_container_width=True)

    with col2:
        st.subheader("Estatísticas Descritivas")
        st.dataframe(df.describe(), use_container_width=True)

with tab_analysis:
    st.subheader("Análise Exploratória")

    show_univariate_grid(
        df,
        numeric_cols=["horas_estudo_mes", "salario"],
        categorical_cols=[],
        target_col=None,
        num_cols=2,
    )

    st.subheader("Correlação e Regressão")
    plot_regression(
        df,
        x_col="horas_estudo_mes",
        y_col="salario",
        model=model,
        x_label="Horas de Estudo (mês)",
        y_label="Salário (R$)",
    )

with tab_prediction:
    st.subheader("Simulações")

    col_input, col_result = st.columns([1, 2])

    with col_input:
        st.info("Insira as horas de estudo para prever o salário.")
        hours_input = st.number_input(
            "Horas de estudo por mês:",
            min_value=1.0,
            max_value=120.0,
            value=48.0,
            step=0.5,
        )
        predict_btn = st.button("Calcular Previsão", type="primary")

    with col_result:
        if predict_btn:
            prediction = predict_salary(model, hours_input)

            st.success(f"### Salário Previsto: **R$ {prediction:,.2f}**")

            # Contextual metrics
            if prediction > df["salario"].mean():
                st.balloons()
                st.caption("🚀 Uau! Isso é acima da média salarial do dataset!")
            else:
                st.caption("Continue estudando para aumentar esse valor! 💪")

    st.divider()
    st.subheader("Métricas do Modelo")

    m1, m2, m3 = st.columns(3)
    m1.metric(
        "Coeficiente R² (Determinação)",
        f"{model_data['r2_score']:.2%}",
        help="Indica o quanto a variância do salário é explicada pelas horas de estudo.",
    )
    m2.metric(
        "Intercepto (w0)",
        f"{model.intercept_:.2f}",
        help="Valor do salário quando horas de estudo é zero.",
    )
    m3.metric(
        "Coeficiente Angular (w1)",
        f"{model.coef_[0]:.2f}",
        help="Quanto o salário aumenta para cada hora extra de estudo.",
    )
