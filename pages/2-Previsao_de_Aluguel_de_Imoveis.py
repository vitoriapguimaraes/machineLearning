import streamlit as st
from utils.load_file import load_dataset
from utils.ui import setup_sidebar, add_back_to_top
from utils.visualizations import (
    show_univariate_grid,
    plot_regression,
)
from utils.models import train_rent_model, predict_rent

st.set_page_config(
    page_title="Previsão de Aluguel de Imóveis", page_icon="🏠", layout="wide"
)

setup_sidebar()
add_back_to_top()

st.title("🏠 Previsão de Aluguel de Imóveis")

# Data Loading
try:
    df = load_dataset("aluguel_imoveis.csv")
except Exception as e:
    st.error(f"Erro ao carregar dados: {e}")
    st.stop()


# Model Training
@st.cache_resource
def get_trained_model(data):
    return train_rent_model(data)


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
        Este projeto aplica técnicas de **Regressão Linear Simples** para investigar a relação entre a área dos imóveis (em metros quadrados) e o valor do aluguel em uma cidade.
        O objetivo é prever o valor do aluguel a partir da área do imóvel.
        - **Variável Independente (X):** Área do imóvel (m²).
        - **Variável Dependente (y):** Valor do aluguel.
        """
    )

    st.subheader("Amostra dos Dados")
    st.dataframe(df.head(), use_container_width=True)

    st.subheader("Estatísticas Descritivas")
    st.dataframe(df.describe(), use_container_width=True)

with tab_analysis:
    st.markdown("### Análise Exploratória")

    show_univariate_grid(
        df,
        numeric_cols=["area_m2", "valor_aluguel"],
        categorical_cols=[],
        target_col=None,
        num_cols=2,
    )

    st.markdown("### Correlação e Regressão")
    plot_regression(
        df,
        x_col="area_m2",
        y_col="valor_aluguel",
        model=model,
        x_label="Área (m²)",
        y_label="Valor do Aluguel (R$)",
    )

with tab_prediction:
    st.markdown("### Simulações")

    col_input, col_result = st.columns([1, 2])

    with col_input:
        st.info("Insira a área do imóvel para prever o aluguel.")
        area_input = st.number_input(
            "Área do imóvel (m²):",
            min_value=10.0,
            max_value=1000.0,
            value=50.0,
            step=1.0,
        )
        predict_btn = st.button("Calcular Previsão", type="primary")

    with col_result:
        if predict_btn:
            prediction = predict_rent(model, area_input)

            st.success(f"### Aluguel Previsto: **R$ {prediction:,.2f}**")

            # Contextual metrics
            if prediction > df["valor_aluguel"].mean():
                st.caption("ℹ️ Este valor está acima da média de aluguel do dataset.")
            else:
                st.caption(
                    "ℹ️ Este valor está abaixo ou na média de aluguel do dataset."
                )

    st.divider()
    st.subheader("Métricas do Modelo")

    m1, m2, m3 = st.columns(3)
    m1.metric(
        "Coeficiente R² (Determinação)",
        f"{model_data['r2_score']:.2%}",
        help="Indica o quanto a variância do aluguel é explicada pela área.",
    )
    m2.metric(
        "Intercepto (w0)", f"{model.intercept_:.2f}", help="Valor base do aluguel."
    )
    m3.metric(
        "Coeficiente Angular (w1)",
        f"{model.coef_[0]:.2f}",
        help="Quanto o aluguel aumenta para cada m² extra.",
    )
