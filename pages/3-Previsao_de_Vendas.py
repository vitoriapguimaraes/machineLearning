import streamlit as st
from utils.load_file import load_dataset
from utils.ui import setup_sidebar, add_back_to_top
from utils.visualizations import plot_forecast
from utils.models import train_sales_model, predict_sales, evaluate_sales_model

st.set_page_config(page_title="Previsão de Vendas", page_icon="📈", layout="wide")

setup_sidebar()
add_back_to_top()

st.title("📈 Previsão de Vendas")

# Data Loading
try:
    df = load_dataset("vendas.csv")
except Exception as e:
    st.error(f"Erro ao carregar dados: {e}")
    st.stop()


# Model Training
@st.cache_resource
def get_trained_model(data):
    return train_sales_model(data)


model_data = get_trained_model(df)
series = model_data["series"]

# Tabs
tab_overview, tab_analysis, tab_prediction = st.tabs(
    ["Visão Geral", "Análise", "Previsão"]
)

with tab_overview:
    st.markdown(
        """
        ### Entendendo o Problema
        Este projeto utiliza **Suavização Exponencial Simples** para prever o total de vendas futuras com base em dados históricos.
        O objetivo é projetar as vendas para o próximo mês (Janeiro/2024).
        - **Dados:** Série temporal de vendas diárias de 2023.
        - **Modelo:** Simple Exponential Smoothing (Holt-Winters).
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
    st.markdown("### Análise da Série Temporal")

    # Plot Model Fit using Plotly
    model = model_data["model"]
    fitted_values = model.fittedvalues

    plot_forecast(
        history_series=series,
        forecast_series=fitted_values,
        title="Ajuste do Modelo: Dados Reais vs Valores Ajustados",
        history_label="Dados Reais",
        forecast_label="Valores Ajustados (Modelo)",
    )

with tab_prediction:
    st.markdown("### Previsão para Futuro")

    periods = {"1 Mês": 30, "1 Semestre": 180, "1 Ano": 365}

    for period_name, days_input in periods.items():

        with st.container(border=True):

            forecast = predict_sales(model_data, days=days_input)

            metrics = evaluate_sales_model(model_data)

            col_metric, col_plot = st.columns([1, 3])

            col_metric.subheader(f"para {period_name}")
            col_metric.metric(
                "RMSE (Erro Quadrático Médio)",
                f"{metrics['rmse']:.2f}",
                help="Raiz do Erro Quadrático Médio. Quanto menor, melhor.",
            )
            col_metric.metric(
                "MAPE (Erro Percentual Absoluto)",
                f"{metrics['mape']:.2f}%",
                help="Erro Percentual Absoluto Médio. Indica a precisão em porcentagem.",
            )

            with col_plot:
                plot_forecast(
                    history_series=series,
                    forecast_series=forecast,
                    title=f"Previsão de Vendas ({period_name})",
                )
