import streamlit as st
import plotly.express as px
from utils.load_file import load_dataset
from utils.ui import setup_sidebar, add_back_to_top
from utils.models import train_credit_model, predict_credit

st.set_page_config(page_title="Score de Crédito", page_icon="💳", layout="wide")

setup_sidebar()
add_back_to_top()

st.title("💳 Previsão de Score de Crédito")

# Data Loading
try:
    df = load_dataset("cartao_financeiro.csv")
except Exception as e:
    st.error(f"Erro ao carregar dados: {e}")
    st.stop()


# Model Training
@st.cache_resource
def get_trained_model(data):
    return train_credit_model(data)


model_data = get_trained_model(df)

# Tabs
tab1, tab2, tab3 = st.tabs(["Visão Geral", "Análise", "Previsão"])

with tab1:
    st.subheader("Entendendo o Problema")
    st.markdown(
        """
        Este projeto visa prever o **Score de Crédito** de clientes bancários, classificando-os em categorias (Ruim, Padrão, Bom) com base em seu histórico financeiro.
        O modelo de Inteligência Artificial analisa diversos fatores para auxiliar na tomada de decisão de concessão de crédito.
        - **Dados:** Histórico financeiro de clientes.
        - **Modelo:** Random Forest Classifier.
        """
    )

    st.subheader("Amostra dos Dados")
    st.dataframe(df.head(), use_container_width=True)

with tab2:

    col_score, col_feature = st.columns(2)

    with col_score:
        fig = px.pie(
            df, names="score_credito", title="Distribuição de Score de Crédito"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_feature:
        feature_importance = model_data["feature_importance"]
        fig = px.bar(
            feature_importance.head(10),
            x="importance",
            y="feature",
            orientation="h",
            title="Top 10 Variáveis Mais Importantes",
            labels={"importance": "Importância", "feature": "Variável"},
        )
        fig.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

with tab3:

    st.metric("Acurácia do Modelo", f"{model_data['accuracy']:.2%}")

    st.subheader("Simulação de Score")
    st.markdown("Complete com informações do cliente para prever o Score.")
    st.caption(
        "A simulação tem como prioridade as características mais relevantes (como Dívida, Mix de Crédito e Juros) para estimar o Score."
    )

    default_input = df.iloc[0].to_dict()

    with st.form("prediction_form"):

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            mix_credito_options = df["mix_credito"].unique().tolist()
            mix_credito = st.selectbox(
                "Mix de Crédito",
                options=mix_credito_options,
                help="Classificação da variedade de contas de crédito (Cartões, Empréstimos, Financiamentos, etc.)",
            )
            dias_atraso = st.number_input(
                "Dias de Atraso", min_value=0, value=int(df["dias_atraso"].mean())
            )

        with col2:
            divida_total = st.number_input(
                "Dívida Total", min_value=0.0, value=float(df["divida_total"].mean())
            )
            idade = st.number_input(
                "Idade", min_value=18, max_value=100, value=int(df["idade"].mean())
            )

        with col3:
            juros_emprestimo = st.number_input(
                "Juros de Empréstimo",
                min_value=0.0,
                value=float(df["juros_emprestimo"].mean()),
            )
            salario_anual = st.number_input(
                "Salário Anual",
                min_value=0.0,
                value=float(df["salario_anual"].mean()),
            )

        with col4:
            num_cartoes = st.number_input(
                "Número de Cartões", min_value=0, value=int(df["num_cartoes"].mean())
            )
            investimento_mensal = st.number_input(
                "Investimento Mensal", value=float(df["investimento_mensal"].mean())
            )

        submit_btn = st.form_submit_button("Calcular Score", type="primary")

    if submit_btn:
        # Construct input dictionary based on form + defaults from template
        input_data = default_input.copy()

        # Update with form values
        input_data["mix_credito"] = mix_credito
        input_data["divida_total"] = divida_total
        input_data["juros_emprestimo"] = juros_emprestimo
        input_data["num_cartoes"] = num_cartoes
        input_data["dias_atraso"] = dias_atraso
        input_data["idade"] = idade
        input_data["salario_anual"] = salario_anual
        input_data["investimento_mensal"] = investimento_mensal

        # Remove target if present
        if "score_credito" in input_data:
            del input_data["score_credito"]

        prediction = predict_credit(model_data, input_data)

        translation_map = {"Good": "Bom", "Standard": "Padrão", "Poor": "Ruim"}
        score_pt = translation_map.get(prediction, prediction)

        msg = f"Score Previsto: {score_pt} ({prediction})"

        if prediction == "Good":
            st.success(msg)
        elif prediction == "Standard":
            st.info(msg)
        elif prediction == "Poor":
            st.warning(msg)
        else:
            st.write(msg)
