import streamlit as st

from utils.ui import setup_sidebar, add_back_to_top

st.set_page_config(page_title="Machine Learning", page_icon="🤖", layout="wide")

add_back_to_top()

st.title("Machine Learning")

st.info(
    "Acesse os estudos de dados com técnicas de Machine Learning, na lista abaixo ou na barra lateral"
)

st.page_link(
    "pages/1-Previsao_de_Salario_por_Estudos.py",
    label="Previsão de Salário por Estudos",
    icon="🎓",
    use_container_width=True,
)

st.page_link(
    "pages/2-Previsao_de_Aluguel_de_Imoveis.py",
    label="Previsão de Aluguel de Imóveis",
    icon="🏠",
    use_container_width=True,
)

st.page_link(
    "pages/3-Previsao_de_Vendas.py",
    label="Previsão de Vendas",
    icon="📈",
    use_container_width=True,
)

st.page_link(
    "pages/4-Score_de_Credito_dos_Clientes.py",
    label="Score de Crédito dos Clientes",
    icon="💳",
    use_container_width=True,
)

st.page_link(
    "pages/5-Robo_com_Q-Learning.py",
    label="Robô com Q-Learning",
    icon="🖥️",
    use_container_width=True,
)

st.page_link(
    "pages/6-Rotatividade_de_Funcionarios.py",
    label="Rotatividade de Funcionários",
    icon="👤",
    use_container_width=True,
)

st.page_link(
    "pages/7-Avaliacao_de_Risco_de_Credito.py",
    label="Avaliação de Risco de Crédito",
    icon="🏦",
    use_container_width=True,
)

st.markdown("---")

st.subheader("Ferramentas Utilizadas")
st.info("Python | Pandas | Plotly | Scikit-learn | XGBoost | Statsmodels | Streamlit")

st.subheader("Competências Desenvolvidas")
st.markdown(
    """
    - **Pré-processamento:** Limpeza, imputação, encoding e balanceamento de classes.
    - **Modelagem Supervisionada:** Regressão (Linear/Logística), Random Forest, XGBoost.
    - **Aprendizado por Reforço:** Q-Learning (Agentes Autônomos).
    - **Séries Temporais:** Suavização Exponencial (Holt-Winters).
    - **Avaliação de Modelos:** R², RMSE, Curva ROC/AUC, Matriz de Confusão, Precision-Recall.
    - **Análise de Negócio:** Risk Scoring e Cálculo de Risco Relativo (RR).
    """
)

setup_sidebar()
