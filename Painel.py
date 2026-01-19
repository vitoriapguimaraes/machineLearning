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

st.markdown("---")

st.subheader("Ferramentas Utilizadas")
st.info("a adicionar")

st.subheader("Competências Desenvolvidas")
st.info("a adicionar")

setup_sidebar()
