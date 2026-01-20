import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from utils.ui import setup_sidebar, add_back_to_top
from utils.visualizations import plot_heatmap, plot_confusion_matrix

# Configuração da Página
st.set_page_config(page_title="Risco de Crédito", page_icon="🏦", layout="wide")

setup_sidebar()
add_back_to_top()

st.title("🏦 Avaliação de Risco de Crédito Bancário")


# --- Carregamento e Preparação dos Dados ---
@st.cache_data
def load_data():
    DATA_PATH = "data/bank_dt_full_analysis.csv"
    try:
        df = pd.read_csv(DATA_PATH)
        # Preenchimento básico de nulos com mediana
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].median())
        return df
    except FileNotFoundError:
        st.error(f"Arquivo de dados não encontrado: {DATA_PATH}")
        return None


df = load_data()


# --- Treinamento do Modelo (Cacheado) ---
@st.cache_resource
def train_model(data):
    if data is None:
        return None, None, None, None, None

    numeric_columns = [
        "age",
        "sex_num",
        "last_month_salary",
        "number_dependents",
        "total_emprestimos",
        "qtd_real_estate",
        "qtd_others",
        "perc_real_estate",
        "perc_others",
        "more_90_days_overdue",
        "using_lines_not_secured_personal_assets",
        "number_times_delayed_payment_loan_30_59_days",
        "debt_ratio",
        "number_times_delayed_payment_loan_60_89_days",
    ]

    feature_cols = [c for c in numeric_columns if c in data.columns]

    X = data[feature_cols]
    y = data["default_flag"]

    # Estratificar é importante devido ao desbalanceamento
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # class_weight='balanced' ajusta os pesos para lidar com classes desbalanceadas
    model = LogisticRegression(
        solver="liblinear", random_state=42, class_weight="balanced"
    )
    model.fit(X_train, y_train)

    return model, X_test, y_test, feature_cols


model, X_test, y_test, feature_cols = train_model(df)

# --- Interface ---
tabs = st.tabs(
    ["Visão Geral", "Análise Exploratória", "Modelagem & Previsão", "Simulador"]
)

# --- Aba: Visão Geral ---
with tabs[0]:
    st.subheader("Enunciado do Projeto")
    st.markdown(
        """
        No atual cenário financeiro, a diminuição das taxas de juros tem gerado um notável aumento na demanda por crédito no banco "Super Caja". No entanto, essa crescente demanda tem sobrecarregado a equipe de análise de crédito, que atualmente está imersa em um processo manual ineficiente e demorado para avaliar as inúmeras solicitações de empréstimo. Diante desse desafio, propõe-se uma solução inovadora: a automatização do processo de análise por meio de técnicas avançadas de análise de dados. O objetivo principal é melhorar a eficiência e a precisão na avaliação do risco de crédito, permitindo ao banco tomar decisões informadas sobre a concessão de crédito e reduzir o risco de empréstimos não reembolsáveis. Esta proposta também destaca a integração de uma métrica existente de pagamentos em atraso, fortalecendo assim a capacidade do modelo. Este projeto não apenas oferece a oportunidade de se aprofundar na análise de dados, mas também proporciona a aquisição de habilidades-chave na classificação de clientes, no uso da matriz de confusão e na realização de consultas complexas no BigQuery, preparando-o para enfrentar desafios analíticos em diversos campos.
        """
    )
    st.subheader("Objetivo do Projeto")
    st.markdown(
        """
        Automatizar a análise de risco de crédito utilizando Machine Learning para classificar clientes com base na probabilidade de inadimplência. O banco busca maior eficiência na concessão de crédito.
    """
    )
    st.subheader("Resultados e Conclusões")
    st.markdown(
        """
        No desenvolvimento deste modelo de **Risk Scoring** com Regressão Logística, processamos dados de ~36 mil clientes (2% inadimplentes).
        """
    )
    st.markdown(
        """
        **Principais Descobertas:**
        *   **Perfil de Risco:** Clientes inadimplentes mostram forte correlação com **menor renda** e **histórico de atrasos** (>90 dias e 30-59 dias).
        *   **Performance:** Acurácia global de **79%**. O modelo é conservador (alta especificidade), minimizando a concessão de crédito a maus pagadores, mas pode ser cauteloso com alguns bons.
        *   **Impacto:** Permite triagem automatizada eficiente, focando a análise humana nos casos limítrofes.
        """
    )

# --- Aba: Análise Exploratória ---
with tabs[1]:
    if df is not None:

        col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
        col_kpi1.metric("Total de Clientes", f"{len(df):,}")
        col_kpi2.metric(
            "Taxa Média de Inadimplência", f"{df['default_flag'].mean():.2%}"
        )
        col_kpi3.metric("Salário Médio", f"R$ {df['last_month_salary'].mean():.2f}")

        st.subheader("Análise de Risco Relativo (RR)")
        st.info(
            """
        **Risco Relativo (RR):** Compara a taxa de inadimplência de um grupo específico com a taxa média global.
        - **RR > 1.0:** Indica um grupo de **maior risco**.
        - **RR < 1.0:** Indica um grupo de **menor risco** (proteção).
        """
        )

        base_risk = df["default_flag"].mean()

        col_eda1, col_eda2 = st.columns(2)

        with col_eda1:
            # RR por Faixa Etária
            # Criar bins para idade
            df["AgeGroup"] = pd.cut(
                df["age"],
                bins=[0, 30, 45, 60, 100],
                labels=["<30 Anos", "30-45 Anos", "45-60 Anos", "60+ Anos"],
            )
            rr_age = (
                df.groupby("AgeGroup", observed=True)["default_flag"]
                .agg(["mean", "count"])
                .reset_index()
            )
            rr_age["RR"] = rr_age["mean"] / base_risk

            fig_rr = px.bar(
                rr_age,
                x="AgeGroup",
                y="RR",
                title="Risco Relativo por Faixa Etária",
                text=rr_age["RR"].apply(lambda x: f"{x:.2f}x"),
                color="RR",
                color_continuous_scale="RdYlGn_r",
            )
            fig_rr.add_hline(
                y=1, line_dash="dot", annotation_text=f"Risco Base ({base_risk:.1%})"
            )
            fig_rr.update_layout(yaxis_title="Risco Relativo (RR)")
            st.plotly_chart(fig_rr, use_container_width=True)

        with col_eda2:
            # RR por Histórico de Atrasos (>90 dias)
            # Binária: Teve atraso vs Não teve
            df["HasDelinquency"] = df["more_90_days_overdue"].apply(
                lambda x: "Com Atraso Grave" if x > 0 else "Sem Atraso Grave"
            )
            rr_delinq = (
                df.groupby("HasDelinquency", observed=True)["default_flag"]
                .agg(["mean", "count"])
                .reset_index()
            )
            rr_delinq["RR"] = rr_delinq["mean"] / base_risk

            fig_rr_d = px.bar(
                rr_delinq,
                x="HasDelinquency",
                y="RR",
                title="Impacto de Atrasos Graves (>90d) no Risco",
                text=rr_delinq["RR"].apply(lambda x: f"{x:.2f}x"),
                color="RR",
                color_continuous_scale="RdYlGn_r",
            )
            fig_rr_d.add_hline(y=1, line_dash="dot", annotation_text="Risco Base")
            fig_rr_d.update_layout(yaxis_title="Risco Relativo (RR)")
            st.plotly_chart(fig_rr_d, use_container_width=True)

        st.subheader("Outras Correlações")
        corr_cols = [
            "default_flag",
            "age",
            "debt_ratio",
            "total_emprestimos",
            "more_90_days_overdue",
            "last_month_salary",
        ]
        corr_matrix = df[corr_cols].corr()

        plot_heatmap(df, corr_cols, height=400)

# --- Aba: Modelagem & Previsão ---
with tabs[2]:
    if model is not None:
        st.subheader("Métricas do Modelo (Regressão Logística)")

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        col_res1, col_res2 = st.columns([1, 2])

        with col_res1:
            auc_test = roc_auc_score(y_test, y_proba)
            st.metric("AUC-ROC (Teste)", f"{auc_test:.2%}", delta_color="normal")
            st.metric("Acurácia", f"{model.score(X_test, y_test):.2%}")

            st.markdown("##### Detalhes por Classe")
            report = classification_report(y_test, y_pred, output_dict=True)
            report_df = (
                pd.DataFrame(report)
                .transpose()
                .drop(["accuracy", "macro avg", "weighted avg"])
            )
            st.dataframe(report_df.style.format("{:.2f}"), use_container_width=True)

        with col_res2:
            cm = confusion_matrix(y_test, y_pred)

            plot_confusion_matrix(
                cm,
                x_labels=["Bom Pagador (0)", "Inadimplente (1)"],
                y_labels=["Bom Pagador (0)", "Inadimplente (1)"],
            )

# --- Aba: Simulador ---
with tabs[3]:
    st.subheader("Simulador de Score de Crédito")

    if model is not None:

        col_form, c_score = st.columns([3, 1])

        with col_form.form("simulacao_form"):
            st.markdown(
                "Insira os dados do cliente para calcular a probabilidade de default."
            )
            col_s1, col_s2, col_s3 = st.columns(3)

            with col_s1:
                age_input = st.number_input("Idade", 18, 100, 40)
                salary_input = st.number_input(
                    "Renda Mensal (R$)", 0.0, 500000.0, 5000.0
                )
                dependents_input = st.number_input("Nº Dependentes", 0, 20, 2)

            with col_s2:
                loans_input = st.number_input("Total de Empréstimos", 0, 50, 1)
                real_estate_input = st.number_input("Qtd. Imóveis", 0, 20, 0)
                debt_ratio_input = st.slider(
                    "Taxa de Endividamento (Debt Ratio)", 0.0, 5.0, 0.3
                )

            with col_s3:
                late_30_59 = st.number_input("Atrasos 30-59 dias", 0, 20, 0)
                late_60_89 = st.number_input("Atrasos 60-89 dias", 0, 20, 0)
                late_90 = st.number_input("Atrasos > 90 dias", 0, 20, 0)

            submit_btn = st.form_submit_button("Calcular Risco")

        with c_score:
            with st.container(border=True):

                if submit_btn:
                    # Preparar input
                    input_data = {
                        "age": age_input,
                        "sex_num": 0,  # Valor padrão genérico
                        "last_month_salary": salary_input,
                        "number_dependents": dependents_input,
                        "total_emprestimos": loans_input,
                        "qtd_real_estate": real_estate_input,
                        "qtd_others": max(0, loans_input - real_estate_input),
                        "perc_real_estate": (
                            real_estate_input / loans_input if loans_input > 0 else 0
                        ),
                        "perc_others": (
                            (loans_input - real_estate_input) / loans_input
                            if loans_input > 0
                            else 0
                        ),
                        "more_90_days_overdue": late_90,
                        "using_lines_not_secured_personal_assets": 0,
                        "number_times_delayed_payment_loan_30_59_days": late_30_59,
                        "debt_ratio": debt_ratio_input,
                        "number_times_delayed_payment_loan_60_89_days": late_60_89,
                    }

                    input_df = pd.DataFrame([input_data])
                    # Garantir mesmas colunas usadas no treino
                    input_df = input_df[feature_cols]

                    prob_default = model.predict_proba(input_df)[0][1]
                    score = int((1 - prob_default) * 1000)  # Score tipo Serasa 0-1000

                    st.metric("Probabilidade de Default", f"{prob_default:.1%}")
                    st.metric("Score Calculado", f"{score}/1000")

                    if prob_default > 0.5:
                        st.error("🚨 **Alto Risco detectado**")
                        st.markdown(
                            "Recomendação: **Negar Crédito** ou exigir garantias adicionais."
                        )
                    elif prob_default > 0.2:
                        st.warning("⚠️ **Risco Moderado**")
                        st.markdown(
                            "Recomendação: **Aprovar com cautela** (limite reduzido)."
                        )
                    else:
                        st.success("✅ **Baixo Risco**")
                        st.markdown("Recomendação: **Aprovar Crédito**.")

                else:
                    st.markdown("### Resultado indisponível")
                    st.markdown(
                        "Preencha os dados do cliente e clique em 'Calcular Risco' para obter o resultado."
                    )
