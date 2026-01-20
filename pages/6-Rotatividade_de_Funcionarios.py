import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
from pathlib import Path
from utils.ui import setup_sidebar, add_back_to_top
from utils.visualizations import plot_confusion_matrix

st.set_page_config(page_title="Previsor de Rotatividade", page_icon="👤", layout="wide")

setup_sidebar()
add_back_to_top()

st.title("👤 Previsor de Rotatividade de Funcionários")

MODEL_DIR = Path("./data/model")

tabs = st.tabs(["Visão Geral", "Previsão", "Métricas do modelo", "Feature Importance"])


with tabs[0]:
    st.subheader("Enunciado do Projeto")
    st.markdown(
        """
    No dinâmico ambiente empresarial atual, a retenção de talentos tornou-se um fator crítico para o sucesso sustentável das organizações.
    Este projeto concentra-se na análise dos dados de recursos humanos de uma empresa com o objetivo de desenvolver um modelo de machine learning supervisionado
    para prever com precisão se um funcionário está propenso a deixar a organização.

    A aplicação de técnicas avançadas de machine learning na supervisão de recursos humanos representa uma oportunidade única para otimizar a tomada de decisões
    baseada em dados e melhorar a eficiência operacional no âmbito da gestão de talentos.
    """
    )

    st.subheader("Objetivo")
    st.markdown(
        """
    Desenvolver um modelo de machine learning supervisionado capaz de prever a probabilidade de um funcionário deixar a empresa, utilizando dados históricos do departamento de Recursos Humanos.
    O modelo busca identificar padrões e fatores que influenciam a rotatividade para apoiar ações preventivas de retenção.
    """
    )

    st.subheader("Resultados e Conclusões")
    st.markdown(
        """
    O projeto alcançou com sucesso o objetivo de desenvolver um modelo preditivo.
    - **Análise Exploratória**: Verificou-se que a rotatividade está associada a fatores como idade (mais jovens saem mais), menor tempo de casa, menor salário e menor experiência total.
    - **Modelagem**: O algoritmo **XGBoost** foi o destaque, alcançando **99.6% de acurácia** nos dados de teste, superando Random Forest e Regressão Logística.
    - **Fatores Críticos**: As variáveis mais influentes detectadas pelo modelo foram:
        - `MaritalStatus`: Solteiros tendem a ter maior rotatividade.
        - `TotalWorkingYears` e `YearsAtCompany`: Menor tempo de casa/experiência aumenta o risco.
        - `Age`: Jovens são mais propensos a sair.

    O modelo fornece insights estratégicos para o RH antecipar desligamentos e planejar ações de retenção focadas.
    """
    )


with tabs[1]:
    main_column1, spacer, main_column2 = st.columns([3, 0.2, 1])

    with main_column1:
        st.subheader("Parâmetros de Entrada")

        col1, spacer_inner, col2, spacer_inner2, col3 = st.columns([1, 0.1, 1, 0.1, 1])

        with col1:
            st.markdown("##### Dados Temporais")
            age = st.slider("Idade", 18, 60, 30)
            total_years = st.slider("Anos totais de experiência", 0, 30, 10)
            years_at_company = st.slider("Anos na empresa atual", 0, 20, 5)
            years_with_manager = st.slider("Anos com o mesmo gerente", 0, 10, 3)

        with col2:
            st.markdown("##### Dados Pessoais")
            marital = st.pills(
                "Estado civil", ["Single", "Married", "Divorced"], default="Single"
            )
            gender = st.pills("Gênero", ["Male", "Female"], default="Male")
            travel = st.selectbox(
                "Frequência de viagens",
                ["Non-Travel", "Travel_Rarely", "Travel_Frequently"],
                index=1,
            )
            distance = st.slider("Distância da casa (km)", 0, 50, 10)

        with col3:
            st.markdown("##### Dados Profissionais")
            department = st.selectbox(
                "Departamento",
                ["Sales", "Research & Development", "Human Resources"],
                index=0,
            )
            job_role = st.selectbox(
                "Cargo",
                [
                    "Sales Executive",
                    "Research Scientist",
                    "Laboratory Technician",
                    "Manufacturing Director",
                    "Healthcare Representative",
                    "Manager",
                    "Sales Representative",
                    "Research Director",
                    "Human Resources",
                ],
            )

        # Montagem do DataFrame de entrada
        input_data = pd.DataFrame(
            {
                "Age": [age],
                "TotalWorkingYears": [total_years],
                "YearsAtCompany": [years_at_company],
                "YearsWithCurrManager": [years_with_manager],
                "MaritalStatus": [marital],
                "BusinessTravel": [travel],
                "Department": [department],
                "Gender": [gender],
                "JobRole": [job_role],
                "DistanceFromHome": [distance],
                "Attrition": ["No"],
            }
        )

        # Engenharia de Features (Feature Engineering) básica igual ao treinamento
        bins = [18, 25, 35, 45, 55, 65]
        labels = ["18-25", "26-35", "36-45", "46-55", "56+"]
        input_data["AgeGroup"] = pd.cut(
            input_data["Age"], bins=bins, labels=labels, right=False
        )
        input_data["FarFromHome"] = (input_data["DistanceFromHome"] > 10).astype(int)
        input_data["CompanyExperienceRatio"] = input_data["YearsAtCompany"] / (
            input_data["TotalWorkingYears"] + 1
        )
        input_data["AvgYearsPerCompany"] = input_data["TotalWorkingYears"] / (
            input_data["YearsAtCompany"] + 1
        )
        input_data["SalaryHikePerIncome"] = 0  # Placeholder se necessário pelo modelo

        input_data_model = input_data.copy()

        # Carregamento dos objetos do modelo
        try:
            encoders = joblib.load(MODEL_DIR / "label_encoders.pkl")
            scaler = joblib.load(MODEL_DIR / "scaler.pkl")
            num_cols = joblib.load(MODEL_DIR / "num_cols.pkl")
            xgb_model = joblib.load(MODEL_DIR / "xgb_model.pkl")

            # Aplicação dos Encoders
            categorical_cols = input_data_model.select_dtypes(
                include=["object", "category"]
            ).columns
            for col in categorical_cols:
                if col in encoders:
                    try:
                        input_data_model[col] = encoders[col].transform(
                            input_data_model[col]
                        )
                    except ValueError:
                        # Caso valor novo não visto no treino
                        input_data_model[col] = 0

            # Aplicação do Scaler
            # Filtrar colunas numéricas que realmente existem no input e no scaler
            numeric_cols_to_scale = [
                col for col in num_cols if col in input_data_model.columns
            ]
            if numeric_cols_to_scale:
                input_data_model[numeric_cols_to_scale] = scaler.transform(
                    input_data_model[numeric_cols_to_scale]
                )

        except FileNotFoundError as e:
            st.error(
                f"❌ Erro ao carregar arquivos do modelo. Verifique se a pasta `predictionHumanResources/data/model` existe e contêm os arquivos .pkl.\nDetalhe: {e}"
            )
            st.stop()
        except Exception as e:
            st.error(f"❌ Erro inesperado ao processar dados: {e}")
            st.stop()

    with main_column2:
        with st.container(border=True):
            st.subheader("Previsão")
            try:
                # Garantir colunas na ordem correta do modelo
                if hasattr(xgb_model, "feature_names_in_"):
                    for col in xgb_model.feature_names_in_:
                        if col not in input_data_model.columns:
                            input_data_model[col] = 0
                    input_data_model = input_data_model[xgb_model.feature_names_in_]

                # Predição
                prob = xgb_model.predict_proba(input_data_model)[0][1]

                if prob > 0.7:
                    delta_color = "inverse"
                    delta_text = "Alto Risco"
                    st.error(
                        "⚠️ **Atenção:** Alta probabilidade de rotatividade! Ações preventivas são fortemente recomendadas."
                    )
                elif prob > 0.4:
                    delta_color = "off"
                    delta_text = "Risco Moderado"
                    st.warning(
                        "⚠️ **Alerta:** Risco moderado de rotatividade. Vale a pena revisar os planos de desenvolvimento e engajamento."
                    )
                else:
                    delta_color = "normal"
                    delta_text = "Baixo Risco"
                    st.success(
                        "✅ **Estável:** Baixo risco de saída. O funcionário demonstra boa estabilidade."
                    )

                st.metric(
                    label="Probabilidade de Saída",
                    value=f"{prob*100:.1f}%",
                    delta=delta_text,
                    delta_color=delta_color,
                )

            except Exception as e:
                st.error(f"❌ Erro ao fazer previsão: {e}")
                st.exception(e)


with tabs[2]:
    st.subheader("Métricas do modelo")
    try:
        metrics = joblib.load(MODEL_DIR / "model_metrics.pkl")
        conf_matrix = joblib.load(MODEL_DIR / "confusion_matrix.pkl")

        col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
        col_m1.metric(
            "🎯 Acurácia",
            f"{metrics.get('Acurácia', 0):.2%}",
            help="Proporção de acertos totais",
        )
        col_m2.metric(
            "🔍 Precisão",
            f"{metrics.get('Precisão', 0):.2%}",
            help="Dos preditos como saída, quantos realmente saíram",
        )
        col_m3.metric(
            "📡 Recall",
            f"{metrics.get('Recall', 0):.2%}",
            help="Dos que realmente saíram, quantos o modelo achou",
        )
        col_m4.metric(
            "⚖️ F1-Score",
            f"{metrics.get('F1-Score', 0):.2%}",
            help="Média harmônica entre Precisão e Recall",
        )
        auc_val = metrics.get("AUC-ROC")
        col_m5.metric("📈 AUC-ROC", f"{auc_val:.2%}" if auc_val else "N/A")

        col_conf1, col_conf2 = st.columns([1.5, 1])

        with col_conf1:
            plot_confusion_matrix(
                conf_matrix,
                x_labels=["Não Sai (0)", "Sai (1)"],
                y_labels=["Não Sai (0)", "Sai (1)"],
            )

        with col_conf2:
            tn, fp, fn, tp = conf_matrix.ravel()
            st.markdown("##### Matriz de Confusão")
            st.markdown(
                f"""
            **Legenda:**
            - **VN ({tn})**: Verdadeiros Negativos (Ficou e previsto Ficar)
            - **FP ({fp})**: Falsos Positivos (Ficou mas previsto Sair)
            - **FN ({fn})**: Falsos Negativos (Saiu mas previsto Ficar)
            - **VP ({tp})**: Verdadeiros Positivos (Saiu e previsto Sair)
            """
            )

    except FileNotFoundError:
        st.info(
            "Arquivos de métricas não encontrados. Execute o notebook de treino para gerá-los."
        )
    except Exception as e:
        st.error(f"Erro ao carregar métricas: {e}")


with tabs[3]:
    st.subheader("Importância das Variáveis")

    try:
        feature_importance = (
            pd.DataFrame(
                {
                    "Feature": xgb_model.feature_names_in_,
                    "Importância": xgb_model.feature_importances_,
                }
            )
            .sort_values("Importância", ascending=False)
            .head(15)
        )

        feature_importance["Importância (%)"] = (
            feature_importance["Importância"]
            / feature_importance["Importância"].sum()
            * 100
        ).round(2)

        col_fi1, col_fi2 = st.columns([2, 1])

        fig = px.bar(
            feature_importance.sort_values("Importância (%)", ascending=True),
            x="Importância (%)",
            y="Feature",
            orientation="h",
            text="Importância (%)",
            title="<b>Top 15 Variáveis Mais Importantes</b>",
            labels={
                "Importância (%)": "Importância Relativa (%)",
                "Feature": "Variável",
            },
            color_discrete_sequence=["#1f77b4"],
            template="plotly_white",
        )

        fig.update_traces(
            texttemplate="%{text:.1f}%",
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Importância: %{x:.2f}%<extra></extra>",
        )

        fig.update_layout(
            yaxis={"categoryorder": "total ascending"},
            xaxis=dict(showgrid=True, gridcolor="lightgray"),
            showlegend=False,
            margin=dict(l=0, r=0, t=40, b=0),
            height=500,
        )

        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Erro ao gerar gráfico de importância: {e}")
