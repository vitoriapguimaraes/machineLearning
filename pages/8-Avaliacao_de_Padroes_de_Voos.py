import streamlit as st
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go
from utils.ui import setup_sidebar, add_back_to_top
from utils.load_file import load_dataset
from utils.visualizations import COLOR_PALETTE

# --- Configuração Inicial ---
st.set_page_config(page_title="Padrões em Voos", page_icon="✈️", layout="wide")
setup_sidebar()
add_back_to_top()

st.title("✈️ Avaliação de Padrões em Voos")


# --- Carregamento de Dados ---
@st.cache_data
def load_data():
    try:
        # Carrega usando a função utilitária que já gerencia o caminho base 'data/'
        df = load_dataset("flights_created/df_view.csv")
    except Exception as e:
        st.error(f"Erro ao carregar dados: {e}")
        return pd.DataFrame()

    # Limpeza básica
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # Conversão de Datas
    if "FL_DATE" in df.columns:
        df["FL_DATE"] = pd.to_datetime(df["FL_DATE"])

    # Garantir Tipos Numéricos/Booleanos
    for col in ["CANCELLED", "DIVERTED", "DELAY"]:
        if col in df.columns:
            df[col] = df[col].astype(int)

    # Recalcular DELAY_OVERALL se necessário
    if "DELAY_OVERALL" not in df.columns and "ARR_DELAY" in df.columns:
        df["DELAY_OVERALL"] = df["ARR_DELAY"].clip(lower=0)

    # Recalcular DELAY_PER_DISTANCE se necessário
    if (
        "DELAY_PER_DISTANCE" not in df.columns
        and "DISTANCE" in df.columns
        and "DELAY_OVERALL" in df.columns
    ):
        df["DELAY_PER_DISTANCE"] = df.apply(
            lambda row: (
                row["DELAY_OVERALL"] / row["DISTANCE"] if row["DISTANCE"] > 0 else 0
            ),
            axis=1,
        )

    return df


df = load_data()

if df.empty:
    st.warning(
        "Não foi possível carregar os dados. Verifique se os arquivos estão na pasta 'data/flights_created'."
    )
    st.stop()


# --- Funções Auxiliares de Plotagem ---
def create_basic_chart(data, x_col, y_col, title, orientation="v"):
    fig = px.bar(
        data,
        x=x_col,
        y=y_col,
        title=title,
        orientation=orientation,
        color=y_col if orientation == "v" else x_col,  # Cor baseada no valor numérico
        color_continuous_scale=COLOR_PALETTE,
    )
    fig.update_layout(coloraxis_showscale=False)
    return fig


# --- Interface ---
tab1, tab2, tab3, tab4 = st.tabs(
    ["Visão Geral", "Métricas", "Análise de Padrões", "Mapa de Rotas"]
)

with tab1:

    st.subheader("Enunciado do Projeto")
    st.markdown(
        """
        A **Datalab**, consultoria especializada em análise de dados, busca fornecer soluções analíticas de ponta. Neste contexto, a tomada de decisão informada é fundamental para o sucesso nos negócios.

        Neste projeto, atuamos como analistas convidados para explorar um conjunto de dados de aviação, demonstrando habilidades técnicas e analíticas. O foco é entender a dinâmica operacional de voos domésticos, apresentando insights que poderiam ser entregues a um cliente real para otimização de processos.
        """
    )
    st.subheader("Objetivo do Projeto")
    st.markdown(
        """
        O objetivo deste estudo é **analisar padrões de atrasos, cancelamentos e desvios** em voos domésticos dos EUA (Janeiro/2023).

        Buscamos identificar segmentos que diferenciam o comportamento operacional de **companhias aéreas**, **aeroportos** e **períodos de tempo**. O propósito final é gerar insights acionáveis que apoiem decisões estratégicas para melhorar a eficiência operacional e reduzir impactos negativos para passageiros e companhias.
        """
    )
    st.subheader("Resultados e Conclusões")
    st.markdown(
        """
        Analisamos voos de ~30 dias, aplicando estatística descritiva e testes de hipótese.
        
        **Principais Descobertas:**
        *   **Companhias Aéreas:** A **Southwest Airlines** lidera em volume de problemas. Companhias *low-cost* (Frontier, Spirit) têm atraso médio **1.6x maior** que as tradicionais.
        *   **Rotas Críticas:** As cidades de **Chicago, Denver, Atlanta e Dallas** são os maiores polos de atrasos e cancelamentos.
        *   **Padrões Temporais:**
            *   **Quartas-feiras** apresentaram o pior desempenho (atraso médio 2.4x maior).
            *   **Voos Noturnos** (>18h) têm 1.36x mais chance de atraso, com pico crítico na madrugada (2h-3h).
        """
    )

# --- TAB 2: Métricas ---
with tab2:
    st.subheader("Métricas")

    total_flights = len(df)

    # Definir valores padrão para strings formatadas
    avg_delay_val = 0.0
    pct_delay_val = 0.0
    pct_cancel_val = 0.0
    pct_divert_val = 0.0

    if "DELAY_OVERALL" in df.columns:
        avg_delay_val = df["DELAY_OVERALL"].mean()
    if "DELAY" in df.columns:
        pct_delay_val = (df["DELAY"].sum() / total_flights) * 100
    if "CANCELLED" in df.columns:
        pct_cancel_val = (df["CANCELLED"].sum() / total_flights) * 100
    if "DIVERTED" in df.columns:
        pct_divert_val = (df["DIVERTED"].sum() / total_flights) * 100

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total de Voos", f"{total_flights:,}")
    c2.metric("Atraso Médio", f"{avg_delay_val:.1f} min")
    c3.metric("% Atrasos", f"{pct_delay_val:.1f}%")
    c4.metric("% Cancelamentos", f"{pct_cancel_val:.1f}%")
    c5.metric("% Desvios", f"{pct_divert_val:.1f}%")

    with st.expander("Ver amostra dos dados"):
        st.dataframe(df.head(50))

# --- TAB 3: Análise de Padrões ---
with tab3:
    st.subheader("Entendendo os Gargalos")

    metric_choice = st.radio(
        "Métrica de Análise:",
        ["Atraso Médio (min)", "Total de Cancelamentos", "Total de Voos"],
        horizontal=True,
    )

    metric_col_map = {
        "Atraso Médio (min)": ("DELAY_OVERALL", "mean"),
        "Total de Cancelamentos": ("CANCELLED", "sum"),
        "Total de Voos": ("FL_DATE", "count"),
    }
    col_target, agg_func = metric_col_map[metric_choice]

    # Verificar existência das colunas necessárias
    required_cols_chart = [
        col_target,
        "AIRLINE_Description",
        "DAY_OF_WEEK",
        "TIME_HOUR",
    ]
    missing_cols = [c for c in required_cols_chart if c not in df.columns]

    if missing_cols:
        st.error(f"Faltando colunas para gerar gráficos: {missing_cols}")
    else:
        col_g1, col_g2 = st.columns(2)

        with col_g1:
            st.markdown("#### Por Companhia Aérea")
            df_airline = (
                df.groupby("AIRLINE_Description")[col_target]
                .agg(agg_func)
                .reset_index()
                .sort_values(by=col_target, ascending=False)
                .head(10)
            )
            fig_airline = px.bar(
                df_airline,
                x=col_target,
                y="AIRLINE_Description",
                orientation="h",
                title=f"Top 10 Companhias por {metric_choice}",
                color=col_target,
                color_continuous_scale=COLOR_PALETTE,
            )
            fig_airline.update_layout(
                yaxis={"categoryorder": "total ascending"},
                showlegend=False,
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig_airline, use_container_width=True)

        with col_g2:
            st.markdown("#### Por Dia da Semana")
            # Ordenar dias
            days_order = [
                "Segunda-feira",
                "Terça-feira",
                "Quarta-feira",
                "Quinta-feira",
                "Sexta-feira",
                "Sábado",
                "Domingo",
            ]

            # Garantir que DAY_OF_WEEK esteja correto (pode ser int ou str dependendo do CSV)
            # Se for int (0-6), mapear
            if pd.api.types.is_numeric_dtype(df["DAY_OF_WEEK"]):
                day_map = {
                    0: "Segunda-feira",
                    1: "Terça-feira",
                    2: "Quarta-feira",
                    3: "Quinta-feira",
                    4: "Sexta-feira",
                    5: "Sábado",
                    6: "Domingo",
                }
                df_temp_week = df.copy()
                df_temp_week["DAY_OF_WEEK"] = df_temp_week["DAY_OF_WEEK"].map(day_map)
                df_week = (
                    df_temp_week.groupby("DAY_OF_WEEK")[col_target]
                    .agg(agg_func)
                    .reset_index()
                )
            else:
                df_week = (
                    df.groupby("DAY_OF_WEEK")[col_target].agg(agg_func).reset_index()
                )

            df_week["DAY_OF_WEEK"] = pd.Categorical(
                df_week["DAY_OF_WEEK"], categories=days_order, ordered=True
            )
            df_week = df_week.sort_values("DAY_OF_WEEK")

            fig_week = px.line(
                df_week,
                x="DAY_OF_WEEK",
                y=col_target,
                title="Evolução por Dia da Semana",
                markers=True,
            )
            fig_week.update_traces(line_color=COLOR_PALETTE[0], line_width=3)
            st.plotly_chart(fig_week, use_container_width=True)

        st.markdown("#### Por Hora do Dia")
        df_hour = df.groupby("TIME_HOUR")[col_target].agg(agg_func).reset_index()
        fig_hour = px.area(
            df_hour,
            x="TIME_HOUR",
            y=col_target,
            title=f"Tendência Horária ({metric_choice})",
            color_discrete_sequence=[COLOR_PALETTE[1]],
        )
        st.plotly_chart(fig_hour, use_container_width=True)

# --- TAB 4: Mapa de Rotas ---
with tab4:
    st.subheader("Mapa de Rotas Críticas")
    st.caption("Visualização das rotas com maiores índices de problemas.")

    qty_routes = st.slider("Quantidade de Rotas", 5, 100, 30)

    # Verificar colunas de coordenadas
    coord_cols = ["ORIGIN_LAT", "ORIGIN_LON", "DEST_LAT", "DEST_LON"]
    missing_coords = [c for c in coord_cols if c not in df.columns]

    if missing_coords:
        st.warning(
            f"Coordenadas ausentes no dataset ({missing_coords}). O mapa não pode ser gerado corretamente."
        )
    else:
        # Preparar dados para o mapa (Agrupado por Rota)
        # Usar ORIGIN_CITY/DEST_CITY se existirem, senão ORIGIN/DEST
        orig_key = "ORIGIN_CITY" if "ORIGIN_CITY" in df.columns else "ORIGIN"
        dest_key = "DEST_CITY" if "DEST_CITY" in df.columns else "DEST"

        route_grp = (
            df.groupby(
                [orig_key, dest_key, "ORIGIN_LAT", "ORIGIN_LON", "DEST_LAT", "DEST_LON"]
            )
            .agg(
                avg_delay=("DELAY_OVERALL", "mean"), total_flights=("FL_DATE", "count")
            )
            .reset_index()
        )

        top_routes = route_grp.sort_values("avg_delay", ascending=False).head(
            qty_routes
        )

        fig_map = go.Figure()

        # Adicionar rotas (linhas)
        for _, row in top_routes.iterrows():
            fig_map.add_trace(
                go.Scattergeo(
                    lon=[row["ORIGIN_LON"], row["DEST_LON"]],
                    lat=[row["ORIGIN_LAT"], row["DEST_LAT"]],
                    mode="lines",
                    line=dict(width=1.5, color="red"),
                    opacity=0.6,
                    hoverinfo="text",
                    text=f"{row[orig_key]} -> {row[dest_key]} | Delay Médio: {row['avg_delay']:.1f} min",
                    showlegend=False,
                )
            )

        # Adicionar Aeroportos (pontos)
        origins = top_routes[[orig_key, "ORIGIN_LAT", "ORIGIN_LON"]].rename(
            columns={orig_key: "CODE", "ORIGIN_LAT": "LAT", "ORIGIN_LON": "LON"}
        )
        dests = top_routes[[dest_key, "DEST_LAT", "DEST_LON"]].rename(
            columns={dest_key: "CODE", "DEST_LAT": "LAT", "DEST_LON": "LON"}
        )
        airports = pd.concat([origins, dests]).drop_duplicates()

        fig_map.add_trace(
            go.Scattergeo(
                lon=airports["LON"],
                lat=airports["LAT"],
                text=airports["CODE"],
                mode="markers",
                marker=dict(size=5, color="blue"),
                hoverinfo="text",
                name="Aeroportos",
            )
        )

        fig_map.update_layout(
            title_text="Rotas com Maior Atraso Médio",
            geo=dict(
                scope="usa",
                projection_type="albers usa",
                showland=True,
                landcolor="rgb(240, 240, 240)",
            ),
            height=600,
            margin={"r": 0, "t": 40, "l": 0, "b": 0},
        )

        st.plotly_chart(fig_map, use_container_width=True)
