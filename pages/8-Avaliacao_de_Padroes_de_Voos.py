import streamlit as st
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go
from utils.ui import setup_sidebar, add_back_to_top
from utils.load_file import load_dataset
from utils.visualizations import COLOR_PALETTE
from utils.models import train_flight_model, predict_flight_delay

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
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "Visão Geral",
        "Sobre os dados",
        "Análise de Padrões",
        "Mapa de Rotas",
        "Simulador",
    ]
)

with tab1:

    st.subheader("Enunciado do Projeto")
    st.markdown(
        """
        Neste projeto, explora-se um conjunto de dados de aviação, demonstrando habilidades técnicas e analíticas. O foco é entender a dinâmica operacional de voos domésticos, apresentando insights que poderiam ser entregues a um cliente real para otimização de processos.

        O objetivo deste estudo é **analisar padrões de atrasos, cancelamentos e desvios** em voos domésticos dos EUA (Janeiro/2023).

        A análise busca identificar segmentos que diferenciam o comportamento operacional de **companhias aéreas**, **aeroportos** e **períodos de tempo**. O propósito final é gerar insights acionáveis que apoiem decisões estratégicas para melhorar a eficiência operacional e reduzir impactos negativos para passageiros e companhias.
        """
    )
    st.subheader("Resultados e Conclusões")
    st.markdown(
        "Foi analisado voos de ~30 dias, aplicando estatística descritiva e testes de hipótese."
    )
    st.markdown(
        """
        **Principais Descobertas:**
        *   **Companhias Aéreas:** A **Southwest Airlines** lidera em volume de problemas. Companhias *low-cost* (Frontier, Spirit) têm atraso médio **1.6x maior** que as tradicionais.
        *   **Rotas Críticas:** As cidades de **Chicago, Denver, Atlanta e Dallas** são os maiores polos de atrasos e cancelamentos.
        *   **Padrões Temporais:**
            *   **Quartas-feiras** apresentaram o pior desempenho (atraso médio 2.4x maior).
            *   **Voos Noturnos** (>18h) têm 1.36x mais chance de atraso, com pico crítico na madrugada (2h-3h).
        """
    )

# --- TAB 2: Sobre os dados ---
with tab2:
    st.subheader("Sobre os dados")

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

    st.dataframe(df.head(5))

    st.subheader("Estatísticas Descritivas")
    st.dataframe(df.describe())

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


# --- Funções do Mapa (Portadas de app_streamlit.py) ---
def _calcular_cor_horario(time_hour):
    hora_normalizada = time_hour / 23.0
    hue = 200 + (hora_normalizada * 80)
    saturation = 60 + (hora_normalizada * 30)
    if time_hour <= 12:
        lightness = 30 + (time_hour / 12.0) * 40
    else:
        lightness = 70 - ((time_hour - 12) / 11.0) * 40

    hue = max(0, min(hue, 360))
    saturation = max(0, min(saturation, 100))
    lightness = max(0, min(lightness, 100))

    return f"hsl({hue:.0f}, {saturation:.0f}%, {lightness:.0f}%)"


def _calcular_espessuras(rotas_data, col_metric):
    min_metric = rotas_data[col_metric].min()
    max_metric = rotas_data[col_metric].max()

    if max_metric <= min_metric or pd.isna(min_metric) or pd.isna(max_metric):
        return [3] * len(rotas_data)

    espessuras = []
    for valor in rotas_data[col_metric]:
        if pd.isna(valor):
            espessuras.append(3)
            continue
        metric_normalizada = (valor - min_metric) / (max_metric - min_metric)
        espessura = 1.5 + (metric_normalizada**1.5) * 7
        espessuras.append(max(1.5, min(espessura, 8.5)))

    return espessuras


def criar_mapa_rotas_avancado(df, top_n=30, selected_metric="DELAY_PER_DISTANCE"):

    # Preparar dados
    # Remover NaNs críticos
    df_filtered = df.dropna(
        subset=["ORIGIN_LAT", "ORIGIN_LON", "DEST_LAT", "DEST_LON", "DISTANCE"]
    )

    # Agrupar por rota
    agg_dict = {
        "DELAY_OVERALL": "mean",
        "DELAY": "sum",
        "CANCELLED": "sum",
        "DIVERTED": "sum",
        "DELAY_PER_DISTANCE": "mean",
        "FL_DATE": "count",
        "TIME_HOUR": "mean",
    }

    # Adicionar colunas faltantes se não existirem
    for col in agg_dict.keys():
        if col not in df_filtered.columns:
            df_filtered[col] = 0

    rotas_data = (
        df_filtered.groupby(
            [
                "ORIGIN_CITY",
                "DEST_CITY",
                "ORIGIN_LAT",
                "ORIGIN_LON",
                "DEST_LAT",
                "DEST_LON",
                "DISTANCE",
            ],
            as_index=False,
        )
        .agg(agg_dict)
        .rename(columns={"FL_DATE": "TOTAL_VOOS"})
        .sort_values(by="TOTAL_VOOS", ascending=False)  # Default sort
        .head(top_n)
        .reset_index(drop=True)
    )

    if rotas_data.empty:
        return go.Figure().update_layout(title="Sem dados para exibir")

    fig = go.Figure()

    # 1. Adicionar Estados Críticos (Marcadores Hexagonais)
    estados_centros = {
        "CA": {"lon": -119.4, "lat": 36.7, "nome": "Califórnia"},
        "FL": {"lon": -81.5, "lat": 27.9, "nome": "Flórida"},
        "TX": {"lon": -99.9, "lat": 31.0, "nome": "Texas"},
        "CO": {"lon": -105.5, "lat": 39.0, "nome": "Colorado"},
    }
    for estado, dados in estados_centros.items():
        fig.add_trace(
            go.Scattergeo(
                lon=[dados["lon"]],
                lat=[dados["lat"]],
                mode="markers+text",
                marker=dict(
                    size=25,
                    color="rgba(255, 100, 100, 0.15)",
                    symbol="hexagon",
                    line=dict(width=1, color="rgba(255, 50, 50, 0.4)"),
                ),
                text=[estado],
                textfont=dict(size=10, color="rgba(200, 0, 0, 0.6)"),
                textposition="middle center",
                hoverinfo="none",
                showlegend=False,
            )
        )

    # 2. Adicionar Rotas (Linhas Coloridas por Horário e Espessura por Atraso)
    # Usando DELAY_OVERALL para espessura para ser visualmente impactante
    espessuras = _calcular_espessuras(rotas_data, "DELAY_OVERALL")

    for idx, rota in rotas_data.iterrows():
        cor = _calcular_cor_horario(rota["TIME_HOUR"])

        fig.add_trace(
            go.Scattergeo(
                lon=[rota["ORIGIN_LON"], rota["DEST_LON"]],
                lat=[rota["ORIGIN_LAT"], rota["DEST_LAT"]],
                mode="lines",
                line=dict(width=espessuras[idx], color=cor),
                name=f"{rota['ORIGIN_CITY']} -> {rota['DEST_CITY']}",
                showlegend=False,
                hovertemplate=(
                    f"<b>{rota['ORIGIN_CITY']} -> {rota['DEST_CITY']}</b><br>"
                    f"Atraso Médio: {rota['DELAY_OVERALL']:.1f} min<br>"
                    f"Hora Média: {rota['TIME_HOUR']:.1f}h<br>"
                    f"Total de Voos: {rota['TOTAL_VOOS']}<extra></extra>"
                ),
            )
        )

    # 3. Adicionar Marcadores de Origem (Verde) e Destino (Vermelho)
    fig.add_trace(
        go.Scattergeo(
            lon=rotas_data["ORIGIN_LON"],
            lat=rotas_data["ORIGIN_LAT"],
            mode="markers",
            marker=dict(size=6, color="green", line=dict(width=1, color="white")),
            name="Origem",
            hoverinfo="name+text",
            text=rotas_data["ORIGIN_CITY"],
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scattergeo(
            lon=rotas_data["DEST_LON"],
            lat=rotas_data["DEST_LAT"],
            mode="markers",
            marker=dict(size=6, color="red", line=dict(width=1, color="white")),
            name="Destino",
            hoverinfo="name+text",
            text=rotas_data["DEST_CITY"],
            showlegend=False,
        )
    )

    # 4. Adicionar Cidades Críticas (Estrelas)
    cidades_destaque = {
        "Chicago": {"lat": 41.8781, "lon": -87.6298},
        "Denver": {"lat": 39.7392, "lon": -104.9903},
        "Atlanta": {"lat": 33.7490, "lon": -84.3880},
        "Dallas": {"lat": 32.7767, "lon": -96.7970},
    }
    for cidade, coords in cidades_destaque.items():
        fig.add_trace(
            go.Scattergeo(
                lon=[coords["lon"]],
                lat=[coords["lat"]],
                mode="markers",
                marker=dict(
                    size=8,
                    color="gold",
                    symbol="star",
                    line=dict(width=1, color="orange"),
                ),
                name="Hub Crítico",
                text=[cidade],
                hovertemplate=f"<b>{cidade}</b> (Hub Crítico)<extra></extra>",
                showlegend=False,
            )
        )

    # Layout Setup
    fig.update_layout(
        title=dict(
            text="<b>Mapa de Rotas e Gargalos Operacionais</b><br><sub>🎨 Cor da linha: Horário do voo | 📏 Espessura: Intensidade do Atraso</sub>",
            x=0.5,
        ),
        geo=dict(
            scope="usa",
            projection_type="albers usa",
            showland=True,
            landcolor="rgb(243, 243, 238)",
            showlakes=True,
            lakecolor="rgb(220, 235, 255)",
            showsubunits=True,
            subunitcolor="rgb(220, 220, 220)",
            bgcolor="rgba(0,0,0,0)",
        ),
        height=600,
        margin=dict(l=0, r=0, t=80, b=0),
    )

    return fig


# --- TAB 4: Mapa de Rotas ---
with tab4:
    st.subheader("Mapa de Rotas Críticas")
    st.caption(
        "Visualização das rotas com maiores índices de problemas, coloridas por horário e espessura por atraso."
    )

    qty_routes = st.slider("Quantidade de Rotas", 5, 100, 30)

    # Garantir que as colunas necessárias existam
    req_cols = ["ORIGIN_LAT", "ORIGIN_LON", "DEST_LAT", "DEST_LON"]
    missing = [c for c in req_cols if c not in df.columns]

    if missing:
        st.warning(f"Coordenadas ausentes: {missing}. O mapa não pode ser gerado.")
    else:
        # Calcular coluna auxiliar DELAY_PER_DISTANCE se não existir
        if (
            "DELAY_PER_DISTANCE" not in df.columns
            and "DELAY_OVERALL" in df.columns
            and "DISTANCE" in df.columns
        ):
            df["DELAY_PER_DISTANCE"] = df["DELAY_OVERALL"] / df["DISTANCE"].replace(
                0, 1
            )

        # Gerar Mapa Avançado
        with st.spinner("Gerando mapa complexo..."):
            fig_map = criar_mapa_rotas_avancado(df, top_n=qty_routes)
            st.plotly_chart(fig_map, use_container_width=True)

        st.info(
            """
            **Como ler este mapa:**
            - **Cores das Linhas:** Indicam o horário médio dos voos na rota (Azul/Roxo: Noite/Madrugada - Laranja: Dia).
            - **Espessura:** Indica o atraso médio na rota (Mais grossa = Mais atraso).
            - **Marcadores:** (Verde) Origem, (Vermelho) Destino, (*) Hubs Críticos.
            """
        )


# --- TAB 5: Simulador ---
with tab5:
    st.subheader("Simulador de Atrasos")
    st.markdown("Previsão de atrasos com base em Machine Learning (Random Forest).")

    # Treinar/Carregar Modelo
    @st.cache_resource
    def get_trained_model(data):
        return train_flight_model(data)

    if not df.empty:
        with st.spinner("Treinando modelo preditivo... (Isso acontece apenas uma vez)"):
            model_data = get_trained_model(df)

        if "mae" in model_data:
            st.success(
                f"Modelo treinado! Erro Médio Absoluto (MAE): {model_data['mae']:.2f} minutos"
            )

        # Layout: 2 colunas (Inputs | Resultado)
        col_input, col_result = st.columns([2, 1])

        with col_input:
            with st.container(border=True):
                st.subheader("Parâmetros do Voo")

                # Split inputs for better density
                c1, c2 = st.columns(2)
                with c1:
                    # Opções baseadas nos dados reais
                    airlines = sorted(df["AIRLINE_Description"].unique().astype(str))
                    origins = sorted(df["ORIGIN_CITY"].unique().astype(str))

                    input_airline = st.selectbox("Companhia Aérea", airlines)
                    input_origin = st.selectbox("Cidade de Origem", origins)

                    possible_dests = sorted(
                        df[df["ORIGIN_CITY"] == input_origin]["DEST_CITY"]
                        .unique()
                        .astype(str)
                    )
                    if not possible_dests:
                        possible_dests = sorted(df["DEST_CITY"].unique().astype(str))
                    input_dest = st.selectbox("Cidade de Destino", possible_dests)

                with c2:
                    input_dist = st.number_input(
                        "Distância (milhas)", min_value=50, max_value=5000, value=500
                    )
                    input_day = st.selectbox(
                        "Dia da Semana",
                        [
                            "Segunda-feira",
                            "Terça-feira",
                            "Quarta-feira",
                            "Quinta-feira",
                            "Sexta-feira",
                            "Sábado",
                            "Domingo",
                        ],
                        index=0,
                    )
                    input_time = st.slider("Horário da Partida (Hora)", 0, 23, 8)

                st.markdown("---")
                predict_btn = st.button(
                    "Simular Atraso 🎲", type="primary", use_container_width=True
                )

        with col_result:
            with st.container(border=True):
                st.subheader("Resultado")

                if predict_btn:
                    input_data = {
                        "AIRLINE_Description": input_airline,
                        "ORIGIN_CITY": input_origin,
                        "DEST_CITY": input_dest,
                        "DISTANCE": input_dist,
                        "DAY_OF_WEEK": input_day,
                        "TIME_HOUR": input_time,
                    }

                    try:
                        prediction = predict_flight_delay(model_data, input_data)

                        st.metric("Atraso Estimado", f"{prediction:.0f} min")

                        if prediction < 15:
                            st.success("No Horário")
                            st.caption("Atraso insignificante (< 15 min)")
                        elif prediction < 45:
                            st.warning("Atraso Moderado")
                            st.caption("Pode impactar conexões.")
                        else:
                            st.error("Atraso Alto")
                            st.caption("Grande chance de problemas.")

                    except Exception as e:
                        st.error(f"Erro: {e}")
                else:
                    st.info("Configure os parâmetros ao lado e clique em Simular.")
    else:
        st.warning("Dados não disponíveis para treinar o modelo.")
