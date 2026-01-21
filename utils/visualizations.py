import streamlit as st
import plotly.express as px

# Paleta de cores padronizada (Baseada no Bootstrap / Material Design)
# Azul (Primary), Indigo, Roxo, Rosa, Vermelho, Laranja, Amarelo, Verde, Teal, Ciano
COLOR_PALETTE = [
    "#007BFF",  # Azul Principal
    "#DC3545",  # Vermelho
    "#6610F2",  # Indigo
    "#6F42C1",  # Roxo
    "#E83E8C",  # Rosa
    "#FD7E14",  # Laranja
    "#FFC107",  # Amarelo
    "#28A745",  # Verde
    "#20C997",  # Teal
    "#17A2B8",  # Ciano
]


def plot_pie(df, names, height=350, title=None):
    """
    Renderiza um gráfico de pizza.
    """
    fig = px.pie(
        df,
        names=names,
        title=title,
        color_discrete_sequence=COLOR_PALETTE,
    )
    fig.update_layout(height=height)
    st.plotly_chart(fig, use_container_width=True)


def plot_bar(
    df,
    x_col,
    y_col,
    title=None,
    orientation="v",
    color=None,
    height=350,
    labels=None,
    show_legend=True,
    color_map=None,
):
    """
    Renderiza um gráfico de barras.
    """
    fig = px.bar(
        df,
        x=x_col,
        y=y_col,
        orientation=orientation,
        title=title,
        color=color,
        height=height,
        color_discrete_sequence=COLOR_PALETTE,
        labels=labels,
        color_discrete_map=color_map,
    )
    if not show_legend:
        fig.update_layout(showlegend=False)

    st.plotly_chart(fig, use_container_width=True)


def plot_histogram(
    df,
    x,
    color=None,
    title=None,
    barmode="group",
    show_yaxis_title=True,
    labels=None,
    color_map=None,
):
    """
    Renderiza um histograma.
    """
    fig = px.histogram(
        df,
        x=x,
        color=color,
        title=title,
        barmode=barmode,
        color_discrete_sequence=COLOR_PALETTE,
        labels=labels,
        color_discrete_map=color_map,
    )
    if not show_yaxis_title:
        fig.update_layout(yaxis_title=None)
    st.plotly_chart(fig, use_container_width=True)


def plot_boxplot(
    df,
    x,
    y,
    color=None,
    title=None,
    show_xaxis_title=True,
    color_map=None,
    labels=None,
):
    """
    Renderiza um boxplot.
    """
    fig = px.box(
        df,
        x=x,
        y=y,
        color=color,
        title=title,
        color_discrete_sequence=COLOR_PALETTE,
        color_discrete_map=color_map,
        labels=labels,
    )
    if not show_xaxis_title:
        fig.update_layout(xaxis_title=None)
    st.plotly_chart(fig, use_container_width=True)


def plot_heatmap(df, numeric_cols, height=600):
    """
    Renderiza um mapa de calor de correlação.
    """
    corr = df[numeric_cols].corr()
    fig = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        color_continuous_scale=[COLOR_PALETTE[1], "#FFFFFF", COLOR_PALETTE[0]],
        origin="lower",
        range_color=[-1, 1],
    )
    fig.update_layout(height=height)
    st.plotly_chart(fig, use_container_width=True)


def show_grouped_metrics(df, grouped_columns):
    """
    Exibe as métricas descritivas agrupadas em containers dinâmicos.

    Args:
        df: DataFrame com os dados.
        grouped_columns: Dicionário onde keys são títulos dos grupos e values são listas de colunas.
    """
    st.subheader("Estatísticas Descritivas por Grupo")

    for group_title, cols in grouped_columns.items():
        with st.container(border=True):
            st.markdown(group_title)
            available_cols = [c for c in cols if c in df.columns]
            if available_cols:
                st.dataframe(
                    df[available_cols].describe(include="all"), use_container_width=True
                )
            else:
                st.info(f"Nenhuma coluna disponível para o grupo: {group_title}")


def show_univariate_grid(
    df, numeric_cols, categorical_cols, target_col="Categoria", num_cols=3
):
    """
    Exibe uma grade com histogramas de todas as colunas.
    """
    with st.container(border=True):
        st.subheader("Todas as distribuições")
        all_cols = numeric_cols + categorical_cols

        # Evitar redundância se a variável alvo estiver na lista
        if target_col in all_cols:
            all_cols.remove(target_col)

        cols = st.columns(num_cols)
        for i, col in enumerate(all_cols):
            with cols[i % num_cols]:
                fig_all = px.histogram(
                    df,
                    x=col,
                    color=target_col,
                    title=col,
                    barmode="group",
                    color_discrete_sequence=COLOR_PALETTE,
                )
                fig_all.update_layout(
                    showlegend=False,
                    height=300,
                    margin=dict(l=0, r=0, t=30, b=0),
                    yaxis_title=None,
                )
                st.plotly_chart(
                    fig_all, use_container_width=True, key=f"univariate_{col}"
                )


def plot_regression(
    df,
    x_col,
    y_col,
    model,
    title="Regressão Linear",
    x_label="Horas de Estudo",
    y_label="Salário",
):
    """
    Plot the scatter plot of data and the regression line.

    Args:
        df: DataFrame containing the data.
        x_col: Name of the column for x-axis.
        y_col: Name of the column for y-axis.
        model: Trained LinearRegression model.
    """
    import numpy as np
    import plotly.graph_objects as go

    # Create Scatter plot for actual data
    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        title=title,
        labels={x_col: x_label, y_col: y_label},
        color_discrete_sequence=[COLOR_PALETTE[0]],
    )

    # Add Regression Line
    # Generate a range of X values for the line
    x_range = np.linspace(df[x_col].min(), df[x_col].max(), 100).reshape(-1, 1)
    y_range = model.predict(x_range)

    fig.add_trace(
        go.Scatter(
            x=x_range.flatten(),
            y=y_range,
            mode="lines",
            name="Regressão Linear",
            line=dict(color=COLOR_PALETTE[1], width=3),
        )
    )

    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)


def plot_forecast(
    history_series,
    forecast_series,
    title="Previsão de Vendas",
    x_label="Data",
    y_label="Vendas",
    history_label="Dados Históricos",
    forecast_label="Previsão",
):
    """
    Plots the historical data and the forecasted values.

    Args:
        history_series: pandas Series of historical data.
        forecast_series: pandas Series of forecasted data.
        title (str): Plot title.
        x_label (str): X-axis label.
        y_label (str): Y-axis label.
        history_label (str): Label for the historical data trace.
        forecast_label (str): Label for the forecast/fitted data trace.
    """
    import plotly.graph_objects as go

    fig = go.Figure()

    # Historical Data
    fig.add_trace(
        go.Scatter(
            x=history_series.index,
            y=history_series.values,
            mode="lines",
            name=history_label,
            line=dict(color=COLOR_PALETTE[0]),
        )
    )

    # Forecast Data
    fig.add_trace(
        go.Scatter(
            x=forecast_series.index,
            y=forecast_series.values,
            mode="lines",
            name=forecast_label,
            line=dict(color=COLOR_PALETTE[1], dash="dash"),
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_confusion_matrix(
    cm, x_labels, y_labels, title="Matriz de Confusão", height=400, color_scale="Blues"
):
    """
    Renderiza uma matriz de confusão.
    """
    fig = px.imshow(
        cm,
        text_auto=True,
        aspect="equal",
        color_continuous_scale=color_scale,
        x=x_labels,
        y=y_labels,
        title=title,
        labels=dict(x="Predito", y="Real", color="Qtd"),
    )
    fig.update_layout(height=height)
    st.plotly_chart(fig, use_container_width=True)
