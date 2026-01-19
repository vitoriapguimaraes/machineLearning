import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import random
from utils.load_file import load_dataset
from utils.ui import setup_sidebar, add_back_to_top

# ==========================================
# Configuração da Página
# ==========================================
st.set_page_config(page_title="Trading com Q-Learning", page_icon="📈", layout="wide")

st.title("Robô de Trading com Q-Learning")

setup_sidebar()
add_back_to_top()


# Funções de Carregamento e Processamento
@st.cache_data
def load_data():
    try:
        df = load_dataset("trading.csv")
        # Renomear colunas para facilitar
        df = df.rename(
            columns={
                "AAPL.Open": "Open",
                "AAPL.High": "High",
                "AAPL.Low": "Low",
                "AAPL.Close": "Close",
                "AAPL.Volume": "Volume",
                "AAPL.Adjusted": "Adjusted",
            }
        )
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date")
        return df
    except Exception as e:
        st.error(f"Erro ao carregar dados: {e}")
        return pd.DataFrame()


# Lógica do Q-Learning
class QLearningAgent:
    def __init__(
        self,
        state_size,
        action_size=3,
        alpha=0.1,
        gamma=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
    ):
        self.state_size = state_size  # Tamanho da janela de dias
        self.action_size = action_size  # 0: Sit (Hold), 1: Buy, 2: Sell
        self.memory = {}  # Q-Table: dict mapping state_str -> [q_values]
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

    def get_state(self, data, t, n):
        """
        Gera o estado com base nas diferenças de preço dos últimos n dias.
        Retorna uma string representando o estado discretizado.
        """
        d = t - n + 1
        block = data[d : t + 1] if d >= 0 else -d * [data[0]] + list(data[0 : t + 1])
        res = []
        for i in range(n - 1):
            res.append(self.sigmoid(block[i + 1] - block[i]))
        return "".join([str(int(x * 10)) for x in res])  # Simple discretization

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def act(self, state, is_eval=False):
        if not is_eval and random.random() <= self.epsilon:
            return random.randrange(self.action_size)

        if state not in self.memory:
            return random.randrange(self.action_size)  # Explore if unknown

        options = self.memory[state]
        return np.argmax(options)

    def learn(self, state, action, reward, next_state):
        if state not in self.memory:
            self.memory[state] = np.zeros(self.action_size)

        if next_state not in self.memory:
            self.memory[next_state] = np.zeros(self.action_size)

        target = reward + self.gamma * np.amax(self.memory[next_state])
        self.memory[state][action] = (1 - self.alpha) * self.memory[state][
            action
        ] + self.alpha * target

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


# Função auxiliar para formatar moeda
def format_currency(val):
    return f"${val:.2f}"


# Load Data
df = load_data()
if df.empty:
    st.stop()

# Data Split
train_size = int(len(df) * 0.8)
train_data = list(df["Close"][:train_size])
test_data = list(df["Close"][train_size:])
test_dates = list(df["Date"][train_size:])

# ==========================================
# Interface - Tabs
# ==========================================
tab1, tab2, tab3 = st.tabs(
    [
        "Visão Geral",
        "Treinamento do Robô",
        "Teste e Resultados",
    ]
)

with tab1:

    st.markdown(
        """
        Este projeto utiliza **Aprendizado por Reforço (Q-Learning)** para treinar um agente capaz de negociar ações. O agente observa o mercado (estados) e toma decisões de Compra, Venda ou Manter (ações) para maximizar o lucro.
        """
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Total de registros", len(df))
    col2.metric("Período de Treino", len(train_data))
    col3.metric("Período de Teste", len(test_data))

    st.subheader("Histórico de Preços (AAPL)")
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=df["Date"],
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
            )
        ]
    )
    fig.update_layout(xaxis_rangeslider_visible=False, height=500)
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Treinamento do Agente")

    with st.container(border=True):
        st.markdown("#### Configurações do Agente")
        col1, col2, col3 = st.columns(3)
        with col1:
            episodes = st.slider(
                "Número de Episódios (Treino)",
                1,
                100,
                10,
                help="Quantas vezes o robô vai percorrer os dados de treino.",
            )
        with col2:
            window_size = st.slider(
                "Tamanho da Janela (Dias)",
                2,
                30,
                10,
                help="Quantos dias passados o robô 'enxerga' para tomar decisão.",
            )
        with col3:
            initial_money = st.number_input(
                "Capital Inicial ($)", value=10000.0, step=1000.0
            )

        st.markdown("##### Hiperparâmetros")
        col1, col2, col3 = st.columns(3)
        with col1:
            alpha = st.slider("Alpha (Taxa de Aprendizado)", 0.01, 1.0, 0.5)
        with col2:
            gamma = st.slider("Gamma (Fator de Desconto)", 0.01, 1.0, 0.9)
        with col3:
            epsilon = st.slider("Epsilon (Exploração Inicial)", 0.1, 1.0, 1.0)

        if "agent" not in st.session_state:
            st.session_state.agent = None

        start_train = st.button("Iniciar Treinamento", type="primary")

    training_placeholder = st.empty()

    if start_train:
        agent = QLearningAgent(window_size, alpha=alpha, gamma=gamma, epsilon=epsilon)
        data = train_data
        data_length = len(data) - 1

        progress_bar = st.progress(0)
        status_text = st.empty()

        profit_history = []

        for e in range(episodes + 1):
            state = agent.get_state(data, 0, window_size + 1)
            total_profit = 0
            agent.inventory = []  # Inventory of bought stocks

            for t in range(data_length):
                action = agent.act(state)

                # logic
                next_state = agent.get_state(data, t + 1, window_size + 1)
                reward = 0

                if action == 1:  # Buy
                    agent.inventory.append(data[t])
                    # No immediate reward for buying, potential negative reward?
                    # Notebook logic often gives 0 or small penalty

                elif action == 2 and len(agent.inventory) > 0:  # Sell
                    bought_price = agent.inventory.pop(0)
                    profit = data[t] - bought_price
                    reward = max(
                        profit, 0
                    )  # Positive reward only if profit? Or raw profit? Standard is raw profit.
                    # Simple robust reward:
                    if profit > 0:
                        reward = 1
                    elif profit <= 0:
                        reward = -1

                    total_profit += profit

                done = True if t == data_length - 1 else False
                agent.learn(state, action, reward, next_state)
                state = next_state

                if done:
                    profit_history.append(total_profit)
                    # print(f"Episode {e}/{episodes} done. Profit: {total_profit}")

            progress_bar.progress(e / episodes if episodes > 0 else 1)
            status_text.text(
                f"Episódio {e}/{episodes} - Lucro Total: {format_currency(total_profit)}"
            )

        st.session_state.agent = agent
        st.success("Treinamento Concluído!")
        st.info(
            "Acesse a aba 'Teste e Resultados' para visualizar a performance do robô."
        )

        # Plot training curve
        fig_train = go.Figure()
        fig_train.add_trace(
            go.Scatter(
                y=profit_history, mode="lines+markers", name="Lucro por Episódio"
            )
        )
        fig_train.update_layout(
            title="Evolução do Lucro durante o Treinamento",
            xaxis_title="Episódio",
            yaxis_title="Lucro Total ($)",
        )
        st.plotly_chart(fig_train, use_container_width=True)

with tab3:
    st.subheader("Teste do Modelo em Novos Dados")

    if st.session_state.agent is None:
        st.warning(
            "⚠️ Você precisa treinar o robô primeiro na aba 'Treinamento do Robô'."
        )
    else:
        # if st.button("Executar Teste"):
        agent = st.session_state.agent
        data = test_data
        data_length = len(data) - 1
        state = agent.get_state(data, 0, window_size + 1)
        total_profit = 0
        agent.inventory = []

        buy_signals = []
        sell_signals = []
        history = []

        # Simulation Loop
        for t in range(data_length):
            action = agent.act(state, is_eval=True)
            next_state = agent.get_state(data, t + 1, window_size + 1)

            current_price = data[t]
            current_date = test_dates[t]

            if action == 1:  # Buy
                agent.inventory.append(current_price)
                buy_signals.append((current_date, current_price))
                history.append(
                    f"🟢 [COMPRA] Dia {current_date.strftime('%d/%m/%Y')} @ {format_currency(current_price)}"
                )

            elif action == 2 and len(agent.inventory) > 0:  # Sell
                bought_price = agent.inventory.pop(0)
                profit = current_price - bought_price
                total_profit += profit
                sell_signals.append((current_date, current_price))
                icon = "💰" if profit > 0 else "🔻"
                history.append(
                    f"{icon} [VENDA]  Dia {current_date.strftime('%d/%m/%Y')} @ {format_currency(current_price)} | Lucro: {format_currency(profit)}"
                )

            state = next_state

        # Results
        col1, col2 = st.columns(2)
        invest_return = (
            ((total_profit + initial_money) - initial_money) / initial_money * 100
        )

        with col1:
            st.metric(
                label="Lucro Total Final",
                value=format_currency(total_profit),
                delta=f"{invest_return:.2f}%",
            )
        with col2:
            st.metric(
                label="Capital Final Estimado",
                value=format_currency(initial_money + total_profit),
            )

        # Visualization
        st.divider()
        st.markdown("### 📉 Comportamento do Agente no Gráfico")

        fig_test = go.Figure()
        # Price Line
        fig_test.add_trace(
            go.Scatter(
                x=test_dates,
                y=data,
                mode="lines",
                name="Preço Ação",
                line=dict(color="gray", width=1),
            )
        )

        # Buy Markers
        if buy_signals:
            b_dates, b_prices = zip(*buy_signals)
            fig_test.add_trace(
                go.Scatter(
                    x=b_dates,
                    y=b_prices,
                    mode="markers",
                    name="Compra",
                    marker=dict(color="green", symbol="triangle-up", size=10),
                )
            )

        # Sell Markers
        if sell_signals:
            s_dates, s_prices = zip(*sell_signals)
            fig_test.add_trace(
                go.Scatter(
                    x=s_dates,
                    y=s_prices,
                    mode="markers",
                    name="Venda",
                    marker=dict(color="red", symbol="triangle-down", size=10),
                )
            )

        fig_test.update_layout(height=600, title="Atuações de Compra e Venda")
        st.plotly_chart(fig_test, use_container_width=True)

        # Log
        with st.expander("📜 Log Detalhado das Operações"):
            for h in history:
                st.write(h)
