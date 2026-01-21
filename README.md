# Machine Learning & AI Preditiva

> **Modelos que aprendem e preveem.**
> Uma coleção de algoritmos de aprendizado supervisionado e por reforço aplicados a problemas reais de finanças, RH, engenharia e mercado imobiliário.

![Demonstração do Sistema](https://github.com/vitoriapguimaraes/machineLearning/blob/main/demo/navigation.gif)

## Objetivo

Demonstrar a aplicação prática de técnicas avançadas de Machine Learning para resolução de problemas de negócio. Este repositório centraliza projetos que vão desde a previsão de valores (Regressão) e classificação de riscos até agentes autônomos de investimento (Reinforcement Learning), todos integrados em uma interface web interativa para facilitar a experimentação e visualização dos resultados.

## Projetos e Funcionalidades

O portfólio está organizado em módulos independentes, acessíveis através de um **Multi-Page App**:

| Projeto / Módulo    | Descrição e Aplicação                                                                          | Stack e Modelos        |
| :------------------ | :--------------------------------------------------------------------------------------------- | :--------------------- |
| Predição de Salário | Estimativa salarial baseada em anos de experiência e nível educacional (Polinomial).           | Scikit-Learn, Ply      |
| Previsão de Aluguel | Modelo para estimar valores de imóveis com base em suas características físicas e localização. | Regressão Linear       |
| Previsão de Vendas  | Forecasting de séries temporais para planejamento de demanda e estoque.                        | Statsmodels (ETS/Holt) |
| Score de Crédito    | Classificação de risco de crédito para aprovação de empréstimos bancários.                     | Random Forest, KNN     |
| Trading Bot (RL)    | Agente autônomo treinado com Q-Learning para operar no mercado financeiro (Simulação).         | Reinforcement Learning |
| Rotatividade (RH)   | Análise de fatores que levam ao _turnover_ e predição de saída de funcionários.                | XGBoost                |
| Risco Bancário      | Avaliação detalhada de perfis de clientes para mitigação de riscos financeiros.                | Regressão Logística    |
| Padrões em Voos     | Análise de tráfego aéreo e Simulador de Atrasos com Machine Learning.                          | Random Forest, Plotly  |

## Tecnologias Utilizadas

- **Linguagem**: Python 3.10+
- **Framework Web**: Streamlit
- **Machine Learning**: Scikit-Learn, XGBoost, Statsmodels
- **Manipulação de Dados**: Pandas, NumPy
- **Visualização**: Plotly Express, Matplotlib, Seaborn

## Como Executar

Siga os passos abaixo para rodar a aplicação localmente:

1. **Clone o repositório**

   ```bash
   git clone https://github.com/vitoriapguimaraes/dataScience.git
   cd dataScience/machineLearning
   ```

2. **Instale as dependências**
   Recomenda-se usar um ambiente virtual (`venv`).

   ```bash
   pip install -r requirements.txt
   ```

3. **Execute a aplicação**

   ```bash
   streamlit run Painel.py
   ```

4. **Acesse no navegador**
   O app abrirá automaticamente em: `http://localhost:8501`

## Estrutura de Diretórios

```dash
machineLearning/
├── data/                # Datasets brutos e processados
├── notebooks/           # Jupyter Notebooks para treino e exploração
├── pages/               # Páginas da aplicação Streamlit (cada projeto)
│   ├── 1-Predicao_de_Salario_por_Estudos.py
│   ├── 2-Previsao_Aluguel_Imoveis.py
│   ├── 3-Previsao_de_Vendas.py
│   ├── ...
│   └── 8-Avaliacao_de_Padroes_de_Voos.py
├── utils/               # Módulos auxiliares e modelos
│   ├── load_file.py     # Carregamento de dados
│   ├── models.py        # Definição e treino dos modelos ML
│   ├── ui.py            # Componentes visuais
│   └── visualizations.py # Gráficos
├── Painel.py            # Página Inicial
└── README.md            # Documentação
```

## Status

✅ Concluído

## Mais Sobre Mim

Acesse os arquivos disponíveis na [Pasta Documentos](https://github.com/vitoriapguimaraes/vitoriapguimaraes/tree/main/DOCUMENTOS) para mais informações sobre minhas qualificações e certificações.
