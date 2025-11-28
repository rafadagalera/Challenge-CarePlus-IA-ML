# Análise de Telemedicina Medicare - Insights por Região e Demografia

Este projeto realiza uma análise completa do uso de telemedicina no Medicare, utilizando técnicas de análise exploratória de dados e machine learning para identificar padrões por região e características demográficas.

## 📊 Dataset

O dataset utilizado é o **Medicare Telehealth Trends Public Use File** (TMEDTREND_PUBLIC_250827.csv), contendo dados agregados sobre o uso de telemedicina no Medicare de 2020 a 2025.

### Variáveis Principais:
- **Pct_Telehealth**: Percentual de beneficiários que utilizaram telemedicina (variável alvo)
- **Bene_Geo_Desc**: Descrição geográfica (National, State, etc.)
- **Bene_RUCA_Desc**: Classificação rural/urbana (Rural, Urban)
- **Bene_Race_Desc**: Raça/etnia do beneficiário
- **Bene_Sex_Desc**: Sexo do beneficiário
- **Bene_Age_Desc**: Faixa etária
- **Bene_Mdcr_Entlmt_Stus**: Status de elegibilidade (Aged, Disabled, ESRD)
- **Year**: Ano dos dados

## 🎯 Objetivos

1. Realizar análise exploratória de dados (EDA) com correlações, distribuições e heatmaps
2. Criar modelo de classificação binária (Alta vs Baixa Adoção)
3. Criar modelo de regressão para prever Pct_Telehealth
4. Responder 3 perguntas de pesquisa sobre padrões de adoção

## 🔬 Perguntas de Pesquisa

### Pergunta 1: Previsão de Alta Adoção
**É possível prever se uma combinação de características regionais e demográficas leva uma região a estar entre os grupos de alta adoção de telemedicina?**

**Resposta:** Sim! Os modelos de classificação apresentaram:
- **Random Forest**: Accuracy de 78.9%, ROC-AUC de 0.863
- **Logistic Regression**: Accuracy de 68.6%, ROC-AUC de 0.748

**Características mais importantes:**
1. Ano (Year): 30.1% de importância
2. Região Geográfica (Geo_Encoded): 16.8%
3. Status de Elegibilidade (Status_Encoded): 13.4%
4. Total de Beneficiários Elegíveis: 13.1%
5. Faixa Etária (Age_Encoded): 11.7%

### Pergunta 2: Fatores de Impacto
**Quais fatores demográficos e regionais têm maior impacto na taxa de adoção de telemedicina?**

**Resposta:** Principais descobertas:

1. **Região (RUCA)**:
   - Áreas Urbanas: 20.1%
   - Áreas Rurais: 16.9%
   - Diferença: 3.2 pontos percentuais

2. **Faixa Etária**:
   - 0-64 anos: 31.8% (maior adoção)
   - 65-74 anos: 18.1%
   - 75-84 anos: 17.5%
   - 85+ anos: 17.2%

3. **Raça/Etnia**:
   - American Indian/Alaska Native: 23.1%
   - Black/African American: 22.2%
   - Hispanic: 21.8%
   - Asian/Pacific Islander: 19.5%
   - Non-Hispanic White: 19.3%

4. **Sexo**:
   - Mulheres: 20.5%
   - Homens: 17.9%

### Pergunta 3: Evolução Temporal
**Como a taxa de adoção de telemedicina varia ao longo do tempo e quais grupos demográficos mostraram maior crescimento?**

**Resposta:** A adoção de telemedicina diminuiu significativamente após o pico de 2020:

- **2020**: 32.4% (pico durante pandemia)
- **2021**: 23.5%
- **2022**: 19.3%
- **2023**: 16.2%
- **2024**: 15.9%
- **2025**: 14.1%

**Tendências por grupo:**
- Todos os grupos demográficos mostraram declínio após 2020
- Grupos mais jovens (0-64) mantiveram taxas relativamente mais altas
- Áreas urbanas tiveram declínio mais acentuado que áreas rurais

## 📈 Modelos de Machine Learning

### Classificação Binária (Alta vs Baixa Adoção)

**Threshold:** Mediana de Pct_Telehealth (17.76%)

| Modelo | Accuracy | ROC-AUC | Precision | Recall | F1-Score |
|--------|----------|---------|-----------|--------|----------|
| Random Forest | 78.9% | 0.863 | 0.79 | 0.79 | 0.79 |
| Logistic Regression | 68.6% | 0.748 | 0.69 | 0.69 | 0.69 |

### Regressão (Predição de Pct_Telehealth)

| Modelo | R² | RMSE | MAE |
|--------|----|----|-----|
| Random Forest | 0.469 | 0.095 | 0.068 |
| Linear Regression | 0.259 | 0.112 | 0.086 |

## 📁 Estrutura do Projeto

```
ChallengeIA/
├── Medicare Telehealth Trends/
│   └── 2025-Q1/
│       └── TMEDTREND_PUBLIC_250827.csv
├── telehealth_analysis.py      # Script principal de análise
├── requirements.txt             # Dependências Python
├── README.md                    # Este arquivo
├── analysis_output.txt          # Saída completa da análise
└── Gráficos gerados:
    ├── distribuicao_variaveis.png
    ├── heatmap_correlacao.png
    ├── matriz_confusao_classificacao.png
    ├── curvas_roc.png
    ├── predicao_regressao.png
    └── evolucao_temporal.png
```

## 🚀 Como Executar

### 1. Instalar Dependências

```bash
# Criar ambiente virtual
python3 -m venv venv

# Ativar ambiente virtual
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows

# Instalar pacotes
pip install -r requirements.txt
```

### 2. Executar Análise

```bash
python telehealth_analysis.py
```

O script irá:
- Carregar e limpar os dados
- Realizar análise exploratória
- Treinar modelos de classificação e regressão
- Gerar gráficos e métricas
- Responder as 3 perguntas de pesquisa

## 📊 Gráficos Gerados

1. **distribuicao_variaveis.png**: Distribuição de Pct_Telehealth (histograma, box plot, Q-Q plot)
2. **heatmap_correlacao.png**: Matriz de correlação entre variáveis
3. **matriz_confusao_classificacao.png**: Matrizes de confusão dos modelos de classificação
4. **curvas_roc.png**: Curvas ROC comparando modelos de classificação
5. **predicao_regressao.png**: Gráficos de predição vs valores reais (regressão)
6. **evolucao_temporal.png**: Evolução temporal da adoção por diferentes grupos demográficos

## 🔍 Principais Insights

1. **Fatores de Alta Adoção:**
   - Beneficiários mais jovens (0-64 anos) têm maior taxa de adoção
   - Áreas urbanas apresentam maior adoção que rurais
   - Mulheres têm maior taxa de adoção que homens

2. **Tendência Temporal:**
   - Pico de adoção em 2020 (32.4%) durante a pandemia
   - Declínio constante nos anos seguintes
   - Estabilização em torno de 14-16% a partir de 2023

3. **Capacidade Preditiva:**
   - Modelos conseguem prever alta/baixa adoção com boa acurácia (78.9%)
   - Ano é a variável mais importante (reflete tendência temporal)
   - Características geográficas e demográficas também são relevantes

## 📦 Dependências

- pandas >= 2.1.4
- numpy >= 1.26.2
- matplotlib >= 3.8.2
- seaborn >= 0.13.0
- scikit-learn >= 1.4.0
- scipy >= 1.12.0

## 📝 Notas Técnicas

- **Classe Binária**: Criada usando mediana de Pct_Telehealth como threshold
- **Validação**: Cross-validation com 5 folds
- **Normalização**: StandardScaler aplicado para modelos lineares
- **Tratamento de Dados**: Remoção de linhas com valores NaN e agregações totais ("All")

## 👤 Autor

Análise desenvolvida para o ChallengeIA - Análise de Telemedicina Medicare

