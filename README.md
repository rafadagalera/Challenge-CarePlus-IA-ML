## Este repositório serve como entrega da Sprint 1 da Challenge da Turma 3ESPA em parceria com a CarePlus, para a disciplina de IA e Machine Learning
### Feito por:
#### Beatriz Rocha
#### Luis Alberto
#### Isabelle Torricelli
#### Rafael Nascimento


O dataset utilizado é o TMEDTREND_PUBLIC_250827.csv (fornecido pelo professor na entrega da Sprint), contendo dados agregados sobre o uso de telemedicina de 2020 a 2025.

### Variáveis Principais:
- **Pct_Telehealth**: Percentual de beneficiários que utilizaram telemedicina (variável alvo)
- **Bene_Geo_Desc**: Descrição geográfica (National, State, etc.)
- **Bene_RUCA_Desc**: Classificação rural/urbana (Rural, Urban)
- **Bene_Race_Desc**: Raça/etnia do beneficiário
- **Bene_Sex_Desc**: Sexo do beneficiário
- **Bene_Age_Desc**: Faixa etária
- **Bene_Mdcr_Entlmt_Stus**: Status de elegibilidade (Aged, Disabled, ESRD)
- **Year**: Ano dos dados


## Perguntas discutidas durante este estudo

### Pergunta 1: Previsão de Alta Adoção
**É possível prever se uma combinação de características regionais e demográficas leva uma região a estar entre os grupos de alta adoção de telemedicina?**

### Pergunta 2: Fatores de Impacto
**Quais fatores demográficos e regionais têm maior impacto na taxa de adoção de telemedicina?**

### Pergunta 3: Evolução Temporal
**Como a taxa de adoção de telemedicina varia ao longo do tempo e quais grupos demográficos mostraram maior crescimento?**

## Estrutura do Projeto

```
ChallengeIA/

│── TMEDTREND_PUBLIC_250827.csv
├── telehealth_analysis.py      
├── requirements.txt             
├── README.md                    
├── analysis_output.txt         
└── Gráficos gerados:
    ├── distribuicao_variaveis.png
    ├── heatmap_correlacao.png
    ├── matriz_confusao_classificacao.png
    ├── curvas_roc.png
    ├── predicao_regressao.png
    └── evolucao_temporal.png
```

## Como Executar

```
python3 -m venv venv

source venv/bin/activate  # Linux/Mac
ou
venv\Scripts\activate  # Windows

pip install -r requirements.txt

python telehealth_analysis.py
```

O script irá:
- Carregar e limpar os dados
- Realizar análise exploratória
- Treinar modelos de classificação e regressão
- Gerar gráficos e métricas
- Responder as 3 perguntas de pesquisa

## Gráficos Gerados

1. **distribuicao_variaveis.png**: Distribuição de Pct_Telehealth (histograma, box plot, Q-Q plot)
2. **heatmap_correlacao.png**: Matriz de correlação entre variáveis
3. **matriz_confusao_classificacao.png**: Matrizes de confusão dos modelos de classificação
4. **curvas_roc.png**: Curvas ROC comparando modelos de classificação
5. **predicao_regressao.png**: Gráficos de predição vs valores reais (regressão)
6. **evolucao_temporal.png**: Evolução temporal da adoção por diferentes grupos demográficos

##  Principais Insights

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

