"""
Análise de Telemedicina Medicare - Insights por Região e Demografia
Análise exploratória, modelos de ML e respostas a perguntas de pesquisa
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    roc_auc_score, roc_curve, mean_squared_error, r2_score,
    mean_absolute_error
)
import warnings
warnings.filterwarnings('ignore')

# Configuração de visualização
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    plt.style.use('seaborn-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

print("="*80)
print("ANÁLISE DE TELEMEDICINA MEDICARE - INSIGHTS POR REGIÃO E DEMOGRAFIA")
print("="*80)

# ============================================================================
# 1. CARREGAMENTO E PREPARAÇÃO DOS DADOS
# ============================================================================

print("\n[1] Carregando dados...")
df = pd.read_csv('TMEDTREND_PUBLIC_250827.csv')

print(f"Shape inicial: {df.shape}")
print(f"\nColunas: {df.columns.tolist()}")
print(f"\nPrimeiras linhas:")
print(df.head())

# Remover linhas com valores vazios em Pct_Telehealth
df_clean = df.dropna(subset=['Pct_Telehealth']).copy()
print(f"\nShape após remover NaN em Pct_Telehealth: {df_clean.shape}")

# Filtrar apenas linhas com dados válidos (não "All" em todas as dimensões)
# Vamos manter linhas que tenham pelo menos uma característica específica
df_analysis = df_clean[
    ~((df_clean['Bene_Geo_Desc'] == 'All') & 
      (df_clean['Bene_Race_Desc'] == 'All') & 
      (df_clean['Bene_Sex_Desc'] == 'All') & 
      (df_clean['Bene_Age_Desc'] == 'All') &
      (df_clean['Bene_RUCA_Desc'] == 'All'))
].copy()

print(f"Shape após filtrar agregações totais: {df_analysis.shape}")

# Criar variável binária de Alta/Baixa Adoção
# Usaremos a mediana como threshold
median_pct = df_analysis['Pct_Telehealth'].median()
df_analysis['Alta_Adocao'] = (df_analysis['Pct_Telehealth'] >= median_pct).astype(int)

print(f"\nMediana de Pct_Telehealth: {median_pct:.4f}")
print(f"Distribuição de Alta_Adocao:")
print(df_analysis['Alta_Adocao'].value_counts())
print(f"Proporção: {df_analysis['Alta_Adocao'].mean():.3f}")

# ============================================================================
# 2. ANÁLISE EXPLORATÓRIA DE DADOS
# ============================================================================

print("\n" + "="*80)
print("[2] ANÁLISE EXPLORATÓRIA DE DADOS")
print("="*80)

# 2.1 Estatísticas descritivas
print("\n[2.1] Estatísticas Descritivas de Pct_Telehealth:")
print(df_analysis['Pct_Telehealth'].describe())

# 2.2 Distribuição de Pct_Telehealth
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Histograma
axes[0, 0].hist(df_analysis['Pct_Telehealth'], bins=50, edgecolor='black', alpha=0.7)
axes[0, 0].axvline(median_pct, color='red', linestyle='--', linewidth=2, label=f'Mediana: {median_pct:.3f}')
axes[0, 0].set_xlabel('Pct_Telehealth')
axes[0, 0].set_ylabel('Frequência')
axes[0, 0].set_title('Distribuição de Pct_Telehealth')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Box plot
axes[0, 1].boxplot(df_analysis['Pct_Telehealth'], vert=True)
axes[0, 1].set_ylabel('Pct_Telehealth')
axes[0, 1].set_title('Box Plot de Pct_Telehealth')
axes[0, 1].grid(True, alpha=0.3)

# Distribuição por Alta/Baixa Adoção
df_analysis.boxplot(column='Pct_Telehealth', by='Alta_Adocao', ax=axes[1, 0])
axes[1, 0].set_xlabel('Alta Adoção (0=Baixa, 1=Alta)')
axes[1, 0].set_ylabel('Pct_Telehealth')
axes[1, 0].set_title('Distribuição por Classe de Adoção')
axes[1, 0].grid(True, alpha=0.3)

# Q-Q plot para normalidade
from scipy import stats
stats.probplot(df_analysis['Pct_Telehealth'], dist="norm", plot=axes[1, 1])
axes[1, 1].set_title('Q-Q Plot (Normalidade)')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('distribuicao_variaveis.png', dpi=300, bbox_inches='tight')
print("\n[✓] Gráfico de distribuição salvo: distribuicao_variaveis.png")
plt.close()

# 2.3 Análise por variáveis categóricas
print("\n[2.2] Análise por Região (RUCA):")
ruca_stats = df_analysis.groupby('Bene_RUCA_Desc')['Pct_Telehealth'].agg(['mean', 'std', 'count'])
print(ruca_stats)

print("\n[2.3] Análise por Raça:")
race_stats = df_analysis[df_analysis['Bene_Race_Desc'] != 'All'].groupby('Bene_Race_Desc')['Pct_Telehealth'].agg(['mean', 'std', 'count'])
print(race_stats)

print("\n[2.4] Análise por Idade:")
age_stats = df_analysis[df_analysis['Bene_Age_Desc'] != 'All'].groupby('Bene_Age_Desc')['Pct_Telehealth'].agg(['mean', 'std', 'count'])
print(age_stats)

print("\n[2.5] Análise por Sexo:")
sex_stats = df_analysis[df_analysis['Bene_Sex_Desc'] != 'All'].groupby('Bene_Sex_Desc')['Pct_Telehealth'].agg(['mean', 'std', 'count'])
print(sex_stats)

# 2.4 Preparação de variáveis numéricas para correlação
df_numeric = df_analysis.copy()

# Codificar variáveis categóricas para análise de correlação
le_geo = LabelEncoder()
le_race = LabelEncoder()
le_sex = LabelEncoder()
le_age = LabelEncoder()
le_ruca = LabelEncoder()
le_status = LabelEncoder()

df_numeric['Geo_Encoded'] = le_geo.fit_transform(df_numeric['Bene_Geo_Desc'].astype(str))
df_numeric['Race_Encoded'] = le_race.fit_transform(df_numeric['Bene_Race_Desc'].astype(str))
df_numeric['Sex_Encoded'] = le_sex.fit_transform(df_numeric['Bene_Sex_Desc'].astype(str))
df_numeric['Age_Encoded'] = le_age.fit_transform(df_numeric['Bene_Age_Desc'].astype(str))
df_numeric['RUCA_Encoded'] = le_ruca.fit_transform(df_numeric['Bene_RUCA_Desc'].astype(str))
df_numeric['Status_Encoded'] = le_status.fit_transform(df_numeric['Bene_Mdcr_Entlmt_Stus'].astype(str))

# Selecionar variáveis numéricas para correlação
numeric_cols = ['Year', 'Total_Bene_TH_Elig', 'Total_PartB_Enrl', 'Total_Bene_Telehealth',
                'Pct_Telehealth', 'Geo_Encoded', 'Race_Encoded', 'Sex_Encoded',
                'Age_Encoded', 'RUCA_Encoded', 'Status_Encoded', 'Alta_Adocao']

corr_matrix = df_numeric[numeric_cols].corr()

# 2.5 Heatmap de correlação
plt.figure(figsize=(14, 12))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Matriz de Correlação - Variáveis Numéricas', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('heatmap_correlacao.png', dpi=300, bbox_inches='tight')
print("\n[✓] Heatmap de correlação salvo: heatmap_correlacao.png")
plt.close()

# Correlações mais fortes com Pct_Telehealth
print("\n[2.6] Correlações com Pct_Telehealth:")
corr_with_target = corr_matrix['Pct_Telehealth'].sort_values(ascending=False)
print(corr_with_target)

# ============================================================================
# 3. PREPARAÇÃO PARA MODELOS DE MACHINE LEARNING
# ============================================================================

print("\n" + "="*80)
print("[3] PREPARAÇÃO PARA MODELOS DE ML")
print("="*80)

# Selecionar features relevantes
feature_cols = ['Year', 'Geo_Encoded', 'Race_Encoded', 'Sex_Encoded', 
                'Age_Encoded', 'RUCA_Encoded', 'Status_Encoded',
                'Total_Bene_TH_Elig', 'Total_PartB_Enrl']

X = df_numeric[feature_cols].copy()
y_class = df_numeric['Alta_Adocao'].copy()
y_reg = df_numeric['Pct_Telehealth'].copy()

# Remover NaN se houver
mask = ~(X.isna().any(axis=1) | y_class.isna() | y_reg.isna())
X = X[mask].copy()
y_class = y_class[mask].copy()
y_reg = y_reg[mask].copy()

print(f"\nShape final para ML: X={X.shape}, y_class={y_class.shape}, y_reg={y_reg.shape}")

# Split dos dados
X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(
    X, y_class, y_reg, test_size=0.2, random_state=42, stratify=y_class
)

# Normalização
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nTrain set: {X_train.shape[0]} amostras")
print(f"Test set: {X_test.shape[0]} amostras")

# ============================================================================
# 4. MODELOS DE CLASSIFICAÇÃO
# ============================================================================

print("\n" + "="*80)
print("[4] MODELOS DE CLASSIFICAÇÃO (Alta vs Baixa Adoção)")
print("="*80)

# 4.1 Random Forest Classifier
print("\n[4.1] Random Forest Classifier")
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
rf_classifier.fit(X_train, y_class_train)
y_class_pred_rf = rf_classifier.predict(X_test)
y_class_proba_rf = rf_classifier.predict_proba(X_test)[:, 1]

print(f"Accuracy: {accuracy_score(y_class_test, y_class_pred_rf):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_class_test, y_class_proba_rf):.4f}")
print("\nClassification Report:")
print(classification_report(y_class_test, y_class_pred_rf))

# Feature importance
feature_importance_rf = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': rf_classifier.feature_importances_
}).sort_values('Importance', ascending=False)
print("\nFeature Importance (Random Forest):")
print(feature_importance_rf)

# 4.2 Logistic Regression
print("\n[4.2] Logistic Regression")
lr_classifier = LogisticRegression(random_state=42, max_iter=1000)
lr_classifier.fit(X_train_scaled, y_class_train)
y_class_pred_lr = lr_classifier.predict(X_test_scaled)
y_class_proba_lr = lr_classifier.predict_proba(X_test_scaled)[:, 1]

print(f"Accuracy: {accuracy_score(y_class_test, y_class_pred_lr):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_class_test, y_class_proba_lr):.4f}")
print("\nClassification Report:")
print(classification_report(y_class_test, y_class_pred_lr))

# Cross-validation
cv_scores_rf = cross_val_score(rf_classifier, X_train, y_class_train, cv=5, scoring='roc_auc')
cv_scores_lr = cross_val_score(lr_classifier, X_train_scaled, y_class_train, cv=5, scoring='roc_auc')

print(f"\nCross-Validation ROC-AUC (RF): {cv_scores_rf.mean():.4f} (+/- {cv_scores_rf.std()*2:.4f})")
print(f"Cross-Validation ROC-AUC (LR): {cv_scores_lr.mean():.4f} (+/- {cv_scores_lr.std()*2:.4f})")


# Curvas ROC
fpr_rf, tpr_rf, _ = roc_curve(y_class_test, y_class_proba_rf)
fpr_lr, tpr_lr, _ = roc_curve(y_class_test, y_class_proba_lr)

plt.figure(figsize=(10, 8))
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {roc_auc_score(y_class_test, y_class_proba_rf):.3f})', linewidth=2)
plt.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC = {roc_auc_score(y_class_test, y_class_proba_lr):.3f})', linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.title('Curvas ROC - Modelos de Classificação')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('curvas_roc.png', dpi=300, bbox_inches='tight')
print("[✓] Curvas ROC salvas: curvas_roc.png")
plt.close()

# ============================================================================
# 5. MODELOS DE REGRESSÃO
# ============================================================================

print("\n" + "="*80)
print("[5] MODELOS DE REGRESSÃO (Predição de Pct_Telehealth)")
print("="*80)

# 5.1 Random Forest Regressor
print("\n[5.1] Random Forest Regressor")
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
rf_regressor.fit(X_train, y_reg_train)
y_reg_pred_rf = rf_regressor.predict(X_test)

mse_rf = mean_squared_error(y_reg_test, y_reg_pred_rf)
rmse_rf = np.sqrt(mse_rf)
mae_rf = mean_absolute_error(y_reg_test, y_reg_pred_rf)
r2_rf = r2_score(y_reg_test, y_reg_pred_rf)

print(f"MSE: {mse_rf:.6f}")
print(f"RMSE: {rmse_rf:.6f}")
print(f"MAE: {mae_rf:.6f}")
print(f"R²: {r2_rf:.4f}")

# 5.2 Linear Regression
print("\n[5.2] Linear Regression")
lr_regressor = LinearRegression()
lr_regressor.fit(X_train_scaled, y_reg_train)
y_reg_pred_lr = lr_regressor.predict(X_test_scaled)

mse_lr = mean_squared_error(y_reg_test, y_reg_pred_lr)
rmse_lr = np.sqrt(mse_lr)
mae_lr = mean_absolute_error(y_reg_test, y_reg_pred_lr)
r2_lr = r2_score(y_reg_test, y_reg_pred_lr)

print(f"MSE: {mse_lr:.6f}")
print(f"RMSE: {rmse_lr:.6f}")
print(f"MAE: {mae_lr:.6f}")
print(f"R²: {r2_lr:.4f}")

# Cross-validation
cv_r2_rf = cross_val_score(rf_regressor, X_train, y_reg_train, cv=5, scoring='r2')
cv_r2_lr = cross_val_score(lr_regressor, X_train_scaled, y_reg_train, cv=5, scoring='r2')

print(f"\nCross-Validation R² (RF): {cv_r2_rf.mean():.4f} (+/- {cv_r2_rf.std()*2:.4f})")
print(f"Cross-Validation R² (LR): {cv_r2_lr.mean():.4f} (+/- {cv_r2_lr.std()*2:.4f})")

# Gráficos de predição vs real
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

axes[0].scatter(y_reg_test, y_reg_pred_rf, alpha=0.5, s=20)
axes[0].plot([y_reg_test.min(), y_reg_test.max()], 
             [y_reg_test.min(), y_reg_test.max()], 'r--', lw=2)
axes[0].set_xlabel('Valor Real')
axes[0].set_ylabel('Valor Predito')
axes[0].set_title(f'Random Forest Regressor (R² = {r2_rf:.3f})')
axes[0].grid(True, alpha=0.3)

axes[1].scatter(y_reg_test, y_reg_pred_lr, alpha=0.5, s=20, color='green')
axes[1].plot([y_reg_test.min(), y_reg_test.max()], 
             [y_reg_test.min(), y_reg_test.max()], 'r--', lw=2)
axes[1].set_xlabel('Valor Real')
axes[1].set_ylabel('Valor Predito')
axes[1].set_title(f'Linear Regression (R² = {r2_lr:.3f})')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('predicao_regressao.png', dpi=300, bbox_inches='tight')
print("\n[✓] Gráficos de predição salvos: predicao_regressao.png")
plt.close()

# ============================================================================
# 6. PERGUNTAS DE PESQUISA E INSIGHTS
# ============================================================================

print("\n" + "="*80)
print("[6] PERGUNTAS DE PESQUISA E INSIGHTS")
print("="*80)

# Pergunta 1: É possível prever se uma combinação de características regionais 
# e demográficas leva uma região a estar entre os grupos de alta adoção de telemedicina?

print("\n" + "-"*80)
print("PERGUNTA 1: É possível prever se uma combinação de características")
print("regionais e demográficas leva uma região a estar entre os grupos de")
print("alta adoção de telemedicina?")
print("-"*80)

print(f"\nRESPOSTA:")
print(f"Sim, é possível prever com boa acurácia. Os modelos de classificação")
print(f"apresentaram os seguintes resultados:")
print(f"\n  Random Forest:")
print(f"    - Accuracy: {accuracy_score(y_class_test, y_class_pred_rf):.1%}")
print(f"    - ROC-AUC: {roc_auc_score(y_class_test, y_class_proba_rf):.3f}")
print(f"\n  Logistic Regression:")
print(f"    - Accuracy: {accuracy_score(y_class_test, y_class_pred_lr):.1%}")
print(f"    - ROC-AUC: {roc_auc_score(y_class_test, y_class_proba_lr):.3f}")

print(f"\nAs características mais importantes para prever alta adoção são:")
for idx, row in feature_importance_rf.head(5).iterrows():
    print(f"  {row['Feature']}: {row['Importance']:.3f}")

# Análise específica por região e demografia
print(f"\n\nAnálise por Região (RUCA):")
ruca_adoption = df_analysis.groupby('Bene_RUCA_Desc').agg({
    'Pct_Telehealth': 'mean',
    'Alta_Adocao': 'mean'
}).sort_values('Pct_Telehealth', ascending=False)
print(ruca_adoption)

print(f"\n\nAnálise por Raça:")
race_adoption = df_analysis[df_analysis['Bene_Race_Desc'] != 'All'].groupby('Bene_Race_Desc').agg({
    'Pct_Telehealth': 'mean',
    'Alta_Adocao': 'mean'
}).sort_values('Pct_Telehealth', ascending=False)
print(race_adoption)

# Pergunta 2: Quais fatores demográficos e regionais têm maior impacto 
# na taxa de adoção de telemedicina?

print("\n" + "-"*80)
print("PERGUNTA 2: Quais fatores demográficos e regionais têm maior impacto")
print("na taxa de adoção de telemedicina?")
print("-"*80)

print(f"\nRESPOSTA:")
print(f"Com base na análise de correlação e importância das features:")

# Análise de impacto por categoria
print(f"\n1. Impacto Regional (RUCA):")
ruca_impact = df_analysis[df_analysis['Bene_RUCA_Desc'] != 'All'].groupby('Bene_RUCA_Desc')['Pct_Telehealth'].mean()
urban_pct = ruca_impact.get('Urban', 0)
rural_pct = ruca_impact.get('Rural', 0)
if urban_pct > 0 and rural_pct > 0:
    diff = urban_pct - rural_pct
    print(f"   - Áreas Urbanas: {urban_pct:.1%}")
    print(f"   - Áreas Rurais: {rural_pct:.1%}")
    print(f"   - Diferença: {diff:.1%} (Urbanas têm {abs(diff)*100:.1f} pontos percentuais {'mais' if diff > 0 else 'menos'})")

print(f"\n2. Impacto por Faixa Etária:")
age_impact = df_analysis[df_analysis['Bene_Age_Desc'] != 'All'].groupby('Bene_Age_Desc')['Pct_Telehealth'].mean().sort_values(ascending=False)
for age, pct in age_impact.items():
    print(f"   - {age}: {pct:.1%}")

print(f"\n3. Impacto por Raça:")
for race, pct in race_adoption['Pct_Telehealth'].items():
    print(f"   - {race}: {pct:.1%}")

print(f"\n4. Impacto por Sexo:")
sex_impact = df_analysis[df_analysis['Bene_Sex_Desc'] != 'All'].groupby('Bene_Sex_Desc')['Pct_Telehealth'].mean()
for sex, pct in sex_impact.items():
    print(f"   - {sex}: {pct:.1%}")

# Pergunta 3: Como a taxa de adoção de telemedicina varia ao longo do tempo
# e quais grupos demográficos mostraram maior crescimento?

print("\n" + "-"*80)
print("PERGUNTA 3: Como a taxa de adoção de telemedicina varia ao longo do")
print("tempo e quais grupos demográficos mostraram maior crescimento?")
print("-"*80)

print(f"\nRESPOSTA:")

# Análise temporal
yearly_trend = df_analysis.groupby('Year')['Pct_Telehealth'].agg(['mean', 'std', 'count'])
print(f"\nEvolução Temporal da Adoção de Telemedicina:")
print(yearly_trend)

# Calcular crescimento por grupo demográfico
print(f"\n\nCrescimento por Grupo Demográfico (comparando primeiro e último ano):")

years = sorted(df_analysis['Year'].unique())
if len(years) >= 2:
    first_year = years[0]
    last_year = years[-1]
    
    # Por Raça
    print(f"\n1. Por Raça:")
    for race in df_analysis[df_analysis['Bene_Race_Desc'] != 'All']['Bene_Race_Desc'].unique():
        race_data = df_analysis[(df_analysis['Bene_Race_Desc'] == race) & 
                                (df_analysis['Bene_Geo_Desc'] == 'National')]
        first_pct = race_data[race_data['Year'] == first_year]['Pct_Telehealth'].mean()
        last_pct = race_data[race_data['Year'] == last_year]['Pct_Telehealth'].mean()
        if not pd.isna(first_pct) and not pd.isna(last_pct):
            growth = last_pct - first_pct
            print(f"   - {race}: {first_pct:.1%} → {last_pct:.1%} (Δ {growth:+.1%})")
    
    # Por Região
    print(f"\n2. Por Região (RUCA):")
    for ruca in ['Urban', 'Rural']:
        ruca_data = df_analysis[(df_analysis['Bene_RUCA_Desc'] == ruca) & 
                                (df_analysis['Bene_Geo_Desc'] == 'National')]
        first_pct = ruca_data[ruca_data['Year'] == first_year]['Pct_Telehealth'].mean()
        last_pct = ruca_data[ruca_data['Year'] == last_year]['Pct_Telehealth'].mean()
        if not pd.isna(first_pct) and not pd.isna(last_pct):
            growth = last_pct - first_pct
            print(f"   - {ruca}: {first_pct:.1%} → {last_pct:.1%} (Δ {growth:+.1%})")
    
    # Por Idade
    print(f"\n3. Por Faixa Etária:")
    for age in df_analysis[df_analysis['Bene_Age_Desc'] != 'All']['Bene_Age_Desc'].unique():
        age_data = df_analysis[(df_analysis['Bene_Age_Desc'] == age) & 
                              (df_analysis['Bene_Geo_Desc'] == 'National')]
        first_pct = age_data[age_data['Year'] == first_year]['Pct_Telehealth'].mean()
        last_pct = age_data[age_data['Year'] == last_year]['Pct_Telehealth'].mean()
        if not pd.isna(first_pct) and not pd.isna(last_pct):
            growth = last_pct - first_pct
            print(f"   - {age}: {first_pct:.1%} → {last_pct:.1%} (Δ {growth:+.1%})")

# Gráfico de evolução temporal
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Evolução geral
yearly_avg = df_analysis.groupby('Year')['Pct_Telehealth'].mean()
axes[0, 0].plot(yearly_avg.index, yearly_avg.values, marker='o', linewidth=2, markersize=8)
axes[0, 0].set_xlabel('Ano')
axes[0, 0].set_ylabel('Pct_Telehealth Médio')
axes[0, 0].set_title('Evolução Temporal da Adoção de Telemedicina')
axes[0, 0].grid(True, alpha=0.3)

# Evolução por RUCA
for ruca in ['Urban', 'Rural']:
    ruca_data = df_analysis[(df_analysis['Bene_RUCA_Desc'] == ruca) & 
                           (df_analysis['Bene_Geo_Desc'] == 'National')]
    if not ruca_data.empty:
        yearly_ruca = ruca_data.groupby('Year')['Pct_Telehealth'].mean()
        axes[0, 1].plot(yearly_ruca.index, yearly_ruca.values, marker='o', 
                       linewidth=2, markersize=6, label=ruca)
axes[0, 1].set_xlabel('Ano')
axes[0, 1].set_ylabel('Pct_Telehealth Médio')
axes[0, 1].set_title('Evolução por Região (RUCA)')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Evolução por Raça
for race in df_analysis[df_analysis['Bene_Race_Desc'] != 'All']['Bene_Race_Desc'].unique()[:5]:
    race_data = df_analysis[(df_analysis['Bene_Race_Desc'] == race) & 
                           (df_analysis['Bene_Geo_Desc'] == 'National')]
    if not race_data.empty:
        yearly_race = race_data.groupby('Year')['Pct_Telehealth'].mean()
        axes[1, 0].plot(yearly_race.index, yearly_race.values, marker='o', 
                       linewidth=2, markersize=6, label=race)
axes[1, 0].set_xlabel('Ano')
axes[1, 0].set_ylabel('Pct_Telehealth Médio')
axes[1, 0].set_title('Evolução por Raça')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Evolução por Idade
for age in df_analysis[df_analysis['Bene_Age_Desc'] != 'All']['Bene_Age_Desc'].unique():
    age_data = df_analysis[(df_analysis['Bene_Age_Desc'] == age) & 
                          (df_analysis['Bene_Geo_Desc'] == 'National')]
    if not age_data.empty:
        yearly_age = age_data.groupby('Year')['Pct_Telehealth'].mean()
        axes[1, 1].plot(yearly_age.index, yearly_age.values, marker='o', 
                      linewidth=2, markersize=6, label=age)
axes[1, 1].set_xlabel('Ano')
axes[1, 1].set_ylabel('Pct_Telehealth Médio')
axes[1, 1].set_title('Evolução por Faixa Etária')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('evolucao_temporal.png', dpi=300, bbox_inches='tight')
print("\n[✓] Gráfico de evolução temporal salvo: evolucao_temporal.png")
plt.close()

# ============================================================================
# 7. RESUMO FINAL
# ============================================================================

print("\n" + "="*80)
print("[7] RESUMO FINAL")
print("="*80)

print(f"\n✓ Análise exploratória completa realizada")
print(f"✓ Modelos de classificação treinados e avaliados")
print(f"✓ Modelos de regressão treinados e avaliados")
print(f"✓ 3 perguntas de pesquisa respondidas")
print(f"\n✓ Gráficos gerados:")
print(f"  - distribuicao_variaveis.png")
print(f"  - heatmap_correlacao.png")
print(f"  - curvas_roc.png")
print(f"  - predicao_regressao.png")
print(f"  - evolucao_temporal.png")

print("\n" + "="*80)
print("ANÁLISE CONCLUÍDA COM SUCESSO!")
print("="*80)

