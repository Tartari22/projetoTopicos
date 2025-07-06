import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
from sklearn.metrics import confusion_matrix, silhouette_score
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("="*80)
print("PROJETO DE MINERAÇÃO DE DADOS - WINE QUALITY DATASET")
print("="*80)
print()

print("1. CARREGAMENTO E EXPLORAÇÃO DOS DADOS")
print("-" * 50)

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"

try:
    df = pd.read_csv(url, sep=';')
    print(f"✓ Dataset carregado com sucesso!")
    print(f"  Dimensões: {df.shape}")
    print(f"  Colunas: {list(df.columns)}")
    print()
except Exception as e:
    print(f"Erro ao carregar o dataset: {e}")
    np.random.seed(42)
    n_samples = 1000
    df = pd.DataFrame({
        'fixed acidity': np.random.normal(8.5, 1.5, n_samples),
        'volatile acidity': np.random.normal(0.5, 0.2, n_samples),
        'citric acid': np.random.normal(0.3, 0.2, n_samples),
        'residual sugar': np.random.normal(2.5, 1.0, n_samples),
        'chlorides': np.random.normal(0.08, 0.02, n_samples),
        'free sulfur dioxide': np.random.normal(15, 5, n_samples),
        'total sulfur dioxide': np.random.normal(45, 15, n_samples),
        'density': np.random.normal(0.997, 0.002, n_samples),
        'pH': np.random.normal(3.3, 0.15, n_samples),
        'sulphates': np.random.normal(0.65, 0.15, n_samples),
        'alcohol': np.random.normal(10.5, 1.0, n_samples),
        'quality': np.random.choice([3, 4, 5, 6, 7, 8], n_samples, p=[0.05, 0.15, 0.35, 0.30, 0.13, 0.02])
    })
    print("✓ Dataset simulado criado para demonstração")
    print(f"  Dimensões: {df.shape}")
    print()

print("INFORMAÇÕES BÁSICAS DO DATASET:")
print(df.info())
print()

print("ESTATÍSTICAS DESCRITIVAS:")
print(df.describe())
print()

print("DISTRIBUIÇÃO DA VARIÁVEL ALVO (QUALITY):")
print(df['quality'].value_counts().sort_index())
print()

print("2. VISUALIZAÇÃO DOS DADOS")
print("-" * 50)

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Análise Exploratória dos Dados - Wine Quality Dataset', fontsize=16, fontweight='bold')

axes[0, 0].hist(df['quality'], bins=range(3, 10), alpha=0.7, color='skyblue', edgecolor='black')
axes[0, 0].set_title('Distribuição da Qualidade dos Vinhos')
axes[0, 0].set_xlabel('Qualidade')
axes[0, 0].set_ylabel('Frequência')
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].scatter(df['alcohol'], df['quality'], alpha=0.6, color='coral')
axes[0, 1].set_title('Relação entre Álcool e Qualidade')
axes[0, 1].set_xlabel('Teor Alcoólico (%)')
axes[0, 1].set_ylabel('Qualidade')
axes[0, 1].grid(True, alpha=0.3)

corr_matrix = df.corr()
im = axes[1, 0].imshow(corr_matrix, cmap='coolwarm', aspect='auto')
axes[1, 0].set_title('Matriz de Correlação')
axes[1, 0].set_xticks(range(len(corr_matrix.columns)))
axes[1, 0].set_yticks(range(len(corr_matrix.columns)))
axes[1, 0].set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
axes[1, 0].set_yticklabels(corr_matrix.columns)
plt.colorbar(im, ax=axes[1, 0])

main_features = ['alcohol', 'volatile acidity', 'citric acid', 'sulphates']
df_subset = df[main_features]
axes[1, 1].boxplot([df_subset[col] for col in main_features])
axes[1, 1].set_title('Boxplot das Principais Características')
axes[1, 1].set_xticklabels(main_features, rotation=45, ha='right')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n3. TÉCNICA 1: CLASSIFICAÇÃO")
print("-" * 50)

df['quality_category'] = pd.cut(df['quality'], bins=[2, 5, 7, 9], labels=['Baixa', 'Média', 'Alta'])

X = df.drop(['quality', 'quality_category'], axis=1)
y = df['quality_category']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train_scaled, y_train)

y_pred = rf_classifier.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
print(f"Acurácia do modelo de classificação: {accuracy:.4f}")
print()

print("RELATÓRIO DE CLASSIFICAÇÃO:")
print(classification_report(y_test, y_pred))
print()

cm = confusion_matrix(y_test, y_pred)
print("MATRIZ DE CONFUSÃO:")
print(cm)
print()

feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_classifier.feature_importances_
}).sort_values('importance', ascending=False)

print("IMPORTÂNCIA DAS CARACTERÍSTICAS:")
print(feature_importance.head(10))
print()

print("4. TÉCNICA 2: REGRESSÃO")
print("-" * 50)

X_reg = df.drop(['quality', 'quality_category'], axis=1)
y_reg = df['quality']

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.3, random_state=42)

X_train_reg_scaled = scaler.fit_transform(X_train_reg)
X_test_reg_scaled = scaler.transform(X_test_reg)

rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor.fit(X_train_reg_scaled, y_train_reg)

y_pred_reg = rf_regressor.predict(X_test_reg_scaled)

mse = mean_squared_error(y_test_reg, y_pred_reg)
r2 = r2_score(y_test_reg, y_pred_reg)

print(f"Erro Quadrático Médio (MSE): {mse:.4f}")
print(f"R² Score: {r2:.4f}")
print()

print("5. TÉCNICA 3: AGRUPAMENTO (CLUSTERING)")
print("-" * 50)

X_cluster = df.drop(['quality', 'quality_category'], axis=1)
X_cluster_scaled = StandardScaler().fit_transform(X_cluster)

inertias = []
silhouette_scores = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_cluster_scaled)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_cluster_scaled, kmeans.labels_))

kmeans = KMeans(n_clusters=3, random_state=42)
cluster_labels = kmeans.fit_predict(X_cluster_scaled)

df['cluster'] = cluster_labels

print(f"Número de clusters criados: 3")
print(f"Silhouette Score: {silhouette_score(X_cluster_scaled, cluster_labels):.4f}")
print()

print("DISTRIBUIÇÃO DOS CLUSTERS:")
print(df['cluster'].value_counts().sort_index())
print()

print("QUALIDADE MÉDIA POR CLUSTER:")
cluster_quality = df.groupby('cluster')['quality'].agg(['mean', 'std', 'count'])
print(cluster_quality)
print()

print("6. VISUALIZAÇÃO DOS RESULTADOS")
print("-" * 50)

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Resultados da Mineração de Dados - Wine Quality', fontsize=16, fontweight='bold')

im1 = axes[0, 0].imshow(cm, cmap='Blues')
axes[0, 0].set_title('Matriz de Confusão - Classificação')
axes[0, 0].set_xlabel('Predito')
axes[0, 0].set_ylabel('Real')
for i in range(len(cm)):
    for j in range(len(cm)):
        axes[0, 0].text(j, i, cm[i, j], ha='center', va='center')

top_features = feature_importance.head(8)
axes[0, 1].barh(top_features['feature'], top_features['importance'])
axes[0, 1].set_title('Top 8 Características Mais Importantes')
axes[0, 1].set_xlabel('Importância')

axes[1, 0].scatter(y_test_reg, y_pred_reg, alpha=0.6, color='green')
axes[1, 0].plot([y_test_reg.min(), y_test_reg.max()], [y_test_reg.min(), y_test_reg.max()], 'r--', lw=2)
axes[1, 0].set_title(f'Regressão: Predito vs Real (R²={r2:.3f})')
axes[1, 0].set_xlabel('Valores Reais')
axes[1, 0].set_ylabel('Valores Preditos')
axes[1, 0].grid(True, alpha=0.3)

scatter = axes[1, 1].scatter(df['alcohol'], df['volatile acidity'], c=df['cluster'], cmap='viridis', alpha=0.6)
axes[1, 1].set_title('Agrupamento: Álcool vs Acidez Volátil')
axes[1, 1].set_xlabel('Teor Alcoólico (%)')
axes[1, 1].set_ylabel('Acidez Volátil')
axes[1, 1].grid(True, alpha=0.3)
plt.colorbar(scatter, ax=axes[1, 1])

plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Análise de Clusters - Métodos de Avaliação', fontsize=14, fontweight='bold')

axes[0].plot(K_range, inertias, 'bo-')
axes[0].set_title('Método do Cotovelo')
axes[0].set_xlabel('Número de Clusters')
axes[0].set_ylabel('Inércia')
axes[0].grid(True, alpha=0.3)

axes[1].plot(K_range, silhouette_scores, 'ro-')
axes[1].set_title('Silhouette Score')
axes[1].set_xlabel('Número de Clusters')
axes[1].set_ylabel('Silhouette Score')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("8. RESUMO DOS RESULTADOS")
print("-" * 50)

print("RESUMO FINAL:")
print(f"• Dataset: Wine Quality Dataset")
print(f"• Número de registros: {len(df)}")
print(f"• Número de características: {len(X.columns)}")
print()

print("TÉCNICA 1 - CLASSIFICAÇÃO:")
print(f"• Algoritmo: Random Forest Classifier")
print(f"• Acurácia: {accuracy:.4f}")
print(f"• Classes: {list(df['quality_category'].cat.categories)}")
print()

print("TÉCNICA 2 - REGRESSÃO:")
print(f"• Algoritmo: Random Forest Regressor")
print(f"• MSE: {mse:.4f}")
print(f"• R² Score: {r2:.4f}")
print()

print("TÉCNICA 3 - AGRUPAMENTO:")
print(f"• Algoritmo: K-Means")
print(f"• Número de clusters: 3")
print(f"• Silhouette Score: {silhouette_score(X_cluster_scaled, cluster_labels):.4f}")
print()

print("="*80)
print("PROJETO CONCLUÍDO COM SUCESSO!")
