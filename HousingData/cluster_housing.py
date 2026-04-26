# SISTEMAS INTELIGENTES
# Modelos não supervisionados
# Base HousingData (Boston Housing)

# Imports
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
import pickle
from sklearn.cluster import KMeans
import math
from scipy.spatial.distance import cdist
import numpy as np

# ─────────────────────────────────────────
# 1. Abrir os dados
# ─────────────────────────────────────────
dados = pd.read_csv('HousingData.csv')

# ─────────────────────────────────────────
# 2. Preencher valores vazios (NaN)
#    Estratégia: média da coluna — igual ao que faríamos manualmente
# ─────────────────────────────────────────
imputer = SimpleImputer(strategy='mean')
dados_preenchidos = pd.DataFrame(
    imputer.fit_transform(dados),
    columns=dados.columns
)

# Salvar o imputer para uso posterior (inferência)
pickle.dump(imputer, open('imputer_housing.pkl', 'wb'))

print(f"Valores nulos antes: {dados.isnull().sum().sum()}")
print(f"Valores nulos depois: {dados_preenchidos.isnull().sum().sum()}")

# ─────────────────────────────────────────
# 3. Normalizar os dados
#    Todas as colunas são numéricas — sem get_dummies necessário
# ─────────────────────────────────────────
scaler = MinMaxScaler()
normalizador = scaler.fit(dados_preenchidos)

# Salvar o normalizador para uso posterior
pickle.dump(normalizador, open('normalizador_housing.pkl', 'wb'))

dados_norm = pd.DataFrame(
    normalizador.transform(dados_preenchidos),
    columns=dados.columns
)

# ─────────────────────────────────────────
# 4. Hiperparametrizar — determinar número ótimo de clusters
#    Método do cotovelo (elbow) com distância geométrica
# ─────────────────────────────────────────
# Limitar K ao máximo razoável (raiz quadrada do número de amostras)
K_MAX = min(int(math.sqrt(dados_norm.shape[0])), 20)
K = range(1, K_MAX + 1)

distortions = []
for i in K:
    modelo = KMeans(n_clusters=i, random_state=42, n_init=10).fit(dados_norm)
    distortions.append(
        sum(
            np.min(
                cdist(dados_norm, modelo.cluster_centers_, 'euclidean'),
                axis=1
            ) / dados_norm.shape[0]
        )
    )

# Distância geométrica para encontrar o cotovelo
x0, y0 = K[0],  distortions[0]
xn, yn = K[-1], distortions[-1]
distances = []
for i in range(len(distortions)):
    x, y = K[i], distortions[i]
    numerador   = abs((yn - y0) * x - (xn - x0) * y + xn * y0 - yn * x0)
    denominador = math.sqrt((yn - y0) ** 2 + (xn - x0) ** 2)
    distances.append(numerador / denominador)

numero_clusters_otimo = K[distances.index(np.max(distances))]
print(f"\nNúmero ótimo de clusters: {numero_clusters_otimo}")

# ─────────────────────────────────────────
# 5. Treinar e salvar o modelo
# ─────────────────────────────────────────
cluster_model = KMeans(
    n_clusters=numero_clusters_otimo,
    random_state=42,
    n_init=10
).fit(dados_norm)

pickle.dump(cluster_model, open('cluster_housing.pkl', 'wb'))

print(f"\nModelo salvo em cluster_housing.pkl")
print(f"Distribuição dos clusters:")
labels = cluster_model.labels_
for c in range(numero_clusters_otimo):
    print(f"  Cluster {c}: {(labels == c).sum()} imóveis")
