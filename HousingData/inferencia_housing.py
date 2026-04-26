# SISTEMAS INTELIGENTES
# Inferência de cluster — HousingData
# Recebe dados de um imóvel desconhecido e diz a qual cluster pertence

import pickle
import pandas as pd
import numpy as np

# ─────────────────────────────────────────
# Carregar modelo, normalizador e imputer
# ─────────────────────────────────────────
cluster_model = pickle.load(open('cluster_housing.pkl', 'rb'))
normalizador  = pickle.load(open('normalizador_housing.pkl', 'rb'))
imputer       = pickle.load(open('imputer_housing.pkl', 'rb'))

colunas = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX',
           'RM', 'AGE', 'DIS', 'RAD', 'TAX',
           'PTRATIO', 'B', 'LSTAT', 'MEDV']

# ─────────────────────────────────────────
# Dados do imóvel desconhecido
# Altere os valores abaixo para testar diferentes imóveis
# Use None para campos que você não sabe (serão preenchidos pela média)
# ─────────────────────────────────────────
imovel = {
    'CRIM'   : 0.1,    # Taxa de criminalidade per capita
    'ZN'     : 12.5,   # % de terreno residencial zoneado
    'INDUS'  : 7.0,    # % de acres de negócios não varejistas
    'CHAS'   : 0,      # Adjacente ao rio Charles? (1=sim, 0=não)
    'NOX'    : 0.52,   # Concentração de óxido nítrico
    'RM'     : 6.2,    # Nº médio de cômodos por habitação
    'AGE'    : 70.0,   # % de unidades construídas antes de 1940
    'DIS'    : 4.5,    # Distância aos centros de emprego
    'RAD'    : 4,      # Índice de acessibilidade a rodovias
    'TAX'    : 300,    # Taxa de imposto predial
    'PTRATIO': 17.0,   # Razão alunos/professor
    'B'      : 390.0,  # Índice relacionado à proporção de negros
    'LSTAT'  : 10.0,   # % de população de baixo status
    'MEDV'   : None,   # Valor médio do imóvel (None = desconhecido)
}

# ─────────────────────────────────────────
# Pré-processamento do imóvel
# ─────────────────────────────────────────
# Converter para DataFrame (linha única)
df_imovel = pd.DataFrame([imovel], columns=colunas)

# Preencher campos None com a média aprendida no treinamento
df_imovel_preenchido = pd.DataFrame(
    imputer.transform(df_imovel),
    columns=colunas
)

# Normalizar com o scaler do treinamento
df_imovel_norm = pd.DataFrame(
    normalizador.transform(df_imovel_preenchido),
    columns=colunas
)

# ─────────────────────────────────────────
# Inferência
# ─────────────────────────────────────────
cluster_previsto = cluster_model.predict(df_imovel_norm)[0]

# Calcular distância para cada centroide (grau de "certeza")
centroides = cluster_model.cluster_centers_
distancias = np.linalg.norm(centroides - df_imovel_norm.values, axis=1)

print("=" * 50)
print("     RESULTADO DA INFERÊNCIA")
print("=" * 50)
print(f"\n📍 O imóvel pertence ao  CLUSTER {cluster_previsto}\n")

print("📏 Distância para cada cluster (menor = mais próximo):")
for i, d in enumerate(distancias):
    marcador = " ◄ PERTENCE" if i == cluster_previsto else ""
    print(f"   Cluster {i}: {d:.4f}{marcador}")

print("\n📋 Dados informados do imóvel:")
for col in colunas:
    val = imovel[col]
    if val is None:
        print(f"   {col:10s}: (preenchido pela média)")
    else:
        print(f"   {col:10s}: {val}")

print("\n" + "=" * 50)
