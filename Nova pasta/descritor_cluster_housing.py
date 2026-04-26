# SISTEMAS INTELIGENTES
# Descritor de clusters — HousingData

import pickle
import pandas as pd
import numpy as np

# ─────────────────────────────────────────
# Carregar modelo e normalizador
# ─────────────────────────────────────────
cluster_model = pickle.load(open('cluster_housing.pkl', 'rb'))
normalizador  = pickle.load(open('normalizador_housing.pkl', 'rb'))

colunas = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX',
           'RM', 'AGE', 'DIS', 'RAD', 'TAX',
           'PTRATIO', 'B', 'LSTAT', 'MEDV']

# ─────────────────────────────────────────
# Desnormalizar os centroides
# ─────────────────────────────────────────
centroides_norm = pd.DataFrame(
    cluster_model.cluster_centers_,
    columns=colunas
)

centroides = pd.DataFrame(
    normalizador.inverse_transform(centroides_norm),
    columns=colunas
)

# ─────────────────────────────────────────
# Descrição humanizada de cada cluster
# ─────────────────────────────────────────

# Referências para classificação qualitativa
MEDV_baixo  = 17   # abaixo de 17 mil = barato
MEDV_alto   = 30   # acima de 30 mil  = caro
CRIM_baixo  = 1    # criminalidade baixa
CRIM_alto   = 10   # criminalidade alta
RM_pequeno  = 5.5  # poucos cômodos
RM_grande   = 6.5  # muitos cômodos

def classificar(valor, baixo, alto, labels=('baixo', 'médio', 'alto')):
    if valor <= baixo:
        return labels[0]
    elif valor <= alto:
        return labels[1]
    else:
        return labels[2]

print("=" * 60)
print("       DESCRIÇÃO DOS SEGMENTOS (CLUSTERS)")
print("=" * 60)

for i, row in centroides.iterrows():
    preco       = classificar(row['MEDV'],   MEDV_baixo, MEDV_alto)
    crime       = classificar(row['CRIM'],   CRIM_baixo, CRIM_alto)
    comodos     = classificar(row['RM'],     RM_pequeno, RM_grande)
    poluicao    = classificar(row['NOX'],    0.45, 0.60)
    industria   = classificar(row['INDUS'],  5, 15)
    alunos_prof = classificar(row['PTRATIO'],16, 19)
    status_baixo= classificar(row['LSTAT'],  8, 18)
    idade_imovel= classificar(row['AGE'],    40, 75)

    print(f"\n📦 CLUSTER {i}")
    print(f"   Valor médio do imóvel : R$ {row['MEDV']:.1f} mil  ({preco})")
    print(f"   Criminalidade         : {row['CRIM']:.2f}          ({crime})")
    print(f"   Nº médio de cômodos   : {row['RM']:.1f}            ({comodos})")
    print(f"   Poluição (NOX)        : {row['NOX']:.3f}          ({poluicao})")
    print(f"   % área industrial     : {row['INDUS']:.1f}%         ({industria})")
    print(f"   Alunos por professor  : {row['PTRATIO']:.1f}          ({alunos_prof})")
    print(f"   % pop. baixa renda    : {row['LSTAT']:.1f}%          ({status_baixo})")
    print(f"   Idade média dos imóv. : {row['AGE']:.1f}%           ({idade_imovel})")

    # Resumo em linguagem natural
    print(f"\n   💬 Resumo: Imóveis de valor {preco}, com criminalidade {crime},")
    print(f"      {comodos}s cômodos, poluição {poluicao} e área industrial {industria}.")
    print(f"      Proporção de população de baixa renda {status_baixo}.")

print("\n" + "=" * 60)
