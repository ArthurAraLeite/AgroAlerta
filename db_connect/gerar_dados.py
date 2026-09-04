import numpy as np
import pandas as pd
from db import engine

np.random.seed(42)
N = 300  # bem mais que os 50 atuais

registros = []

for _ in range(N):
    # Sorteia se essa amostra vai "tender" a precisar de irrigação ou não,
    # mas com sobreposição entre as classes (não é uma regra perfeita)
    precisa_irrigar = np.random.rand() < 0.4

    if precisa_irrigar:
        umidade = np.random.normal(loc=15, scale=8)   # tende a ser baixa, mas com variação
        temperatura = np.random.normal(loc=30, scale=4)
    else:
        umidade = np.random.normal(loc=35, scale=10)  # tende a ser alta, mas com variação
        temperatura = np.random.normal(loc=24, scale=4)

    ph = np.clip(np.random.normal(loc=6.3, scale=0.7), 3.5, 9.0)
    ec = np.clip(np.random.normal(loc=1.3, scale=0.6), 0.1, 5.0)
    umidade = np.clip(umidade, 2, 60)
    temperatura = np.clip(temperatura, 10, 40)

    # Ruído deliberado: ~10% das amostras recebem o rótulo "errado" de propósito,
    # simulando erro de medição real e casos ambíguos (isso é o que falta em
    # dados sintéticos "perfeitos" e é o que geralmente causa overfitting)
    rotulo = int(precisa_irrigar)
    if np.random.rand() < 0.10:
        rotulo = 1 - rotulo

    registros.append({
        "ph": round(ph, 2),
        "ec": round(ec, 2),
        "temperatura": round(temperatura, 1),
        "umidade": round(umidade, 1),
        "alerta_irrigacao": rotulo
    })

df_novos = pd.DataFrame(registros)

df_novos.to_sql("leituras_solo", engine, if_exists="append", index=False)

print(f"{N} novos registros inseridos.")
print(df_novos.head())
print("\nDistribuição das classes:")
print(df_novos["alerta_irrigacao"].value_counts())
