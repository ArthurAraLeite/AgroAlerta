from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
import joblib
import os

COLUNAS_ENTRADA = ["ph", "ec", "temperatura", "umidade"]
COLUNA_SAIDA = "alerta_irrigacao"
CAMINHO_MODELO = "modelo_treinado.pkl"

def treinar(df):
    X = df[COLUNAS_ENTRADA]
    y = df[COLUNA_SAIDA]

    # Limita a complexidade do modelo (menos árvores, profundidade menor,
    # mínimo de amostras por folha) -- isso reduz a capacidade dele de
    # "memorizar" os dados de treino em vez de generalizar.
    modelo = RandomForestClassifier(
        n_estimators=50,
        max_depth=4,
        min_samples_leaf=5,
        random_state=42
    )

    # Validação cruzada (5 divisões diferentes dos dados) em vez de um único
    # split -- dá uma média mais confiável da performance real, principalmente
    # com poucos dados, onde um único split pode "sortear" um resultado bom
    # ou ruim por acaso.
    scores = cross_val_score(modelo, X, y, cv=5)
    print(f"Acurácia por divisão (cross-validation): {scores}")
    print(f"Acurácia média: {scores.mean():.2f} (+/- {scores.std():.2f})")

    # Treina a versão final com todos os dados disponíveis
    modelo.fit(X, y)
    joblib.dump(modelo, CAMINHO_MODELO)

    return modelo, scores.mean()

def carregar_modelo():
    if os.path.exists(CAMINHO_MODELO):
        return joblib.load(CAMINHO_MODELO)
    return None

def prever(modelo, df_novo):
    X_novo = df_novo[COLUNAS_ENTRADA]
    return modelo.predict(X_novo)
