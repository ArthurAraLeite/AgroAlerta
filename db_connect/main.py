# main.py
from db import buscar_dados, salvar_previsao
from modelo import treinar, carregar_modelo, prever

MINIMO_PARA_RETREINAR = 30  # ajuste conforme volume real de dados

def main():
    df_historico = buscar_dados("leituras_solo")

    modelo = carregar_modelo()

    # Retreina se não existe modelo ainda, ou se já tem dado suficiente novo
    if modelo is None or len(df_historico) >= MINIMO_PARA_RETREINAR:
        modelo, acuracia = treinar(df_historico)
        print(f"Modelo (re)treinado. Acurácia: {acuracia:.2f}")

    # Pega a leitura mais recente que ainda não tem previsão
    df_novo = buscar_dados("leituras_solo", "ORDER BY criado_em DESC LIMIT 1")

    predicao = prever(modelo, df_novo)[0]

    salvar_previsao("predicoes", {
        "leitura_id": int(df_novo.iloc[0]["id"]),
        "alerta_irrigacao_previsto": int(predicao),
        "modelo_versao": "rf_v1"
    })

    print("Previsão salva:", predicao)

if __name__ == "__main__":
    main() 