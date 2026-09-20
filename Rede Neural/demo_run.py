"""
demo_run.py
===========
Script de demonstração e homologação do sistema AgroMultitask.
Treina a rede neural (se necessário) e executa simulações reais e contrastantes:
- Cenário 1: Talhão de Café (Convencional) sob estiagem.
- Cenário 2: Talhão de Tomate (Orgânico) com alerta de chuva (Trava Estrita acionada).
- Cenário 3: Talhão de Soja com alerta fitossanitário de calor/ácaros.
"""

import os
import sys
import json

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from train_pipeline import train_multitask_model
from agro_system import predict_agro_system, formatar_relatorio_terminal


def run_demonstration():
    model_path = "best_agro_multitask.pth"
    scaler_path = "agro_scaler_meta.pkl"

    # 1. Verifica se precisa treinar o modelo
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print("Modelo ou escalador não encontrados. Iniciando treinamento do modelo...")
        train_multitask_model(
            csv_path="dados_solo_2casas.csv",
            n_samples=50000,
            epochs=12,
            batch_size=64,
            learning_rate=1e-3,
            model_save_path=model_path,
            scaler_save_path=scaler_path
        )
    else:
        print(f"Modelo pré-treinado encontrado em '{model_path}'. Pronto para inferência!")

    # =========================================================================
    # CENÁRIO 1: Café Convencional em Solo Arenoso e Seco (Sem chuva prevista)
    # =========================================================================
    print("\n\n" + "#" * 70)
    print(" EXECUTANDO CENÁRIO 1: CAFÉ CONVENCIONAL - DÉFICIT HÍDRICO SEVERO")
    print("#" * 70)

    arduino_cafe = {
        "tipo_solo": "arenoso",
        "pH": 6.10,
        "teor_umildade_%": 1.40,  # muito abaixo do ponto crítico (2.2%) e CC (4.8%)
        "nitrogenio_N_ppm": 2.10,
        "fosforo_P_ppm": 1.90,
        "potassio_K_ppm": 1.80,
        "condutividade_eletrica_dS/m": 0.85,
        "compactacao_solo_kPa": 1450.0
    }

    # Coordenadas em região cafeeira (ex: Sul de Minas / Mogiana)
    relatorio_1 = predict_agro_system(
        lat=-21.55,
        lon=-46.88,
        cultura="cafe",
        is_organico=False,
        dados_arduino_dict=arduino_cafe
    )
    print(formatar_relatorio_terminal(relatorio_1))

    # =========================================================================
    # CENÁRIO 2: Tomate Orgânico com Chuva Iminente (TESTE DA TRAVA ESTRITA)
    # =========================================================================
    print("\n\n" + "#" * 70)
    print(" EXECUTANDO CENÁRIO 2: TOMATE ORGÂNICO - TESTE DA TRAVA ESTRITA DE CHUVA")
    print("#" * 70)

    arduino_tomate = {
        "tipo_solo": "argiloso",
        "pH": 5.20,  # ácido
        "teor_umildade_%": 3.80,  # solo pedindo água
        "nitrogenio_N_ppm": 4.50,
        "fosforo_P_ppm": 5.20,
        "potassio_K_ppm": 3.80,
        "condutividade_eletrica_dS/m": 1.10,
        "compactacao_solo_kPa": 1900.0
    }

    # Mock temporário no Open-Meteo do cenário para forçar a probabilidade de chuva alta
    from weather_client import weather_service
    original_get_features = weather_service.get_weather_features

    def mock_weather_storm(lat, lon):
        features = original_get_features(lat, lon)
        features["precipitation_probability_max_6h"] = 85.0
        features["rain_sum_6h"] = 14.5
        features["rain_sum_12h"] = 26.0
        features["relative_humidity_2m"] = 88.0
        features["temperature_2m"] = 24.0
        return features

    weather_service.get_weather_features = mock_weather_storm

    relatorio_2 = predict_agro_system(
        lat=-22.40,
        lon=-47.55,
        cultura="tomate",
        is_organico=True,
        dados_arduino_dict=arduino_tomate
    )
    print(formatar_relatorio_terminal(relatorio_2))

    # Restaura o weather service original
    weather_service.get_weather_features = original_get_features

    # =========================================================================
    # CENÁRIO 3: Soja Convencional com Salinidade e Solo Compactado
    # =========================================================================
    print("\n\n" + "#" * 70)
    print(" EXECUTANDO CENÁRIO 3: SOJA CONVENCIONAL - ESTRESSE SALINO E MECÂNICO")
    print("#" * 70)

    arduino_soja = {
        "tipo_solo": "ideal",
        "pH": 7.40,
        "teor_umildade_%": 6.80,
        "nitrogenio_N_ppm": 6.80,
        "fosforo_P_ppm": 5.90,
        "potassio_K_ppm": 4.80,
        "condutividade_eletrica_dS/m": 3.40,  # CE elevada (>3.0)
        "compactacao_solo_kPa": 2650.0       # Compactação severa (>2000 kPa)
    }

    relatorio_3 = predict_agro_system(
        lat=-17.79,
        lon=-50.92,
        cultura="soja",
        is_organico=False,
        dados_arduino_dict=arduino_soja
    )
    print(formatar_relatorio_terminal(relatorio_3))

    print("\n" + "=" * 70)
    print(" Demonstração concluída com sucesso!")
    print("=" * 70)


if __name__ == "__main__":
    run_demonstration()
