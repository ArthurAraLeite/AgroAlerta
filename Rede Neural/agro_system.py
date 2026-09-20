"""
agro_system.py
==============
Sistema de Tomada de Decisões Agronômicas Inteligentes baseado em Deep Learning.
Contém a classe 'AgroDecisionSystem' e a função principal 'predict_agro_system'
que consome sensores de solo/Arduino e API Open-Meteo em tempo real,
executa a inferência pela Rede Neural Multitarefa, aplica as travas de segurança
e formula diagnósticos e planos de ação prescritivos (Orgânico vs Convencional).
"""

import os
import json
import pickle
from typing import Dict, Any, Optional
import torch

from agronomy_rules import (
    CULTURAS_CONFIG,
    SOLO_CONFIG,
    MANEJO_CATALOGO,
    diagnosticar_solo
)
from weather_client import get_weather_features
from data_engine import AgroDataEngine
from model_multitask import AgroMultitaskNet


class AgroDecisionSystem:
    """
    Sistema Gerenciador de Decisão Agronômica Multitarefa.
    """

    def __init__(
        self,
        model_path: str = "best_agro_multitask.pth",
        scaler_path: str = "agro_scaler_meta.pkl"
    ):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.data_engine = None
        self._load_resources()

    def _load_resources(self):
        """Carrega os pesos da rede neural e os metadados do scaler."""
        if not os.path.exists(self.scaler_path):
            raise FileNotFoundError(
                f"Arquivo de metadados {self.scaler_path} não encontrado. "
                "Execute o treinamento primeiro via 'train_pipeline.py'!"
            )
        self.data_engine = AgroDataEngine(scaler_path=self.scaler_path)

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Checkpoint {self.model_path} não encontrado. "
                "Execute o treinamento primeiro via 'train_pipeline.py'!"
            )

        checkpoint = torch.load(self.model_path, map_location=self.device)
        in_features = checkpoint["in_features"]
        self.model = AgroMultitaskNet(in_features=in_features, num_irrig_classes=3, num_pest_labels=4)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

    def predict(
        self,
        lat: float,
        lon: float,
        cultura: str,
        is_organico: bool,
        dados_arduino: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Executa a predição completa do sistema para um talhão agrícola.
        """
        cultura_norm = cultura.lower().strip()
        if cultura_norm not in CULTURAS_CONFIG:
            raise ValueError(f"Cultura '{cultura}' não suportada. Opções: {list(CULTURAS_CONFIG.keys())}")

        tipo_solo = dados_arduino.get("tipo_solo", "ideal").lower().strip()
        if tipo_solo not in SOLO_CONFIG:
            tipo_solo = "ideal"

        # 1. Coleta de dados climáticos via Open-Meteo
        dados_clima = get_weather_features(lat, lon)

        # 2. Transformação do vetor de entrada
        X_vec = self.data_engine.transform_single_sample(
            tipo_solo=tipo_solo,
            cultura=cultura_norm,
            is_organico=is_organico,
            dados_arduino=dados_arduino,
            dados_clima=dados_clima
        )

        tensor_x = torch.tensor(X_vec, dtype=torch.float32).to(self.device)

        # 3. Inferência da Rede Neural Multitarefa
        with torch.no_grad():
            preds = self.model.predict_probabilities(tensor_x)

        probs_irrig = preds["probs_irrigacao"].cpu().numpy()[0]
        probs_pragas = preds["probs_pragas"].cpu().numpy()[0]
        pred_vol_mm = float(preds["volume_mm"].cpu().numpy()[0])

        classe_pred_irrig = int(np_argmax(probs_irrig))

        # 4. Verificação da Trava Estrita de Chuva (Segurança Física)
        precip_prob_6h = dados_clima["precipitation_probability_max_6h"]
        rain_sum_6h = dados_clima["rain_sum_6h"]
        trava_chuva_acionada = (precip_prob_6h > 70.0) or (rain_sum_6h >= 5.0)

        if trava_chuva_acionada:
            classe_irrig_final = 0
            volume_mm_final = 0.0
            status_irrig = "SUSPENDER / NÃO REGAR (Trava Estrita de Chuva Acionada)"
            motivo_irrig = (
                f"Alerta de Chuva Iminente: Probabilidade nas próximas 6h de {precip_prob_6h:.1f}% "
                f"e volume previsto de {rain_sum_6h:.1f} mm. Irrigação desnecessária ou de risco."
            )
        else:
            classe_irrig_final = classe_pred_irrig
            volume_mm_final = round(pred_vol_mm, 2)
            nomes_classes = {
                0: "Não Irrigar (Solo com Umidade Suficiente)",
                1: "Irrigar Leve/Moderada (Reposição de Déficit)",
                2: "Irrigar Abundante / Turno Completo (Déficit Hídrico Severo)"
            }
            status_irrig = nomes_classes.get(classe_irrig_final, "Não Definido")
            motivo_irrig = "Predição orientada pelo balanço hídrico do solo e demanda evaporativa da cultura."

        # 5. Interpretação de Riscos Fitossanitários
        pragas_names = [
            ("Risco de Fungos e Doenças Foliares (Míldio/Requeima/Oídio)", probs_pragas[0], "fungos_bacterias"),
            ("Risco de Insetos e Lagartas Mastigadoras", probs_pragas[1], "insetos_lagartas"),
            ("Risco de Ácaros e Tripes (Clima Quente e Seco)", probs_pragas[2], "acaros_tripes"),
            ("Risco de Estresse Hídrico / Salino", probs_pragas[3], "estresse_hidrico_salino")
        ]

        alertas_risco = []
        categorias_ativas = []
        for nome_label, prob, cat_key in pragas_names:
            nivel = "Baixo"
            if prob >= 0.70:
                nivel = "Alto"
                categorias_ativas.append(cat_key)
            elif prob >= 0.40:
                nivel = "Moderado"
                categorias_ativas.append(cat_key)

            alertas_risco.append({
                "patogeno_estresse": nome_label,
                "probabilidade": round(float(prob) * 100, 1),
                "nivel_alerta": nivel
            })

        # 6. Diagnóstico do Solo
        diag_solo = diagnosticar_solo(
            ph=float(dados_arduino.get("pH", 6.5)),
            nitrogenio=float(dados_arduino.get("nitrogenio_N_ppm", 4.0)),
            fosforo=float(dados_arduino.get("fosforo_P_ppm", 3.0)),
            potassio=float(dados_arduino.get("potassio_K_ppm", 2.5)),
            condutividade=float(dados_arduino.get("condutividade_eletrica_dS/m", 1.0)),
            compactacao=float(dados_arduino.get("compactacao_solo_kPa", 1500.0))
        )

        # 7. Formulação do Plano de Ação Personalizado (Orgânico vs Convencional)
        tipo_manejo = "organico" if is_organico else "convencional"
        catalogo_manejo = MANEJO_CATALOGO[tipo_manejo]

        recomendacoes_praticas = []
        if not categorias_ativas:
            recomendacoes_praticas.append("Manter monitoramento fitossanitário de rotina nas entrelinhas.")
        else:
            for cat in categorias_ativas:
                for item in catalogo_manejo.get(cat, []):
                    recomendacoes_praticas.append(item)

        # Adiciona manejo nutricional com base no solo
        if diag_solo["ph"]["valor"] < 5.8:
            recomendacoes_praticas.append(
                "Realizar calagem com calcário dolomítico (PRNT > 80%) para elevar saturação por bases para 60-70%."
            )
        for nut_item in catalogo_manejo.get("nutricao_npk", []):
            recomendacoes_praticas.append(f"[Nutrição/Correção] {nut_item}")

        # Remove duplicatas mantendo a ordem
        recomendacoes_praticas = list(dict.fromkeys(recomendacoes_praticas))

        # 8. Estruturação do Relatório Final
        relatorio = {
            "meta": {
                "coordenadas": {"latitude": lat, "longitude": lon},
                "cultura": cultura_norm,
                "kc_cultura": CULTURAS_CONFIG[cultura_norm].kc,
                "regime_cultivo": "Cultivo Orgânico Certificado" if is_organico else "Cultivo Convencional",
                "tipo_solo": tipo_solo
            },
            "decisao_irrigacao": {
                "classe_codigo": classe_irrig_final,
                "status": status_irrig,
                "volume_agua_recomendado_mm": volume_mm_final,
                "probabilidades_modelo": {
                    "nao_irrigar_pct": round(float(probs_irrig[0]) * 100, 1),
                    "irrigar_leve_pct": round(float(probs_irrig[1]) * 100, 1),
                    "irrigar_abundante_pct": round(float(probs_irrig[2]) * 100, 1)
                },
                "trava_seguranca_meteorologica": {
                    "acionada": trava_chuva_acionada,
                    "justificativa": motivo_irrig
                }
            },
            "analise_microclimatica_open_meteo": {
                "temperatura_2m_celsius": dados_clima["temperature_2m"],
                "umidade_relativa_2m_pct": dados_clima["relative_humidity_2m"],
                "ponto_orvalho_celsius": dados_clima["dew_point_2m"],
                "temperatura_solo_6cm_celsius": dados_clima["soil_temperature_6cm"],
                "umidade_profunda_0_a_27cm_m3_m3": round((dados_clima["soil_moisture_0_to_1cm"] + dados_clima["soil_moisture_9_to_27cm"]) / 2.0, 3),
                "probabilidade_chuva_6h_pct": dados_clima["precipitation_probability_max_6h"],
                "chuva_prevista_6h_mm": dados_clima["rain_sum_6h"],
                "chuva_prevista_12h_mm": dados_clima["rain_sum_12h"],
                "evapotranspiracao_et0_12h_mm": dados_clima["et0_fao_sum_12h"],
                "velocidade_vento_10m_kmh": dados_clima["wind_speed_10m"]
            },
            "alertas_risco_patogenos": alertas_risco,
            "diagnostico_quimico_fisico_solo": diag_solo,
            "plano_de_acao_prescritivo": {
                "enquadramento_legal": "Insumos permitidos conforme Lei Orgânica 10.831 / MAPA" if is_organico else "Manejo Químico Integrado de Pragas (MIP)",
                "acoes_e_insumos_recomendados": recomendacoes_praticas
            }
        }

        return relatorio


def np_argmax(arr) -> int:
    max_idx = 0
    max_val = arr[0]
    for i in range(1, len(arr)):
        if arr[i] > max_val:
            max_val = arr[i]
            max_idx = i
    return max_idx


_agro_system_instance: Optional[AgroDecisionSystem] = None


def predict_agro_system(
    lat: float,
    lon: float,
    cultura: str,
    is_organico: bool,
    dados_arduino_dict: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Função principal de alto nível para predição agroclimática.
    """
    global _agro_system_instance
    if _agro_system_instance is None:
        _agro_system_instance = AgroDecisionSystem()

    resultado = _agro_system_instance.predict(
        lat=lat,
        lon=lon,
        cultura=cultura,
        is_organico=is_organico,
        dados_arduino=dados_arduino_dict
    )
    return resultado


import sys
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def formatar_relatorio_terminal(relatorio: Dict[str, Any]) -> str:
    """
    Renderiza um relatório agronômico visualmente formatado para console/CLI.
    """
    meta = relatorio["meta"]
    irrig = relatorio["decisao_irrigacao"]
    clima = relatorio["analise_microclimatica_open_meteo"]
    solo = relatorio["diagnostico_quimico_fisico_solo"]
    plano = relatorio["plano_de_acao_prescritivo"]

    linhas = []
    linhas.append("=" * 72)
    linhas.append("  RELATÓRIO DE DECISÃO AGRONÔMICA & IoT - REDE NEURAL MULTITAREFA")
    linhas.append("=" * 72)
    linhas.append(f"  Localização: {meta['coordenadas']['latitude']}°, {meta['coordenadas']['longitude']}°")
    linhas.append(f"  Cultura: {meta['cultura'].upper()} (Kc: {meta['kc_cultura']}) | Solo: {meta['tipo_solo'].upper()}")
    linhas.append(f"  Regime de Cultivo: {meta['regime_cultivo'].upper()}")
    linhas.append("-" * 72)

    linhas.append("\n>> [1. DECISÃO DE MANEJO HÍDRICO (IRRIGAÇÃO)]")
    linhas.append(f"  * Decisão Final: {irrig['status']}")
    linhas.append(f"  * Lâmina Recomendada: {irrig['volume_agua_recomendado_mm']} mm/m²")
    linhas.append(f"  * Probabilidades: Não Irrigar={irrig['probabilidades_modelo']['nao_irrigar_pct']}%, Leve/Moderada={irrig['probabilidades_modelo']['irrigar_leve_pct']}%, Abundante={irrig['probabilidades_modelo']['irrigar_abundante_pct']}%")
    if irrig["trava_seguranca_meteorologica"]["acionada"]:
        linhas.append(f"  [!] TRAVA METEOROLÓGICA ATIVA: {irrig['trava_seguranca_meteorologica']['justificativa']}")

    linhas.append("\n>> [2. CONDIÇÕES AGROMETEOROLÓGICAS (OPEN-METEO)]")
    linhas.append(f"  * Temperatura: {clima['temperatura_2m_celsius']}°C | Umidade Relativa: {clima['umidade_relativa_2m_pct']}%")
    linhas.append(f"  * Ponto de Orvalho: {clima['ponto_orvalho_celsius']}°C | Temp Solo (6cm): {clima['temperatura_solo_6cm_celsius']}°C")
    linhas.append(f"  * Vento (10m): {clima['velocidade_vento_10m_kmh']} km/h | Evapotranspiração ET0 (12h): {clima['evapotranspiracao_et0_12h_mm']} mm")
    linhas.append(f"  * Chuva Futura: 6h = {clima['chuva_prevista_6h_mm']} mm (Prob: {clima['probabilidade_chuva_6h_pct']}%), 12h = {clima['chuva_prevista_12h_mm']} mm")

    linhas.append("\n>> [3. DIAGNÓSTICO DO SOLO (SENSORES IOT)]")
    linhas.append(f"  * pH: {solo['ph']['valor']} -> {solo['ph']['status']}")
    linhas.append(f"  * N-P-K (ppm): N={solo['nitrogenio_n_ppm']['valor']} ({solo['nitrogenio_n_ppm']['status']}), P={solo['fosforo_p_ppm']['valor']} ({solo['fosforo_p_ppm']['status']}), K={solo['potassio_k_ppm']['valor']} ({solo['potassio_k_ppm']['status']})")
    linhas.append(f"  * Salinidade (CE): {solo['condutividade_eletrica_dS_m']['valor']} dS/m -> {solo['condutividade_eletrica_dS_m']['status']}")
    linhas.append(f"  * Compactação: {solo['compactacao_kpa']['valor']} kPa -> {solo['compactacao_kpa']['status']}")

    linhas.append("\n>> [4. RISCOS FITOSSANITÁRIOS IDENTIFICADOS]")
    for alerta in relatorio["alertas_risco_patogenos"]:
        simbolo = "[ALTO]" if alerta["nivel_alerta"] == "Alto" else ("[MODERADO]" if alerta["nivel_alerta"] == "Moderado" else "[BAIXO]")
        linhas.append(f"  {simbolo:<11} {alerta['patogeno_estresse']}: {alerta['probabilidade']}%")

    linhas.append("\n>> [5. PLANO DE AÇÃO PRESCRITIVO]")
    linhas.append(f"  Enquadramento: {plano['enquadramento_legal']}")
    for acao in plano["acoes_e_insumos_recomendados"]:
        linhas.append(f"  [+] {acao}")

    linhas.append("=" * 72)
    return "\n".join(linhas)
