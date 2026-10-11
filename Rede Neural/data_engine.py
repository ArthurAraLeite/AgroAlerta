"""
data_engine.py
==============
Pipeline de processamento de dados para a Rede Neural Multitarefa Agronômica.
Lê o dataset físico-químico 'dados_solo_2casas.csv', simula compactação e variáveis
microclimáticas em conformidade com o Open-Meteo, calcula os rótulos de treinamento
segundo as leis da agronomia de precisão e normaliza as features para Deep Learning.
Processamento 100% vetorizado em NumPy para máxima performance.
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset

from agronomy_rules import (
    CULTURAS_CONFIG,
    SOLO_CONFIG
)


FEATURE_COLS_NUMERIC = [
    "pH",
    "teor_umildade_%",
    "nitrogenio_N_ppm",
    "fosforo_P_ppm",
    "potassio_K_ppm",
    "condutividade_eletrica_dS/m",
    "compactacao_solo_kPa",
    "soil_moisture_0_to_1cm",
    "soil_moisture_1_to_3cm",
    "soil_moisture_3_to_9cm",
    "soil_moisture_9_to_27cm",
    "et0_fao_sum_12h",
    "precipitation_probability_max_6h",
    "rain_sum_6h",
    "rain_sum_12h",
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "soil_temperature_6cm",
    "wind_speed_10m",
    "kc_cultura",
    "ponto_murcha_pct",
    "capacidade_campo_pct"
]

CATEGORICAL_SOLO = ["arenoso", "argiloso", "ideal"]
CATEGORICAL_CULTURA = ["alface", "cafe", "milho", "soja", "tomate"]


class AgroMultiTaskDataset(Dataset):
    """
    Dataset PyTorch para treinamento multitarefa.
    """
    def __init__(self, X: np.ndarray, y_irrig: np.ndarray, y_pragas: np.ndarray, y_volume: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y_irrig = torch.tensor(y_irrig, dtype=torch.long)
        self.y_pragas = torch.tensor(y_pragas, dtype=torch.float32)
        self.y_volume = torch.tensor(y_volume, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_irrig[idx], self.y_pragas[idx], self.y_volume[idx]


class AgroDataEngine:
    """
    Motor de dados para treinamento e pré-processamento de inferência.
    """

    def __init__(self, scaler_path: str = "agro_scaler_meta.pkl"):
        self.scaler_path = scaler_path
        self.scaler = StandardScaler()
        self.meta: Dict[str, Any] = {}

    def load_and_augment_dataset(
        self,
        csv_path: str = "dados_solo_2casas.csv",
        n_samples: int = 50000,
        random_seed: int = 42
    ) -> pd.DataFrame:
        """
        Carrega amostra do dataset real de solo e gera fusão
        com condições meteorológicas e atributos de cultura de forma ultra-rápida.
        """
        np.random.seed(random_seed)

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Arquivo de solo {csv_path} não encontrado!")

        print(f"Carregando amostra de {n_samples:,} registros de '{csv_path}'...", flush=True)
        # Carrega com folga para amostrar uniformemente
        rows_to_read = min(1000000, max(n_samples, n_samples * 2))
        df = pd.read_csv(csv_path, sep=";", nrows=rows_to_read)
        if len(df) > n_samples:
            df = df.sample(n=n_samples, random_state=random_seed).reset_index(drop=True)

        n = len(df)
        print(f"Base de solo carregada: {n} amostras. Aplicando engenharia de features vetorizada...", flush=True)

        # 1. Compactação do solo (kPa) - 1000 a 3000 kPa
        df["compactacao_solo_kPa"] = np.random.uniform(1000.0, 3000.0, size=n).round(1)

        # 2. Atribuição de Cultura e Manejo
        culturas = list(CULTURAS_CONFIG.keys())
        df["cultura"] = np.random.choice(culturas, size=n)
        df["is_organico"] = np.random.choice([0, 1], size=n, p=[0.6, 0.4])

        # Mapeamentos agronômicos
        kc_map = {k: v.kc for k, v in CULTURAS_CONFIG.items()}
        pm_map = {k: v.ponto_murcha_pct for k, v in CULTURAS_CONFIG.items()}
        prof_map = {k: v.profundidade_raiz_m for k, v in CULTURAS_CONFIG.items()}

        cc_map = {k: v.capacidade_campo_pct for k, v in SOLO_CONFIG.items()}
        pc_map = {k: v.ponto_critico_pct for k, v in SOLO_CONFIG.items()}
        da_map = {k: v.densidade_aparente_g_cm3 for k, v in SOLO_CONFIG.items()}

        df["kc_cultura"] = df["cultura"].map(kc_map).astype(np.float32)
        df["ponto_murcha_pct"] = df["cultura"].map(pm_map).astype(np.float32)
        prof_raiz = df["cultura"].map(prof_map).values.astype(np.float32)

        df["capacidade_campo_pct"] = df["tipo_solo"].map(cc_map).fillna(7.2).astype(np.float32)
        ponto_critico = df["tipo_solo"].map(pc_map).fillna(3.6).values.astype(np.float32)
        dens_aparente = df["tipo_solo"].map(da_map).fillna(1.3).values.astype(np.float32)

        # 3. Geração Climática Vetorizada (Open-Meteo)
        df["temperature_2m"] = np.random.normal(loc=26.0, scale=5.5, size=n).clip(12.0, 42.0).round(1)
        df["relative_humidity_2m"] = np.random.normal(loc=65.0, scale=18.0, size=n).clip(20.0, 99.0).round(1)
        df["dew_point_2m"] = (df["temperature_2m"] - ((100.0 - df["relative_humidity_2m"]) / 5.0)).round(1)
        df["soil_temperature_6cm"] = (df["temperature_2m"] + np.random.uniform(-3.0, 2.0, size=n)).round(1)
        df["wind_speed_10m"] = np.random.exponential(scale=10.0, size=n).clip(1.0, 45.0).round(1)

        base_sm = (df["teor_umildade_%"] / 25.0).values
        df["soil_moisture_0_to_1cm"] = np.clip(base_sm * np.random.uniform(0.7, 1.1, size=n), 0.05, 0.50).round(3)
        df["soil_moisture_1_to_3cm"] = np.clip(base_sm * np.random.uniform(0.8, 1.15, size=n), 0.08, 0.50).round(3)
        df["soil_moisture_3_to_9cm"] = np.clip(base_sm * np.random.uniform(0.9, 1.2, size=n), 0.10, 0.50).round(3)
        df["soil_moisture_9_to_27cm"] = np.clip(base_sm * np.random.uniform(0.95, 1.25, size=n), 0.12, 0.50).round(3)

        df["et0_fao_sum_12h"] = np.random.uniform(1.2, 5.8, size=n).round(2)
        has_rain_event = np.random.rand(n) < 0.28
        df["precipitation_probability_max_6h"] = np.where(
            has_rain_event,
            np.random.uniform(60.0, 98.0, size=n),
            np.random.uniform(2.0, 45.0, size=n)
        ).round(1)

        rain_amount_6h = np.where(
            has_rain_event,
            np.random.exponential(scale=7.5, size=n),
            0.0
        ).round(2)
        df["rain_sum_6h"] = rain_amount_6h
        df["rain_sum_12h"] = (rain_amount_6h + np.where(has_rain_event, np.random.exponential(scale=4.0, size=n), 0.0)).round(2)

        # 4. Cálculo Vetorizado dos Rótulos Agronômicos
        print("Calculando rótulos agronômicos vetorizados (FAO-56 e fitossanidade)...", flush=True)
        u_atual = df["teor_umildade_%"].values.astype(np.float32)
        cc = df["capacidade_campo_pct"].values.astype(np.float32)
        kc = df["kc_cultura"].values.astype(np.float32)
        comp = df["compactacao_solo_kPa"].values.astype(np.float32)
        et0 = df["et0_fao_sum_12h"].values.astype(np.float32)
        rain6 = df["rain_sum_6h"].values.astype(np.float32)
        rain12 = df["rain_sum_12h"].values.astype(np.float32)
        prob6 = df["precipitation_probability_max_6h"].values.astype(np.float32)

        # Trava de chuva
        trava_chuva = (prob6 > 70.0) | (rain6 >= 5.0)

        # Balanço hídrico
        deficit_u = np.maximum(0.0, cc - u_atual)
        faixa_disp = np.maximum(0.1, cc - ponto_critico)
        score_def = np.minimum(1.0, deficit_u / (faixa_disp * 1.5))

        etc_12h = np.maximum(0.0, et0 * kc)
        deficit_atm = np.maximum(0.0, etc_12h - rain12)
        score_bal = np.minimum(1.0, deficit_atm / 6.0)

        score_kc = (kc - 0.4) / (1.25 - 0.4)
        score_comp = np.clip((comp - 1000.0) / 2000.0, 0.0, 1.0)
        score_context = 0.6 * score_kc + 0.4 * (1.0 - 0.3 * score_comp)

        indice_rega = (0.40 * score_def) + (0.35 * score_bal) + (0.25 * score_context)

        prof_dm = prof_raiz * 10.0
        lamina_solo = np.maximum(0.0, (cc - u_atual) * dens_aparente * prof_dm * 0.1)
        lamina_liq = np.maximum(0.0, lamina_solo + deficit_atm)

        solo_saturado = (u_atual >= cc) & (rain12 >= etc_12h)
        indice_rega = np.where(solo_saturado, 0.0, indice_rega)
        lamina_liq = np.where(solo_saturado, 0.0, lamina_liq)

        # Categorização de irrigação
        cond_zero = (indice_rega < 0.30) | (lamina_liq < 1.0) | trava_chuva
        cond_leve = (indice_rega < 0.65) | (lamina_liq < 7.0)

        y_irrig = np.where(cond_zero, 0, np.where(cond_leve, 1, 2))
        y_vol = np.where(
            cond_zero,
            0.0,
            np.where(
                cond_leve,
                np.clip(lamina_liq, 1.5, 6.5),
                np.clip(lamina_liq, 7.0, 25.0)
            )
        ).round(2)

        df["target_irrigacao"] = y_irrig
        df["target_volume_mm"] = y_vol

        # 5. Fitossanidade Vetorizada
        t = df["temperature_2m"].values
        rh = df["relative_humidity_2m"].values
        ce = df["condutividade_eletrica_dS/m"].values
        pm = df["ponto_murcha_pct"].values

        # Fungos: UR >= 78% e 18 <= T <= 28.5
        cond_f = (rh >= 78.0) & (t >= 18.0) & (t <= 28.5)
        df["target_fungos"] = cond_f.astype(int)

        # Insetos/Lagartas: 22 <= T <= 32 e 45 <= UR <= 75
        cond_ins = (t >= 22.0) & (t <= 32.0) & (rh >= 45.0) & (rh <= 75.0)
        df["target_insetos"] = cond_ins.astype(int)

        # Ácaros: T >= 29.5 e UR <= 45
        cond_ac = (t >= 29.5) & (rh <= 45.0)
        df["target_acaros"] = cond_ac.astype(int)

        # Estresse Salino/Hídrico: CE >= 3.0 ou umidade <= PM
        cond_est = (ce >= 3.0) | (u_atual <= pm)
        df["target_estresse"] = cond_est.astype(int)

        print(f"Dataset pronto! Distribuição de Irrigação:\n{df['target_irrigacao'].value_counts(normalize=True).round(3)}", flush=True)
        return df

    def prepare_feature_matrix(self, df: pd.DataFrame, is_train: bool = True) -> np.ndarray:
        """
        Gera a matriz numérica normalizada pronta para o PyTorch,
        incluindo one-hot encoding para tipo_solo e cultura.
        """
        # 1. One-Hot Solo
        solo_ohe = np.zeros((len(df), len(CATEGORICAL_SOLO)), dtype=np.float32)
        for i, s in enumerate(CATEGORICAL_SOLO):
            solo_ohe[:, i] = (df["tipo_solo"].str.lower() == s).astype(np.float32)

        # 2. One-Hot Cultura
        cult_ohe = np.zeros((len(df), len(CATEGORICAL_CULTURA)), dtype=np.float32)
        for i, c in enumerate(CATEGORICAL_CULTURA):
            cult_ohe[:, i] = (df["cultura"].str.lower() == c).astype(np.float32)

        # 3. is_organico (binário)
        organico_col = df["is_organico"].values.astype(np.float32).reshape(-1, 1)

        # 4. Numéricas contínuas normalizadas
        num_vals = df[FEATURE_COLS_NUMERIC].values.astype(np.float32)
        if is_train:
            num_scaled = self.scaler.fit_transform(num_vals)
            self.meta = {
                "numeric_cols": FEATURE_COLS_NUMERIC,
                "categorical_solo": CATEGORICAL_SOLO,
                "categorical_cultura": CATEGORICAL_CULTURA,
                "total_features": num_scaled.shape[1] + solo_ohe.shape[1] + cult_ohe.shape[1] + 1
            }
            with open(self.scaler_path, "wb") as f:
                pickle.dump({"scaler": self.scaler, "meta": self.meta}, f)
        else:
            num_scaled = self.scaler.transform(num_vals)

        # Concatenação final
        X = np.hstack([num_scaled, solo_ohe, cult_ohe, organico_col])
        return X

    def transform_single_sample(
        self,
        tipo_solo: str,
        cultura: str,
        is_organico: bool,
        dados_arduino: Dict[str, float],
        dados_clima: Dict[str, float]
    ) -> np.ndarray:
        """
        Converte uma única leitura (Arduino + Open-Meteo) em um tensor de entrada normalizado.
        """
        if not hasattr(self.scaler, "mean_"):
            with open(self.scaler_path, "rb") as f:
                saved = pickle.load(f)
                self.scaler = saved["scaler"]
                self.meta = saved["meta"]

        cult_info = CULTURAS_CONFIG.get(cultura.lower(), CULTURAS_CONFIG["milho"])
        solo_info = SOLO_CONFIG.get(tipo_solo.lower(), SOLO_CONFIG["ideal"])

        row_dict = {
            "pH": float(dados_arduino.get("pH", 6.5)),
            "teor_umildade_%": float(dados_arduino.get("teor_umildade_%", 3.0)),
            "nitrogenio_N_ppm": float(dados_arduino.get("nitrogenio_N_ppm", 3.5)),
            "fosforo_P_ppm": float(dados_arduino.get("fosforo_P_ppm", 3.0)),
            "potassio_K_ppm": float(dados_arduino.get("potassio_K_ppm", 2.5)),
            "condutividade_eletrica_dS/m": float(dados_arduino.get("condutividade_eletrica_dS/m", 1.2)),
            "compactacao_solo_kPa": float(dados_arduino.get("compactacao_solo_kPa", 1600.0)),
            "soil_moisture_0_to_1cm": float(dados_clima.get("soil_moisture_0_to_1cm", 0.20)),
            "soil_moisture_1_to_3cm": float(dados_clima.get("soil_moisture_1_to_3cm", 0.22)),
            "soil_moisture_3_to_9cm": float(dados_clima.get("soil_moisture_3_to_9cm", 0.25)),
            "soil_moisture_9_to_27cm": float(dados_clima.get("soil_moisture_9_to_27cm", 0.28)),
            "et0_fao_sum_12h": float(dados_clima.get("et0_fao_sum_12h", 2.5)),
            "precipitation_probability_max_6h": float(dados_clima.get("precipitation_probability_max_6h", 10.0)),
            "rain_sum_6h": float(dados_clima.get("rain_sum_6h", 0.0)),
            "rain_sum_12h": float(dados_clima.get("rain_sum_12h", 0.0)),
            "temperature_2m": float(dados_clima.get("temperature_2m", 25.0)),
            "relative_humidity_2m": float(dados_clima.get("relative_humidity_2m", 60.0)),
            "dew_point_2m": float(dados_clima.get("dew_point_2m", 17.0)),
            "soil_temperature_6cm": float(dados_clima.get("soil_temperature_6cm", 24.0)),
            "wind_speed_10m": float(dados_clima.get("wind_speed_10m", 10.0)),
            "kc_cultura": cult_info.kc,
            "ponto_murcha_pct": cult_info.ponto_murcha_pct,
            "capacidade_campo_pct": solo_info.capacidade_campo_pct
        }

        num_vector = np.array([[row_dict[col] for col in FEATURE_COLS_NUMERIC]], dtype=np.float32)
        num_scaled = self.scaler.transform(num_vector)

        solo_ohe = np.array([[(1.0 if tipo_solo.lower() == s else 0.0) for s in CATEGORICAL_SOLO]], dtype=np.float32)
        cult_ohe = np.array([[(1.0 if cultura.lower() == c else 0.0) for c in CATEGORICAL_CULTURA]], dtype=np.float32)
        org_val = np.array([[1.0 if is_organico else 0.0]], dtype=np.float32)

        x_out = np.hstack([num_scaled, solo_ohe, cult_ohe, org_val])
        return x_out
