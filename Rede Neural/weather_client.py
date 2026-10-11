"""
weather_client.py
=================
Módulo de integração com a API Open-Meteo.
Extrai variáveis climáticas atuais e previsão horária (forecast 6h e 12h)
para alimentar a engenharia de features da Rede Neural Multitarefa.
Inclui cache local, retries exponenciais e fallback resiliente.
"""

import logging
from typing import Dict, Any, Optional
import numpy as np

# Configuração de logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("WeatherClient")


class OpenMeteoWeatherClient:
    """
    Cliente para consulta da API Open-Meteo com caching e extração
    de agregados climáticos essenciais para a tomada de decisão agrícola.
    """

    def __init__(self, cache_expire_after: int = 1800):
        self.cache_expire_after = cache_expire_after
        self.client = None
        self._setup_client()

    def _setup_client(self):
        try:
            import openmeteo_requests
            import requests_cache
            from retry_requests import retry

            cache_session = requests_cache.CachedSession(".cache_weather", expire_after=self.cache_expire_after)
            retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
            self.client = openmeteo_requests.Client(session=retry_session)
        except Exception as e:
            logger.warning(f"Não foi possível inicializar o cliente Open-Meteo oficial: {e}. O modo fallback estará ativo.")
            self.client = None

    def get_weather_features(self, lat: float, lon: float) -> Dict[str, float]:
        """
        Consulta a API Open-Meteo e retorna o dicionário com as features
        climáticas necessárias para o modelo de Deep Learning:
        - soil_moisture_0_to_1cm, 1_to_3cm, 3_to_9cm, 9_to_27cm
        - et0_fao_sum_12h (mm acumulado 12h)
        - precipitation_probability_max_6h (%)
        - rain_sum_6h (mm acumulado 6h)
        - rain_sum_12h (mm acumulado 12h)
        - temperature_2m (°C)
        - relative_humidity_2m (%)
        - dew_point_2m (°C)
        - soil_temperature_6cm (°C)
        - wind_speed_10m (km/h)
        """
        if self.client is None:
            return self._get_fallback_features(lat, lon, motivo="Cliente Open-Meteo não disponível")

        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": [
                "temperature_2m",
                "relative_humidity_2m",
                "soil_temperature_6cm",
                "soil_moisture_0_to_1cm",
                "soil_moisture_1_to_3cm",
                "soil_moisture_3_to_9cm",
                "soil_moisture_9_to_27cm",
                "precipitation_probability",
                "rain",
                "precipitation",
                "dew_point_2m",
                "et0_fao_evapotranspiration",
                "wind_speed_10m"
            ],
            "current": [
                "temperature_2m",
                "relative_humidity_2m",
                "precipitation",
                "rain",
                "wind_speed_10m"
            ],
            "timezone": "America/Sao_Paulo",
            "forecast_hours": 24,
            "past_hours": 1
        }

        try:
            responses = self.client.weather_api(url, params=params)
            response = responses[0]

            # Processamento Current
            current = response.Current()
            curr_temp = float(current.Variables(0).Value())
            curr_rh = float(current.Variables(1).Value())
            curr_wind = float(current.Variables(4).Value())

            # Processamento Hourly
            hourly = response.Hourly()
            h_temp = hourly.Variables(0).ValuesAsNumpy()
            h_rh = hourly.Variables(1).ValuesAsNumpy()
            h_soil_temp6 = hourly.Variables(2).ValuesAsNumpy()
            h_sm_0_1 = hourly.Variables(3).ValuesAsNumpy()
            h_sm_1_3 = hourly.Variables(4).ValuesAsNumpy()
            h_sm_3_9 = hourly.Variables(5).ValuesAsNumpy()
            h_sm_9_27 = hourly.Variables(6).ValuesAsNumpy()
            h_precip_prob = hourly.Variables(7).ValuesAsNumpy()
            h_rain = hourly.Variables(8).ValuesAsNumpy()
            h_precip = hourly.Variables(9).ValuesAsNumpy()
            h_dew_point = hourly.Variables(10).ValuesAsNumpy()
            h_et0 = hourly.Variables(11).ValuesAsNumpy()
            h_wind = hourly.Variables(12).ValuesAsNumpy()

            # Extração de janelas horárias: 6h e 12h
            idx_start = 1 if len(h_rain) > 24 else 0  # ignora past_hours se presente
            window_6h = slice(idx_start, idx_start + 6)
            window_12h = slice(idx_start, idx_start + 12)

            precip_prob_max_6h = float(np.nanmax(h_precip_prob[window_6h])) if len(h_precip_prob) >= 6 else 0.0
            rain_sum_6h = float(np.nansum(h_rain[window_6h])) if len(h_rain) >= 6 else 0.0
            rain_sum_12h = float(np.nansum(h_rain[window_12h])) if len(h_rain) >= 12 else 0.0
            et0_sum_12h = float(np.nansum(h_et0[window_12h])) if len(h_et0) >= 12 else 2.5

            soil_moist_0_1 = float(np.nanmean(h_sm_0_1[window_6h]))
            soil_moist_1_3 = float(np.nanmean(h_sm_1_3[window_6h]))
            soil_moist_3_9 = float(np.nanmean(h_sm_3_9[window_6h]))
            soil_moist_9_27 = float(np.nanmean(h_sm_9_27[window_6h]))

            soil_temp_6cm = float(np.nanmean(h_soil_temp6[window_6h]))
            dew_point = float(np.nanmean(h_dew_point[window_6h]))

            return {
                "soil_moisture_0_to_1cm": soil_moist_0_1,
                "soil_moisture_1_to_3cm": soil_moist_1_3,
                "soil_moisture_3_to_9cm": soil_moist_3_9,
                "soil_moisture_9_to_27cm": soil_moist_9_27,
                "et0_fao_sum_12h": max(0.2, et0_sum_12h),
                "precipitation_probability_max_6h": precip_prob_max_6h,
                "rain_sum_6h": rain_sum_6h,
                "rain_sum_12h": rain_sum_12h,
                "temperature_2m": curr_temp if not np.isnan(curr_temp) else float(np.nanmean(h_temp[window_6h])),
                "relative_humidity_2m": curr_rh if not np.isnan(curr_rh) else float(np.nanmean(h_rh[window_6h])),
                "dew_point_2m": dew_point,
                "soil_temperature_6cm": soil_temp_6cm,
                "wind_speed_10m": curr_wind if not np.isnan(curr_wind) else float(np.nanmean(h_wind[window_6h]))
            }

        except Exception as err:
            logger.warning(f"Erro na requisição à API Open-Meteo ({err}). Utilizando dados de contingência.")
            return self._get_fallback_features(lat, lon, motivo=str(err))

    def _get_fallback_features(self, lat: float, lon: float, motivo: str) -> Dict[str, float]:
        """
        Gera conjunto de features de contingência agrícola representativas
        quando o serviço remoto estiver indisponível.
        """
        logger.info(f"Modo Fallback ativado ({motivo}). Latitude: {lat}, Longitude: {lon}")
        return {
            "soil_moisture_0_to_1cm": 0.22,
            "soil_moisture_1_to_3cm": 0.25,
            "soil_moisture_3_to_9cm": 0.28,
            "soil_moisture_9_to_27cm": 0.30,
            "et0_fao_sum_12h": 2.8,
            "precipitation_probability_max_6h": 15.0,
            "rain_sum_6h": 0.0,
            "rain_sum_12h": 0.0,
            "temperature_2m": 24.5,
            "relative_humidity_2m": 68.0,
            "dew_point_2m": 18.2,
            "soil_temperature_6cm": 23.0,
            "wind_speed_10m": 11.5
        }


# Instância global reutilizável
weather_service = OpenMeteoWeatherClient()


def get_weather_features(lat: float, lon: float) -> Dict[str, float]:
    """
    Função de alto nível para consultar as features climáticas Open-Meteo.
    """
    return weather_service.get_weather_features(lat, lon)
