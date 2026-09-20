import openmeteo_requests

import pandas as pd
import requests_cache
from retry_requests import retry

# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)

# Make sure all required weather variables are listed here
# The order of variables in hourly or daily is important to assign them correctly below
url = "https://api.open-meteo.com/v1/forecast"
params = {
	"latitude": -22.395727,
	"longitude": -47.545199,
	"daily": ["temperature_2m_max", "temperature_2m_min", "moon_phase", "et0_fao_evapotranspiration", "uv_index_max"],
	"hourly": ["temperature_2m", "relative_humidity_2m", "soil_temperature_0cm", "soil_temperature_6cm", "soil_temperature_18cm", "soil_temperature_54cm", "soil_moisture_0_to_1cm", "soil_moisture_1_to_3cm", "soil_moisture_3_to_9cm", "soil_moisture_9_to_27cm", "soil_moisture_27_to_81cm", "apparent_temperature", "precipitation_probability", "rain", "showers", "precipitation", "dew_point_2m", "cloud_cover", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high", "evapotranspiration", "et0_fao_evapotranspiration", "wind_speed_10m", "wind_speed_80m", "wind_speed_120m", "temperature_80m", "temperature_120m", "temperature_180m", "is_day"],
	"models": "best_match",
	"current": ["temperature_2m", "relative_humidity_2m", "is_day", "apparent_temperature", "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m", "precipitation", "rain", "showers", "cloud_cover"],
	"timezone": "America/Sao_Paulo",
	"forecast_hours": 1,
	"past_hours": 1,
}
responses = openmeteo.weather_api(url, params = params)

# Process first location. Add a for-loop for multiple locations or weather models
response = responses[0]
print(f"Coordinates: {response.Latitude()}°N {response.Longitude()}°E")
print(f"Elevation: {response.Elevation()} m asl")
print(f"Timezone: {response.Timezone()}{response.TimezoneAbbreviation()}")
print(f"Timezone difference to GMT+0: {response.UtcOffsetSeconds()}s")

# Process current data. The order of variables needs to be the same as requested.
current = response.Current()
current_temperature_2m = current.Variables(0).Value()
current_relative_humidity_2m = current.Variables(1).Value()
current_is_day = current.Variables(2).Value()
current_apparent_temperature = current.Variables(3).Value()
current_wind_speed_10m = current.Variables(4).Value()
current_wind_direction_10m = current.Variables(5).Value()
current_wind_gusts_10m = current.Variables(6).Value()
current_precipitation = current.Variables(7).Value()
current_rain = current.Variables(8).Value()
current_showers = current.Variables(9).Value()
current_cloud_cover = current.Variables(10).Value()

print(f"\nCurrent time: {current.Time()}")
print(f"Current temperature_2m: {current_temperature_2m}")
print(f"Current relative_humidity_2m: {current_relative_humidity_2m}")
print(f"Current is_day: {current_is_day}")
print(f"Current apparent_temperature: {current_apparent_temperature}")
print(f"Current wind_speed_10m: {current_wind_speed_10m}")
print(f"Current wind_direction_10m: {current_wind_direction_10m}")
print(f"Current wind_gusts_10m: {current_wind_gusts_10m}")
print(f"Current precipitation: {current_precipitation}")
print(f"Current rain: {current_rain}")
print(f"Current showers: {current_showers}")
print(f"Current cloud_cover: {current_cloud_cover}")

# Process hourly data. The order of variables needs to be the same as requested.
hourly = response.Hourly()
hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
hourly_relative_humidity_2m = hourly.Variables(1).ValuesAsNumpy()
hourly_soil_temperature_0cm = hourly.Variables(2).ValuesAsNumpy()
hourly_soil_temperature_6cm = hourly.Variables(3).ValuesAsNumpy()
hourly_soil_temperature_18cm = hourly.Variables(4).ValuesAsNumpy()
hourly_soil_temperature_54cm = hourly.Variables(5).ValuesAsNumpy()
hourly_soil_moisture_0_to_1cm = hourly.Variables(6).ValuesAsNumpy()
hourly_soil_moisture_1_to_3cm = hourly.Variables(7).ValuesAsNumpy()
hourly_soil_moisture_3_to_9cm = hourly.Variables(8).ValuesAsNumpy()
hourly_soil_moisture_9_to_27cm = hourly.Variables(9).ValuesAsNumpy()
hourly_soil_moisture_27_to_81cm = hourly.Variables(10).ValuesAsNumpy()
hourly_apparent_temperature = hourly.Variables(11).ValuesAsNumpy()
hourly_precipitation_probability = hourly.Variables(12).ValuesAsNumpy()
hourly_rain = hourly.Variables(13).ValuesAsNumpy()
hourly_showers = hourly.Variables(14).ValuesAsNumpy()
hourly_precipitation = hourly.Variables(15).ValuesAsNumpy()
hourly_dew_point_2m = hourly.Variables(16).ValuesAsNumpy()
hourly_cloud_cover = hourly.Variables(17).ValuesAsNumpy()
hourly_cloud_cover_low = hourly.Variables(18).ValuesAsNumpy()
hourly_cloud_cover_mid = hourly.Variables(19).ValuesAsNumpy()
hourly_cloud_cover_high = hourly.Variables(20).ValuesAsNumpy()
hourly_evapotranspiration = hourly.Variables(21).ValuesAsNumpy()
hourly_et0_fao_evapotranspiration = hourly.Variables(22).ValuesAsNumpy()
hourly_wind_speed_10m = hourly.Variables(23).ValuesAsNumpy()
hourly_wind_speed_80m = hourly.Variables(24).ValuesAsNumpy()
hourly_wind_speed_120m = hourly.Variables(25).ValuesAsNumpy()
hourly_temperature_80m = hourly.Variables(26).ValuesAsNumpy()
hourly_temperature_120m = hourly.Variables(27).ValuesAsNumpy()
hourly_temperature_180m = hourly.Variables(28).ValuesAsNumpy()
hourly_is_day = hourly.Variables(29).ValuesAsNumpy()

hourly_data = {
	"date": pd.date_range(
		start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
		end =  pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
		freq = pd.Timedelta(seconds = hourly.Interval()),
		inclusive = "left"
	).tz_convert(response.Timezone().decode())
}

hourly_data["temperature_2m"] = hourly_temperature_2m
hourly_data["relative_humidity_2m"] = hourly_relative_humidity_2m
hourly_data["soil_temperature_0cm"] = hourly_soil_temperature_0cm
hourly_data["soil_temperature_6cm"] = hourly_soil_temperature_6cm
hourly_data["soil_temperature_18cm"] = hourly_soil_temperature_18cm
hourly_data["soil_temperature_54cm"] = hourly_soil_temperature_54cm
hourly_data["soil_moisture_0_to_1cm"] = hourly_soil_moisture_0_to_1cm
hourly_data["soil_moisture_1_to_3cm"] = hourly_soil_moisture_1_to_3cm
hourly_data["soil_moisture_3_to_9cm"] = hourly_soil_moisture_3_to_9cm
hourly_data["soil_moisture_9_to_27cm"] = hourly_soil_moisture_9_to_27cm
hourly_data["soil_moisture_27_to_81cm"] = hourly_soil_moisture_27_to_81cm
hourly_data["apparent_temperature"] = hourly_apparent_temperature
hourly_data["precipitation_probability"] = hourly_precipitation_probability
hourly_data["rain"] = hourly_rain
hourly_data["showers"] = hourly_showers
hourly_data["precipitation"] = hourly_precipitation
hourly_data["dew_point_2m"] = hourly_dew_point_2m
hourly_data["cloud_cover"] = hourly_cloud_cover
hourly_data["cloud_cover_low"] = hourly_cloud_cover_low
hourly_data["cloud_cover_mid"] = hourly_cloud_cover_mid
hourly_data["cloud_cover_high"] = hourly_cloud_cover_high
hourly_data["evapotranspiration"] = hourly_evapotranspiration
hourly_data["et0_fao_evapotranspiration"] = hourly_et0_fao_evapotranspiration
hourly_data["wind_speed_10m"] = hourly_wind_speed_10m
hourly_data["wind_speed_80m"] = hourly_wind_speed_80m
hourly_data["wind_speed_120m"] = hourly_wind_speed_120m
hourly_data["temperature_80m"] = hourly_temperature_80m
hourly_data["temperature_120m"] = hourly_temperature_120m
hourly_data["temperature_180m"] = hourly_temperature_180m
hourly_data["is_day"] = hourly_is_day

hourly_dataframe = pd.DataFrame(data = hourly_data)
print("\nHourly data\n", hourly_dataframe)

# Process daily data. The order of variables needs to be the same as requested.
daily = response.Daily()
daily_temperature_2m_max = daily.Variables(0).ValuesAsNumpy()
daily_temperature_2m_min = daily.Variables(1).ValuesAsNumpy()
daily_moon_phase = daily.Variables(2).ValuesAsNumpy()
daily_et0_fao_evapotranspiration = daily.Variables(3).ValuesAsNumpy()
daily_uv_index_max = daily.Variables(4).ValuesAsNumpy()

daily_data = {
	"date": pd.date_range(
		start = pd.to_datetime(daily.Time(), unit = "s", utc = True),
		end =  pd.to_datetime(daily.TimeEnd(), unit = "s", utc = True),
		freq = pd.Timedelta(seconds = daily.Interval()),
		inclusive = "left"
	).tz_convert(response.Timezone().decode())
}

daily_data["temperature_2m_max"] = daily_temperature_2m_max
daily_data["temperature_2m_min"] = daily_temperature_2m_min
daily_data["moon_phase"] = daily_moon_phase
daily_data["et0_fao_evapotranspiration"] = daily_et0_fao_evapotranspiration
daily_data["uv_index_max"] = daily_uv_index_max

daily_dataframe = pd.DataFrame(data = daily_data)
print("\nDaily data\n", daily_dataframe)
