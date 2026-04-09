"""Fetch and cache race-day humidity and wind from the Open-Meteo historical API.

Temperature comes from config.RACE_TEMP_F (John Hancock PDF, authoritative on-site measurements).
Humidity and wind come from Open-Meteo's ERA5 reanalysis averaged over the 10am-3pm race window.
First call hits the API (~10s for 20 years); subsequent calls read from a JSON cache file.
Used by rq2.sensitivity_weather() to build the weather-augmented mixed-effects model.
"""
import json
import numpy as np
import requests
from src import config as cfg


def _fetch_open_meteo():
    """Call Open-Meteo's archive API for each race date and average the 10am-3pm window."""
    result = {}
    for year, dt in cfg.RACE_DATES.items():
        hourly = requests.get('https://archive-api.open-meteo.com/v1/archive', params={
            'latitude': 42.2287, 'longitude': -71.5226,
            'start_date': dt, 'end_date': dt,
            'hourly': 'relative_humidity_2m,wind_speed_10m',
            'wind_speed_unit': 'mph', 'timezone': 'America/New_York',
        }).json()['hourly']
        result[str(year)] = {
            'humidity_pct': round(float(np.mean([hourly['relative_humidity_2m'][h] for h in range(10, 16)]))),
            'wind_mph': round(float(np.mean([hourly['wind_speed_10m'][h] for h in range(10, 16)])), 1),
        }
    return result


def get_race_weather():
    """Return (humidity_pct, wind_mph) dicts keyed by year. Cached after first fetch."""
    if cfg.WEATHER_CACHE.exists():
        cached = json.loads(cfg.WEATHER_CACHE.read_text())
    else:
        cached = _fetch_open_meteo()
        cfg.WEATHER_CACHE.write_text(json.dumps(cached))
    humidity = {int(y): v['humidity_pct'] for y, v in cached.items()}
    wind = {int(y): v['wind_mph'] for y, v in cached.items()}
    return humidity, wind
