"""Race-day humidity and wind from Open-Meteo ERA5 (10am-3pm average), cached to JSON.

Temperature from config.RACE_TEMP_F (John Hancock PDF). Called by rq2._add_weather_cols.
"""
import json
import numpy as np
import requests
from boston_marathon import config as cfg


def _fetch_open_meteo():
    """Query the Open-Meteo archive API for each race date and average the midday window."""
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
    """Return two dicts keyed by year: (humidity_pct, wind_mph). Reads from cache if available."""
    if cfg.WEATHER_CACHE.exists():
        cached = json.loads(cfg.WEATHER_CACHE.read_text())
    else:
        cached = _fetch_open_meteo()
        cfg.WEATHER_CACHE.write_text(json.dumps(cached))
    humidity = {int(y): v['humidity_pct'] for y, v in cached.items()}
    wind = {int(y): v['wind_mph'] for y, v in cached.items()}
    return humidity, wind
