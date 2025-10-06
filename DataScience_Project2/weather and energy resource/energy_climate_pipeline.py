"""Energy & Climate Data Pipeline (2014–2024).

This module downloads historical daily weather data for select US cities using the
Open-Meteo archive API. It aggregates the daily observations to monthly values and
computes cooling and heating degree days using a 65°F baseline. The optional
`fetch_state_generation_monthly` helper demonstrates how state-level energy
mix data could be pulled from the EIA v2 API when the appropriate route and API
key are supplied via environment variables.

Running the module as a script will download the weather data, write the daily
per-city Parquet files to ``data/raw`` and a monthly combined Parquet file to
``data/curated``. The EIA step is skipped unless both ``EIA_API_KEY`` and
``EIA_ROUTE`` are configured.

Example::

    $ export EIA_API_KEY=...  # optional
    $ export EIA_ROUTE=electricity/...  # optional
    $ python energy_climate_pipeline.py
"""

from __future__ import annotations

import os
import pathlib
from typing import Dict

import pandas as pd
import requests
from dateutil import tz  # noqa: F401  # Imported for potential timezone validation in extensions
from tenacity import retry, stop_after_attempt, wait_exponential

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = pathlib.Path("data")
RAW_DIR = DATA_DIR / "raw"
CURATED_DIR = DATA_DIR / "curated"
RAW_DIR.mkdir(parents=True, exist_ok=True)
CURATED_DIR.mkdir(parents=True, exist_ok=True)

CITIES: Dict[str, Dict[str, object]] = {
    "NYC": {"lat": 40.7128, "lon": -74.0060, "tz": "America/New_York"},
    "LosAngeles": {"lat": 34.0522, "lon": -118.2437, "tz": "America/Los_Angeles"},
    "Seattle": {"lat": 47.6062, "lon": -122.3321, "tz": "America/Los_Angeles"},
}

START_DATE = "2014-01-01"
END_DATE = "2024-12-31"
BASE_TEMP_F = 65.0

EIA_API_KEY = os.getenv("EIA_API_KEY")
EIA_ROUTE = os.getenv("EIA_ROUTE")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def c_to_f(temp_c: float) -> float:
    """Convert Celsius to Fahrenheit."""

    return (temp_c * 9.0 / 5.0) + 32.0


def compute_hdd_cdd(mean_temp_f: pd.Series, base_f: float = BASE_TEMP_F) -> pd.DataFrame:
    """Return heating and cooling degree days for each observation."""

    hdd = (base_f - mean_temp_f).clip(lower=0.0)
    cdd = (mean_temp_f - base_f).clip(lower=0.0)
    return pd.DataFrame({"hdd_base65F": hdd, "cdd_base65F": cdd})


# ---------------------------------------------------------------------------
# Weather (Open-Meteo)
# ---------------------------------------------------------------------------
@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=1, max=10))
def fetch_open_meteo_daily(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    tz_name: str,
) -> pd.DataFrame:
    """Pull daily weather metrics from Open-Meteo archive API."""

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": [
            "temperature_2m_max",
            "temperature_2m_min",
            "temperature_2m_mean",
            "precipitation_sum",
            "rain_sum",
            "snowfall_sum",
            "weathercode",
        ],
        "timezone": tz_name,
    }

    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()
    payload = response.json()
    daily = payload.get("daily", {})
    if not daily or "time" not in daily:
        raise ValueError("Open-Meteo returned no daily data.")

    df = pd.DataFrame(daily)
    df.rename(
        columns={
            "time": "date",
            "temperature_2m_max": "tmax_c",
            "temperature_2m_min": "tmin_c",
            "temperature_2m_mean": "tmean_c",
            "precipitation_sum": "precip_mm",
            "rain_sum": "rain_mm",
            "snowfall_sum": "snow_mm",
            "weathercode": "weather_code",
        },
        inplace=True,
    )
    df["date"] = pd.to_datetime(df["date"])

    df["tmax_f"] = df["tmax_c"].apply(c_to_f)
    df["tmin_f"] = df["tmin_c"].apply(c_to_f)
    df["tmean_f"] = df["tmean_c"].apply(c_to_f)

    degree_days = compute_hdd_cdd(df["tmean_f"])
    df = pd.concat([df, degree_days], axis=1)
    return df


def build_weather_for_cities() -> pd.DataFrame:
    """Fetch weather for each city and persist daily and monthly datasets."""

    frames = []
    monthly_frames = []
    for city, meta in CITIES.items():
        print(f"[weather] fetching {city} ...")
        daily_df = fetch_open_meteo_daily(meta["lat"], meta["lon"], START_DATE, END_DATE, meta["tz"])
        daily_df.insert(0, "city", city)
        # 创建每个城市的原始数据目录
        city_raw_dir = RAW_DIR / city
        city_raw_dir.mkdir(parents=True, exist_ok=True)
        # 保存为Parquet
        daily_df.to_parquet(city_raw_dir / f"weather_daily_{city}.parquet", index=False)
        # 保存为JSON
        daily_df.to_json(city_raw_dir / f"weather_daily_{city}.json", orient="records", date_format="iso")
        frames.append(daily_df)

        # 计算每个城市的月度数据
        monthly_city = (
            daily_df.assign(
                year=lambda d: d["date"].dt.year,
                month=lambda d: d["date"].dt.month,
            )
            .groupby(["city", "year", "month"], as_index=False)
            .agg(
                {
                    "tmean_c": "mean",
                    "tmax_c": "mean",
                    "tmin_c": "mean",
                    "tmean_f": "mean",
                    "precip_mm": "sum",
                    "rain_mm": "sum",
                    "snow_mm": "sum",
                    "hdd_base65F": "sum",
                    "cdd_base65F": "sum",
                    "weather_code": "first",
                }
            )
            .rename(
                columns={
                    "tmean_c": "tmean_c_avg",
                    "tmax_c": "tmax_c_avg",
                    "tmin_c": "tmin_c_avg",
                    "tmean_f": "tmean_f_avg",
                    "precip_mm": "precip_mm_sum",
                    "rain_mm": "rain_mm_sum",
                    "snow_mm": "snow_mm_sum",
                }
            )
        )
        monthly_city["year_month"] = pd.to_datetime(
            monthly_city["year"].astype(str) + "-" + monthly_city["month"].astype(str).str.zfill(2) + "-01"
        )
        monthly_city = monthly_city.sort_values(["city", "year_month"])
        # 创建每个城市的curated目录
        city_curated_dir = CURATED_DIR / city
        city_curated_dir.mkdir(parents=True, exist_ok=True)
        # 保存为Parquet
        monthly_city.to_parquet(city_curated_dir / f"weather_monthly_{city}.parquet", index=False)
        # 保存为JSON
        monthly_city.to_json(city_curated_dir / f"weather_monthly_{city}.json", orient="records", date_format="iso")
        monthly_frames.append(monthly_city)

    # 合并所有城市的月度数据，继续保存总表
    daily_all = pd.concat(frames, ignore_index=True)
    monthly = pd.concat(monthly_frames, ignore_index=True)
    monthly.to_parquet(CURATED_DIR / "weather_monthly_by_city.parquet", index=False)
    monthly.to_json(CURATED_DIR / "weather_monthly_by_city.json", orient="records", date_format="iso")
    return monthly


# ---------------------------------------------------------------------------
# Optional: EIA v2 (State Mix)
# ---------------------------------------------------------------------------
@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=1, max=10))
def eia_query(route: str, params: dict) -> pd.DataFrame:
    """Generic EIA v2 query returning a DataFrame."""

    if not EIA_API_KEY:
        raise RuntimeError("Set EIA_API_KEY environment variable to use EIA endpoints.")

    url = f"https://api.eia.gov/v2/{route}"
    request_params = {**params, "api_key": EIA_API_KEY}
    response = requests.get(url, params=request_params, timeout=60)
    response.raise_for_status()
    payload = response.json()
    data = payload.get("response", {}).get("data", [])
    if not data:
        return pd.DataFrame()
    return pd.DataFrame(data)


def fetch_state_generation_monthly() -> pd.DataFrame:
    """Example scaffold for state-level generation by fuel."""

    if not EIA_ROUTE:
        raise RuntimeError("Set the EIA_ROUTE environment variable for the desired EIA dataset.")

    params = {
        "data": ["value"],
        "facets[stateId][]": ["NY", "CA", "WA"],
        "frequency": "monthly",
        "start": "2014-01",
        "end": "2024-12",
    }
    df = eia_query(route=EIA_ROUTE, params=params)
    if df.empty:
        print("[eia] Query returned no data. Validate route and facets via the EIA API Browser.")
        return df

    out_path = RAW_DIR / "eia_generation_state_monthly.parquet"
    df.to_parquet(out_path, index=False)
    return df


# ---------------------------------------------------------------------------
# Main Entrypoint
# ---------------------------------------------------------------------------
def main() -> None:
    print("=== Step 1: Weather (Open-Meteo) ===")
    monthly = build_weather_for_cities()
    print(monthly.head())

    if EIA_API_KEY and EIA_ROUTE:
        print("=== Step 2 (Optional): EIA state generation mix ===")
        eia_df = fetch_state_generation_monthly()
        if not eia_df.empty:
            print(eia_df.head())
    else:
        print("=== Skipping EIA step (set EIA_API_KEY and EIA_ROUTE to enable) ===")


if __name__ == "__main__":
    main()
