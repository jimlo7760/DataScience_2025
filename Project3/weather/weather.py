import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry
import json
import time
from datetime import datetime

# ==========================================
# CONFIGURATION
# ==========================================
START_DATE = "2000-01-01"
END_DATE = datetime.now().strftime('%Y-%m-%d')

# 3 Representative Points per State
STATE_COORDS = {
    # "iowa": [
    #     {"lat": 43.0, "lon": -93.6}, 
    #     {"lat": 41.6, "lon": -93.6}, 
    #     {"lat": 40.7, "lon": -93.6}
    # ],
    # "illinois": [
    #     {"lat": 41.8, "lon": -89.0}, 
    #     {"lat": 40.0, "lon": -89.0}, 
    #     {"lat": 38.0, "lon": -89.0}
    # ],
    "kansas": [
        {"lat": 39.5, "lon": -98.0}, 
        {"lat": 38.5, "lon": -98.0}, 
        {"lat": 37.5, "lon": -98.0}
    ]
}

def get_weekly_state_data(state_name, coords):
    print(f"Processing {state_name.upper()}...", end=" ", flush=True)
    
    # Setup Open-Meteo Client
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)
    
    url = "https://archive-api.open-meteo.com/v1/archive"
    state_frames = []

    # 1. Fetch Daily Data
    for point in coords:
        try:
            params = {
                "latitude": point['lat'],
                "longitude": point['lon'],
                "start_date": START_DATE,
                "end_date": END_DATE,
                "daily": ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "shortwave_radiation_sum"],
                "temperature_unit": "fahrenheit",
                "precipitation_unit": "inch",
                "timezone": "auto"
            }
            
            responses = openmeteo.weather_api(url, params=params)
            response = responses[0]
            
            daily = response.Daily()
            daily_data = {
                "date": pd.date_range(
                    start=pd.to_datetime(daily.Time(), unit="s", utc=True),
                    end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
                    freq=pd.Timedelta(seconds=daily.Interval()),
                    inclusive="left"
                ),
                "tmax": daily.Variables(0).ValuesAsNumpy(),
                "tmin": daily.Variables(1).ValuesAsNumpy(),
                "precip": daily.Variables(2).ValuesAsNumpy(),
                "solar": daily.Variables(3).ValuesAsNumpy()
            }
            state_frames.append(pd.DataFrame(data=daily_data))
            time.sleep(10) # Pause to respect API limits
            
        except Exception as e:
            print(f"Error on point {point}: {e}")
            time.sleep(60)
            continue

    if not state_frames: return []

    # 2. Average & Resample
    combined = pd.concat(state_frames)
    daily_avg = combined.groupby("date").mean()
    
    weekly_avg = daily_avg.resample('W-MON').agg({
        'tmax': 'mean',
        'tmin': 'mean',
        'precip': 'sum',
        'solar': 'sum'
    }).reset_index()

    weekly_avg['date'] = weekly_avg['date'].dt.strftime('%Y-%m-%d')
    return weekly_avg.round(2).to_dict(orient="records")

def generate_metadata(state_name, data_list, coords):
    """Generates a descriptive metadata dictionary."""
    if not data_list: return {}
    
    return {
        "dataset_name": f"{state_name.title()} Weekly Crop Weather",
        "description": "Weekly aggregated weather data suitable for crop growth modeling (GDD, Photosynthesis, Water).",
        "source": "Open-Meteo Historical Weather API (ERA5 Reanalysis)",
        "temporal_coverage": {
            "start_date": data_list[0]['date'],
            "end_date": data_list[-1]['date'],
            "frequency": "Weekly (Aggregated Mondays)"
        },
        "spatial_coverage": {
            "method": "Average of 3 lat/lon points (North, Center, South)",
            "coordinates_used": coords
        },
        "columns": {
            "date": "Start date of the week (YYYY-MM-DD)",
            "tmax": "Average Maximum Air Temperature (Fahrenheit)",
            "tmin": "Average Minimum Air Temperature (Fahrenheit)",
            "precip": "Total Accumulated Precipitation (Inches)",
            "solar": "Total Accumulated Solar Radiation (MegaJoules per square meter)"
        },
        "aggregation_logic": {
            "temperature": "Mean (Average of daily values)",
            "precip_solar": "Sum (Total accumulation over the week)"
        }
    }

def main():
    for state, points in STATE_COORDS.items():
        # 1. Get Data
        data = get_weekly_state_data(state, points)
        
        if data:
            # 2. Save Data File
            data_file = f"{state}.json"
            with open(data_file, "w") as f:
                json.dump(data, f, indent=4)
                
            # 3. Save Metadata File
            meta_data = generate_metadata(state, data, points)
            meta_file = f"{state}_metadata.json"
            with open(meta_file, "w") as f:
                json.dump(meta_data, f, indent=4)
                
            print(f"Done! Saved {data_file} and {meta_file}")
        
        time.sleep(5)

if __name__ == "__main__":
    main()