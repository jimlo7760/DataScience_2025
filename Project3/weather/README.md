# Weather Data (Iowa Example)

This directory contains scripts and data for weather collection and aggregation, suitable for crop modeling.

## Example Dataset: Iowa Weekly Crop Weather

- **Description:** Weekly aggregated weather data suitable for crop growth modeling (GDD, Photosynthesis, Water).
- **Source:** Open-Meteo Historical Weather API (ERA5 Reanalysis)
- **Temporal Coverage:**
  - Start: 2000-01-03
  - End: 2025-12-08
  - Frequency: Weekly (Aggregated Mondays)
- **Spatial Coverage:**
  - Method: Average of 3 lat/lon points (North, Center, South)
  - Coordinates used:
    - (43.0, -93.6) — North
    - (41.6, -93.6) — Center
    - (40.7, -93.6) — South

## Data Columns
- `date`: Start date of the week (YYYY-MM-DD)
- `tmax`: Average Maximum Air Temperature (Fahrenheit)
- `tmin`: Average Minimum Air Temperature (Fahrenheit)
- `precip`: Total Accumulated Precipitation (Inches)
- `solar`: Total Accumulated Solar Radiation (MegaJoules per square meter)

## Aggregation Logic
- **Temperature:** Mean (Average of daily values)
- **Precipitation & Solar:** Sum (Total accumulation over the week)

## Usage
Typical usage:
```bash
python3 main.py
```

---
For more details, see the metadata file (`iowa_metadata.json`).