# Energy & Climate Data Pipeline

This repository provides a compact Python script that downloads historical daily
weather data for New York City, Los Angeles, and Seattle from 2014 through 2024
using the Open-Meteo archive API. The script stores per-city daily Parquet files
under `data/raw` and a combined monthly aggregation under `data/curated`. It can
optionally fetch monthly state-level electricity generation from the US Energy
Information Administration (EIA) v2 API when an API key and dataset route are
provided.

## Prerequisites

* Python 3.10+
* Access to the public Open-Meteo archive API (no key required)
* (Optional) An [EIA API key](https://www.eia.gov/opendata/register.php) and
  the appropriate v2 dataset route for state-level generation by fuel

Install dependencies into your environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running the pipeline

```bash
python energy_climate_pipeline.py
```

The script will create the `data/raw` and `data/curated` folders (if missing)
and write Parquet outputs for the weather data. Network access is required to
call the Open-Meteo API.

### Optional EIA step

To enable the EIA extraction, set the following environment variables before
running the script:

```bash
export EIA_API_KEY="your_api_key"
export EIA_ROUTE="electricity/..."
```

Use the [EIA API browser](https://www.eia.gov/opendata/browser/) to identify the
correct dataset route and facets for the energy mix you want to retrieve. When
available, a Parquet file is saved to `data/raw/eia_generation_state_monthly.parquet`.

## Outputs



* `data/raw/<city>/weather_daily_<city>.parquet` — Daily weather data for each city (Celsius and Fahrenheit), including heating/cooling degree days.
* `data/raw/<city>/weather_daily_<city>.json` — Same as above, in JSON format.
* `data/curated/<city>/weather_monthly_<city>.parquet` — Monthly aggregated weather data for each city.
* `data/curated/<city>/weather_monthly_<city>.json` — Same as above, in JSON format.
* `data/curated/weather_monthly_by_city.parquet` — Monthly aggregation for all cities.
* `data/curated/weather_monthly_by_city.json` — Same as above, in JSON format.
* `data/raw/eia_generation_state_monthly.parquet` — (Optional) State-level monthly electricity generation from EIA API.

* `data/curated/resource_endowment_by_state.json` / `.parquet` — State-level energy resource endowment (EIA API, includes hydro, solar, wind, coal, natural gas, monthly and annual totals).
* `data/curated/resource_endowment_by_city.json` / `.parquet` — City-level energy resource endowment (mapped from state data, same structure).


## Metadata & Data Dictionary

### Resource Endowment Data (by State/City)



**Dataset Metadata:**
- Dataset name: Resource Endowment by State/City
- Time coverage: 2014-01 ~ 2024-12 (monthly, annual totals)
- Spatial coverage: California (CA), New York (NY), Washington (WA) and their major cities (LosAngeles, NYC, Seattle)
- Source: EIA v2 API [Electric Power Operational Data](https://www.eia.gov/opendata/browser/electricity/electric-power-operational-data)
- Collection method: API automated fetch, city data mapped from state data
- Last updated: Can be updated daily (recommended to refresh regularly)
- Units: All energy quantities are in megawatt-hours (MWh)
- Data structure: Each record contains annual totals (`resource_totals_mwh`) and monthly breakdowns (`monthly_resource_report`, period as YYYY-MM)
- File format: JSON (records), Parquet
- Missing value handling: 0 or null if no data


**File Description:**
Each record is for a state (or city), including annual totals and monthly breakdowns of major energy types (unit: MWh).


**Field Description:**

| Field                   | Type        | Description (EN)                                 |
|-------------------------|-------------|--------------------------------------------------|
| state                   | str         | State Abbreviation                               |
| city                    | str         | City Name (city table only)                      |
| resource_totals_mwh     | dict        | Annual total generation by resource type (MWh)    |
| └ hydro                 | float       | Hydropower (MWh)                                 |
| └ solar                 | float       | Solar (MWh)                                      |
| └ wind                  | float       | Wind (MWh)                                       |
| └ coal                  | float       | Coal (MWh)                                       |
| └ natural_gas           | float       | Natural Gas (MWh)                                |
| monthly_resource_report | list[dict]  | Monthly breakdown                                |
| └ period                | str         | Period (YYYY-MM)                                 |
| └ hydro                 | float       | Hydropower (MWh)                                 |
| └ solar                 | float       | Solar (MWh)                                      |
| └ wind                  | float       | Wind (MWh)                                       |
| └ coal                  | float       | Coal (MWh)                                       |
| └ natural_gas           | float       | Natural Gas (MWh)                                |


**Data Sources:**
- EIA v2 API: [Electric Power Operational Data](https://www.eia.gov/opendata/browser/electricity/electric-power-operational-data)
- City data mapped from state data (e.g., NYC→NY, LosAngeles→CA, Seattle→WA)


**Data Science Best Practices:**
- All data includes explicit units (MWh) and retains original API fields for traceability.
- JSON uses records format, Parquet structure is identical, suitable for big data analysis and visualization.
- Field naming is standardized for automation.
- Recommend integrating with weather, economic, and other multi-source data for comprehensive analysis.


---
### Metadata from Original Sources


#### Open-Meteo API (https://open-meteo.com/)
- Source fields: `temperature_2m_max`, `temperature_2m_min`, `temperature_2m_mean`, `precipitation_sum`, `rain_sum`, `snowfall_sum`, `weathercode`, `time`, `latitude`, `longitude`, `timezone`
- All weather data is retrieved from the Open-Meteo archive API, which provides historical daily weather metrics for any location worldwide. See [Open-Meteo API docs](https://open-meteo.com/en/docs#archive) for details.
  - `rain_sum`: Total rainfall (mm)
  - `snowfall_sum`: Total snowfall (mm)
  - `weathercode`: Weather condition code (see Open-Meteo docs)
#### EIA v2 API (https://www.eia.gov/opendata/)
- Source fields: `stateId`, `period`, `value`, `fueltype`, etc. (see EIA documentation for details)
- State-level electricity generation data is optionally retrieved from the U.S. Energy Information Administration (EIA) v2 API.

---

### Daily Weather Data (`weather_daily_<city>.*`)
Each record contains the following fields:

| Field         | Type      | Description (EN)                |
| ------------- | --------- | ------------------------------- |
| city          | str       | City name                       |
| date          | str/date  | Date (YYYY-MM-DD)               |
| tmax_c        | float     | Max temperature (°C)            |
| tmin_c        | float     | Min temperature (°C)            |
| tmean_c       | float     | Mean temperature (°C)           |
| tmax_f        | float     | Max temperature (°F)            |
| tmin_f        | float     | Min temperature (°F)            |
| tmean_f       | float     | Mean temperature (°F)           |
| precip_mm     | float     | Precipitation (mm)              |
| rain_mm       | float     | Rainfall (mm)                   |
| snow_mm       | float     | Snowfall (mm)                   |
| weather_code  | int/str   | Weather code (Open-Meteo)       |
| hdd_base65F   | float     | Heating degree days (base 65°F) |
| cdd_base65F   | float     | Cooling degree days (base 65°F) |


### Monthly Weather Data (`weather_monthly_<city>.*`)
Each record contains the following fields:

| Field           | Type    | Description (EN)                |
| --------------- | ------- | ------------------------------- |
| city            | str     | City name                       |
| year            | int     | Year                            |
| month           | int     | Month (1-12)                    |
| tmean_c_avg     | float   | Mean temperature (°C, monthly)  |
| tmax_c_avg      | float   | Max temperature (°C, monthly)   |
| tmin_c_avg      | float   | Min temperature (°C, monthly)   |
| tmean_f_avg     | float   | Mean temperature (°F, monthly)  |
| precip_mm_sum   | float   | Total precipitation (mm, month) |
| rain_mm_sum     | float   | Total rainfall (mm, month)      |
| snow_mm_sum     | float   | Total snowfall (mm, month)      |
| hdd_base65F     | float   | Heating degree days (monthly)   |
| cdd_base65F     | float   | Cooling degree days (monthly)   |
| weather_code    | int/str | Weather code (Open-Meteo, first of month) |
| year_month      | str/date| Year and month (YYYY-MM-01)     |


---
### Other Notes
- All JSON files use records orientation, with ISO date fields.
- Parquet files have the same structure as JSON.
- All numeric fields are null if data is missing.

## Notes

* The Open-Meteo API may throttle repeated calls. Retries with exponential backoff
  are enabled for resilience.
* Modify the `CITIES`, `START_DATE`, or `END_DATE` constants in
  `energy_climate_pipeline.py` to tailor the dataset to other regions or periods.
