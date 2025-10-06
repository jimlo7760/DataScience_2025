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


* `data/raw/<city>/weather_daily_<city>.parquet` — 每个城市的每日天气数据（Celsius和Fahrenheit），包含冷暖负荷度日。
* `data/raw/<city>/weather_daily_<city>.json` — 同上，JSON格式。
* `data/curated/<city>/weather_monthly_<city>.parquet` — 每个城市的月度聚合天气数据。
* `data/curated/<city>/weather_monthly_<city>.json` — 同上，JSON格式。
* `data/curated/weather_monthly_by_city.parquet` — 所有城市的月度聚合汇总。
* `data/curated/weather_monthly_by_city.json` — 同上，JSON格式。
* `data/raw/eia_generation_state_monthly.parquet` — (可选) EIA API的州级月度发电结构。

* `data/curated/resource_endowment_by_state.json` / `.parquet` — 各州能源资源禀赋（EIA API，含水电、太阳能、风能、煤炭、天然气，月度与年度总量）。
* `data/curated/resource_endowment_by_city.json` / `.parquet` — 各城市能源资源禀赋（由州数据映射，结构同上）。


## Metadata & Data Dictionary

### Resource Endowment Data (by State/City)


**数据集元数据（Metadata）**：
- 数据集名称：Resource Endowment by State/City
- 时间范围（Time coverage）：2014-01 ~ 2024-12（每月，年度总量）
- 空间范围（Spatial coverage）：美国加州（CA）、纽约州（NY）、华盛顿州（WA）及其主要城市（LosAngeles, NYC, Seattle）
- 数据来源（Source）：EIA v2 API [Electric Power Operational Data](https://www.eia.gov/opendata/browser/electricity/electric-power-operational-data)
- 采集方式：API自动抓取，城市数据由州数据映射
- 更新时间（Last updated）：每日可自动更新（建议定期刷新）
- 单位（Units）：所有能源量均为兆瓦时（MWh）
- 数据结构：每条记录包含年度总量（resource_totals_mwh）和月度分项（monthly_resource_report，period为YYYY-MM）
- 文件格式：JSON（records），Parquet
- 缺失值处理：无数据时为0或null

**文件说明**：
每条记录为一个州（或城市），包含年度总量和月度分项的主要能源类型发电量（单位：兆瓦时 MWh）。

**字段说明**：

| 字段/Field              | 类型/Type   | 说明/Description (EN & 中文) |
|-------------------------|-------------|------------------------------|
| state                   | str         | 州名/State Abbreviation      |
| city                    | str         | 城市名/City Name (仅city表)   |
| resource_totals_mwh     | dict        | 各能源类型年度总发电量（MWh）/ Annual total generation by resource type |
| └ hydro                 | float       | 水电/ Hydropower (MWh)       |
| └ solar                 | float       | 太阳能/ Solar (MWh)          |
| └ wind                  | float       | 风能/ Wind (MWh)             |
| └ coal                  | float       | 煤炭/ Coal (MWh)             |
| └ natural_gas           | float       | 天然气/ Natural Gas (MWh)    |
| monthly_resource_report | list[dict]  | 月度分项/ Monthly breakdown  |
| └ period                | str         | 月份/Period (YYYY-MM)        |
| └ hydro                 | float       | 水电/ Hydropower (MWh)       |
| └ solar                 | float       | 太阳能/ Solar (MWh)          |
| └ wind                  | float       | 风能/ Wind (MWh)             |
| └ coal                  | float       | 煤炭/ Coal (MWh)             |
| └ natural_gas           | float       | 天然气/ Natural Gas (MWh)    |

**数据来源**：
- EIA v2 API: [Electric Power Operational Data](https://www.eia.gov/opendata/browser/electricity/electric-power-operational-data)
- 城市数据由州数据映射（如NYC→NY，LosAngeles→CA，Seattle→WA）

**数据科学最佳实践说明**：
- 所有数据均包含明确的单位（MWh），并保留原始API字段，便于追溯。
- JSON为records格式，Parquet结构一致，适合大数据分析与可视化。
- 字段命名规范，便于自动化处理。
- 建议结合气象数据、经济数据等多源信息进行综合分析。


---
### Metadata from Original Sources / 原始数据源元数据

#### Open-Meteo API (https://open-meteo.com/)
- Source fields: `temperature_2m_max`, `temperature_2m_min`, `temperature_2m_mean`, `precipitation_sum`, `rain_sum`, `snowfall_sum`, `weathercode`, `time`, `latitude`, `longitude`, `timezone`
- All weather data is retrieved from the Open-Meteo archive API, which provides historical daily weather metrics for any location worldwide. See [Open-Meteo API docs](https://open-meteo.com/en/docs#archive) for details.
  - `rain_sum`: Total rainfall (mm) / 总降雨量 (毫米)
  - `snowfall_sum`: Total snowfall (mm) / 总降雪量 (毫米)
  - `weathercode`: Weather condition code (see Open-Meteo docs) / 天气状况代码（详见Open-Meteo文档）
#### EIA v2 API (https://www.eia.gov/opendata/)
- Source fields: `stateId`, `period`, `value`, `fueltype`, etc. (see EIA documentation for details)
- State-level electricity generation data is optionally retrieved from the U.S. Energy Information Administration (EIA) v2 API.

---
### Daily Weather Data (`weather_daily_<city>.*`)
Each record contains the following fields (English/Chinese):

| Field         | Type      | Description (EN)                | 说明 (中文)                |
| ------------- | --------- | ------------------------------- | -------------------------- |
| city          | str       | City name                       | 城市名                     |
| date          | str/date  | Date (YYYY-MM-DD)               | 日期 (YYYY-MM-DD)          |
| tmax_c        | float     | Max temperature (°C)            | 最高气温 (°C)              |
| tmin_c        | float     | Min temperature (°C)            | 最低气温 (°C)              |
| tmean_c       | float     | Mean temperature (°C)           | 平均气温 (°C)              |
| tmax_f        | float     | Max temperature (°F)            | 最高气温 (°F)              |
| tmin_f        | float     | Min temperature (°F)            | 最低气温 (°F)              |
| tmean_f       | float     | Mean temperature (°F)           | 平均气温 (°F)              |
| precip_mm     | float     | Precipitation (mm)              | 降水量 (mm)                |
| rain_mm       | float     | Rainfall (mm)                   | 降雨量 (mm)                |
| snow_mm       | float     | Snowfall (mm)                   | 降雪量 (mm)                |
| weather_code  | int/str   | Weather code (Open-Meteo)       | 天气代码（Open-Meteo）     |
| hdd_base65F   | float     | Heating degree days (base 65°F) | 以65°F为基准的采暖度日     |
| cdd_base65F   | float     | Cooling degree days (base 65°F) | 以65°F为基准的制冷度日     |

### Monthly Weather Data (`weather_monthly_<city>.*`)
Each record contains the following fields (English/Chinese):

| Field           | Type    | Description (EN)                | 说明 (中文)                |
| --------------- | ------- | ------------------------------- | -------------------------- |
| city            | str     | City name                       | 城市名                     |
| year            | int     | Year                            | 年份                       |
| month           | int     | Month (1-12)                    | 月份 (1-12)                |
| tmean_c_avg     | float   | Mean temperature (°C, monthly)  | 平均气温 (°C, 月均)        |
| tmax_c_avg      | float   | Max temperature (°C, monthly)   | 最高气温 (°C, 月均)        |
| tmin_c_avg      | float   | Min temperature (°C, monthly)   | 最低气温 (°C, 月均)        |
| tmean_f_avg     | float   | Mean temperature (°F, monthly)  | 平均气温 (°F, 月均)        |
| precip_mm_sum   | float   | Total precipitation (mm, month) | 降水量总和 (mm, 月累计)    |
| rain_mm_sum     | float   | Total rainfall (mm, month)      | 降雨量总和 (mm, 月累计)    |
| snow_mm_sum     | float   | Total snowfall (mm, month)      | 降雪量总和 (mm, 月累计)    |
| hdd_base65F     | float   | Heating degree days (monthly)   | 采暖度日 (月累计)          |
| cdd_base65F     | float   | Cooling degree days (monthly)   | 制冷度日 (月累计)          |
| weather_code    | int/str | Weather code (Open-Meteo, first of month) | 天气代码（Open-Meteo，月首日） |
| year_month      | str/date| Year and month (YYYY-MM-01)     | 年月 (YYYY-MM-01)          |

---
### Other Notes / 其他说明
- All JSON files use records orientation, with ISO date fields. / JSON文件均为records格式，日期字段为ISO格式。
- Parquet files have the same structure as JSON. / Parquet文件结构与JSON一致。
- All numeric fields are null if data is missing. / 所有数值字段如无数据则为null。

## Notes

* The Open-Meteo API may throttle repeated calls. Retries with exponential backoff
  are enabled for resilience.
* Modify the `CITIES`, `START_DATE`, or `END_DATE` constants in
  `energy_climate_pipeline.py` to tailor the dataset to other regions or periods.
