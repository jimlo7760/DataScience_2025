# Multi-City Population, Energy, and Education Dataset

A comprehensive dataset analyzing demographic trends, educational attainment, and energy consumption patterns across three major U.S. cities: New York City, Los Angeles, and Seattle (2014-2023).

## 📊 Dataset Overview

This project combines three key metrics for urban analysis:
- **Population**: Annual city population estimates
- **Energy Consumption**: State-level electricity sales (million kWh)
- **Education Level**: Average educational attainment (years of schooling)

## 🗂️ Files Included

### Data Collection Scripts
- `get_energy.py` - Fetches electricity consumption data from EIA API
- `get_education.py` - Fetches educational attainment data from Census ACS API
- `get_population.py` - Fetches population data from Census ACS API (not provided)

### Analysis & Visualization
- `generate_graph.py` - Creates scatter plots showing Energy vs Population
- `generate_education_graph.py` - Creates scatter plots showing Energy vs Education Level

### Data Files
**Population Data:**
- `nyc_population_2014_2023.csv`
- `la_population_2014_2023.csv`
- `seattle_population_2014_2023.csv`

**Energy Data:**
- `NYC_energy_2014_2023.csv`
- `LA_energy_2014_2023.csv`
- `Seattle_energy_2014_2023.csv`

**Education Data:**
- `NYC_education_2014_2023.csv`
- `LA_education_2014_2023.csv`
- `Seattle_education_2014_2023.csv`

### Metadata
- `metadata.json` - Comprehensive dataset documentation following schema.org standards

## 🚀 Getting Started

### Prerequisites

Install required Python packages:
```bash
pip install requests pandas matplotlib scipy numpy
```

### API Keys Required

1. **Census API Key** (for population and education data)
   - Register at: https://api.census.gov/data/key_signup.html
   - Add to scripts: `API_KEY = "your_census_api_key"`

2. **EIA API Key** (for energy data)
   - Register at: https://www.eia.gov/opendata/register.php
   - Add to `get_energy.py`: `API_KEY = "your_eia_api_key"`

## 📥 Fetching Data

### Get Energy Data
```bash
python get_energy.py
```
Fetches state-level electricity sales for NY, CA, and WA (2014-2023).

### Get Education Data
```bash
python get_education.py
```
Fetches educational attainment data from ACS 5-Year Estimates for NYC, LA, and Seattle (2014-2023).

**Note:** 2024 ACS data not yet available (released ~1 year after survey period).

## 📈 Generating Visualizations

### Energy vs Population Analysis
```bash
python generate_graph.py
```
Creates scatter plots with trend lines showing the relationship between population and energy consumption.

**Output:**
- `la_energy_population.png`
- `nyc_energy_population.png`
- `seattle_energy_population.png`

### Energy vs Education Analysis
```bash
python generate_education_graph.py
```
Creates scatter plots with trend lines showing the relationship between education level and energy consumption.

**Output:**
- `la_energy_education.png`
- `nyc_energy_education.png`
- `seattle_energy_education.png`

## 📋 Data Structure

### Population Files
| Column | Type | Description |
|--------|------|-------------|
| Year | integer | Calendar year (2014-2023) |
| Population | integer | City population estimate |
| City | string | City name |

### Energy Files
| Column | Type | Description |
|--------|------|-------------|
| Year | integer | Calendar year (2014-2023) |
| Consumption | float | Electricity sales (million kWh) |
| City | string | City name |

### Education Files
| Column | Type | Description |
|--------|------|-------------|
| Year | integer | Calendar year (2014-2023) |
| Education level | float | Average years of schooling (12-21) |
| City | string | City name |

**Education Level Scale:**
- 12 years = High school graduate
- 14 years = Associate's degree
- 16 years = Bachelor's degree
- 18 years = Master's degree
- 21 years = Doctorate degree

## 📚 Data Sources

- **Population**: U.S. Census Bureau, American Community Survey (ACS) 1-Year Estimates
- **Education**: U.S. Census Bureau, ACS 5-Year Estimates, Table B15003
- **Energy**: U.S. Energy Information Administration (EIA), State Energy Data System

## ⚠️ Limitations

- Energy data represents **state-level** totals, not city-specific consumption
- ACS 5-Year Estimates have ~1 year release lag
- 2024 data not available until December 2025
- Education level uses simplified weighting scheme (12-21 years)
- Electricity sales don't include other energy sources (natural gas, heating oil, etc.)

## 📄 License

Public Domain - U.S. Government Work

## 📧 Contact

**Jim Lu**  
Email: jimlo7760@gmail.com  
Project: Energy Consumption Analysis Project

## 🔗 Useful Links

- [Census API Documentation](https://www.census.gov/data/developers/data-sets.html)
- [EIA API Documentation](https://www.eia.gov/opendata/)
- [ACS Data Releases](https://www.census.gov/programs-surveys/acs/news/data-releases.html)
- [EIA Electricity Data Browser](https://www.eia.gov/electricity/data/browser/)

---

**Last Updated:** October 5, 2025  
**Data Coverage:** 2014-2023  
**Python Version:** 3.10+