# Multi-City Population, Energy, and Education Dataset

A comprehensive dataset analyzing demographic trends, educational attainment, and energy consumption patterns across three major U.S. cities: New York City, Los Angeles, and Seattle (2014-2024).

## 📊 Dataset Overview

This project combines three key metrics for urban analysis:
- **Population**: Annual city population estimates
- **Energy Consumption**: State-level electricity sales (million kWh)
- **Education Level**: Average educational attainment (years of schooling)

## 🗂️ Files Included

### Data Collection Scripts
- `get_energy.py` - Fetches electricity consumption data from EIA API
- `get_education.py` - Fetches educational attainment data from Census ACS API
- `get_population.py` - Fetches population data from Census ACS API

### Analysis & Visualization
- `generate_graph.py` - Creates individual scatter plots showing Energy vs Population for each city
- `generate_education_graph.py` - Creates scatter plots showing Energy vs Education Level

### Data Files
**Population Data:**
- `nyc_population_2014_2024.csv`
- `la_population_2014_2024.csv`
- `seattle_population_2014_2024.csv`

**Energy Data:**
- `NYC_energy_2014_2024.csv`
- `LA_energy_2014_2024.csv`
- `Seattle_energy_2014_2024.csv`

**Education Data:**
- `NYC_education_2014_2024.csv`
- `LA_education_2014_2024.csv`
- `Seattle_education_2014_2024.csv`

### Metadata
- `metadata.json` - Comprehensive dataset documentation following schema.org standards

## 🚀 Getting Started

### Prerequisites

Install required Python packages:
```bash
pip install pandas matplotlib numpy scipy requests
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
Fetches state-level electricity sales for NY, CA, and WA (2014-2024).

### Get Education Data
```bash
python get_education.py
```
Fetches educational attainment data from ACS 5-Year Estimates for NYC, LA, and Seattle (2014-2024).

### Get Population Data
```bash
python get_population.py
```
Fetches population data from Census ACS API for NYC, LA, and Seattle (2014-2024).

## 📈 Generating Visualizations

### Energy vs Population Analysis
```bash
python generate_graph.py
```
Creates individual scatter plots with trend lines showing the relationship between population and energy consumption for each city.

**Features:**
- Scatter plot with year labels on each data point
- Red dashed trend line showing correlation
- R² value indicating correlation strength
- Statistics box with slope and correlation coefficient
- Professional styling with grid layout

**Output Files:**
- `la_energy_population.png` - Los Angeles analysis
- `nyc_energy_population.png` - New York City analysis
- `seattle_energy_population.png` - Seattle analysis

**Console Output:**
- Correlation analysis with R² values
- Slope (GWh per person)
- P-value for statistical significance
- Correlation strength assessment (Strong/Moderate/Weak)

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
| Year | integer | Calendar year (2014-2024) |
| Population | integer | City population estimate |
| City | string | City name |

### Energy Files
| Column | Type | Description |
|--------|------|-------------|
| Year | integer | Calendar year (2014-2024) |
| Consumption | float | Electricity sales (million kWh) |
| City | string | City/State name |

### Education Files
| Column | Type | Description |
|--------|------|-------------|
| Year | integer | Calendar year (2014-2024) |
| Education level | float | Average years of schooling (12-21) |
| City | string | City name |

**Education Level Scale:**
- 12 years = High school graduate
- 14 years = Associate's degree
- 16 years = Bachelor's degree
- 18 years = Master's degree
- 21 years = Doctorate degree

## 📊 Understanding the Graphs

### Scatter Plot Interpretation
Each graph shows:
- **Data Points**: Individual years (labeled with year number)
- **Trend Line**: Red dashed line showing overall relationship
- **R² Value**: Measures correlation strength (0-1 scale)
  - R² > 0.7: Strong correlation
  - R² 0.4-0.7: Moderate correlation
  - R² < 0.4: Weak correlation
- **Slope**: Change in energy consumption per unit change in population/education

### What to Look For
- **Positive slope**: Energy increases with population/education
- **Steep slope**: Large energy changes for small population/education changes
- **Points above trend line**: Years with higher-than-expected energy use
- **Points below trend line**: Years with lower-than-expected energy use

## 📚 Data Sources

- **Population**: U.S. Census Bureau, American Community Survey (ACS) 1-Year Estimates
- **Education**: U.S. Census Bureau, ACS 5-Year Estimates, Table B15003
- **Energy**: U.S. Energy Information Administration (EIA), State Energy Data System

## ⚠️ Limitations

- Energy data represents **state-level** totals, not city-specific consumption
- ACS 5-Year Estimates have ~1 year release lag
- 2025 data not yet available
- Education level uses simplified weighting scheme (12-21 years)
- Electricity sales don't include other energy sources (natural gas, heating oil, etc.)
- Correlation does not imply causation
    
## 🔧 Troubleshooting

**Import Errors:**
```bash
pip install --upgrade pandas matplotlib numpy scipy
```

**File Not Found:**
- Ensure all CSV files are in the same directory as the script
- Check file names match exactly (case-sensitive)

**Empty Plots:**
- Verify CSV files contain data for the expected years
- Check that Year, Population, and Consumption columns exist

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

**Last Updated:** October 6, 2025  
**Data Coverage:** 2014-2024  
**Python Version:** 3.7+
