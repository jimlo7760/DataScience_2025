# Crops Data

This directory contains scripts and data for crop production and yield analysis.

## Dataset Overview

- **Description:** Historical crop production, yield, and acreage data for three major commodity crops (corn, soybeans, wheat) across three U.S. states.
- **Source:** USDA NASS (National Agricultural Statistics Service)
- **Temporal Coverage:**
  - Available years: Multiple years up to 2025
  - Frequency: Annual
- **Spatial Coverage:**
  - States: Iowa (IA), Illinois (IL), Kansas (KS)
  - Level: State-level aggregation

## Crops Included

1. **IA_CORN** - Iowa Corn (Grain)
   - Unit: BU/ACRE (Bushels per Acre)
   - Measurement: Yield

2. **IL_SOYBEANS** - Illinois Soybeans
   - Unit: BU/ACRE (Bushels per Acre)
   - Measurement: Yield

3. **KS_WHEAT** - Kansas Wheat
   - Unit: BU/ACRE (Bushels per Acre)
   - Measurement: Yield

## Key Data Fields

- `year`: Year of observation (YYYY format)
- `statisticcat_desc`: Type of statistic (e.g., "YIELD", "PRODUCTION", "AREA HARVESTED")
- `Value`: Measured value (numeric)
- `unit_desc`: Unit of measurement (BU/ACRE, etc.)
- `short_desc`: Full description of the measurement (e.g., "CORN, GRAIN - YIELD, MEASURED IN BU / ACRE")
- `commodity_desc`: Crop type (CORN, SOYBEANS, WHEAT)
- `util_practice_desc`: Use/practice category (e.g., GRAIN)
- `class_desc`: Classification (e.g., ALL CLASSES)
- `domain_desc`: Domain category (usually TOTAL)
- `freq_desc`: Frequency (ANNUAL)
- `agg_level_desc`: Aggregation level (STATE)
- `reference_period_desc`: Reference period (YEAR, or YEAR - AUG FORECAST)
- `source_desc`: Data source (SURVEY)

## Data Files

### Raw Data
- `json_raw/IA_CORN_raw.json` - Raw Iowa corn data
- `json_raw/IL_SOYBEANS_raw.json` - Raw Illinois soybeans data
- `json_raw/KS_WHEAT_raw.json` - Raw Kansas wheat data

### Metadata
- `metadata/IA_CORN_meta.json` - Iowa corn metadata
- `metadata/IL_SOYBEANS_meta.json` - Illinois soybeans metadata
- `metadata/KS_WHEAT_meta.json` - Kansas wheat metadata

### Processed Data
- `processed/combined_clean.json` - Cleaned and merged data for all three crops

## Visualization Scripts

### visualize_crops.py
Comprehensive multi-dimensional visualization of crop data:
- Yearly aggregated values for all numeric fields
- Grouped analysis by commodity type, crop class, and production practice
- Multiple subplots for easy comparison

**Usage:**
```bash
python3 visualize_crops.py
```

**Output:** `crop_visualization.png`

### visualize_yield.py
Specialized visualization focused on crop yield trends:
- Separate subplots for each crop (corn, soybeans, wheat)
- Year-over-year yield trends with markers
- Maximum and minimum yield annotations
- Filled area charts for visual emphasis

**Usage:**
```bash
python3 visualize_yield.py
```

**Output:** `crop_yield_visualization.png`

## Sample Statistics

Typical data structure example:
```json
{
  "year": 2025,
  "commodity_desc": "CORN",
  "statisticcat_desc": "YIELD",
  "Value": "216",
  "unit_desc": "BU / ACRE",
  "short_desc": "CORN, GRAIN - YIELD, MEASURED IN BU / ACRE",
  "state_name": "IOWA",
  "location_desc": "IOWA"
}
```

## How to Use

1. **Explore raw data:**
   ```bash
   python3 -c "import json; data = json.load(open('processed/combined_clean.json')); print(json.dumps(data['IA_CORN']['data'][0], indent=2))"
   ```

2. **Generate visualizations:**
   ```bash
   python3 visualize_crops.py    # Multi-dimensional analysis
   python3 visualize_yield.py     # Yield trends focus
   ```

3. **Analyze specific crops:**
   - Modify visualization scripts to filter by `commodity_desc`, `statisticcat_desc`, or `year` as needed

## Notes

- Data may contain forecast periods (indicated by "FORECAST" in `reference_period_desc`)
- All yield values are in BU/ACRE (bushels per acre)
- State-level data represents aggregated county/district information
- Empty fields (e.g., county_name at state level) are expected
