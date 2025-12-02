import os
import json
import time
import requests
from statistics import mean

API_KEY = "_"
BASE_URL = "http://quickstats.nass.usda.gov/api/api_GET/"


# ======================================================
# Utility: parse period_desc into month/season
# ======================================================
MONTH_MAP = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4,
    "MAY": 5, "JUN": 6, "JUL": 7, "AUG": 8,
    "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12
}

SEASON_KEYWORDS = {
    "SPRING": "SPRING",
    "SUMMER": "SUMMER",
    "WINTER": "WINTER",
    "FALL": "FALL",
    "AUTUMN": "AUTUMN"
}


def parse_period(period_desc):
    """
    Returns: (month, season)
    """
    if not period_desc:
        return None, None

    # Month (APR, JUN, etc)
    parts = period_desc.split()
    for p in parts:
        if p in MONTH_MAP:
            return MONTH_MAP[p], None

    # Season (WINTER, SPRING, etc)
    for k in SEASON_KEYWORDS:
        if k in period_desc.upper():
            return None, SEASON_KEYWORDS[k]

    return None, None


# ======================================================
# API caller
# ======================================================
def nass_query(params):
    params = params.copy()
    params["key"] = API_KEY
    params["format"] = "JSON"

    t0 = time.time()
    r = requests.get(BASE_URL, params=params)
    t1 = time.time()

    response_time = round(t1 - t0, 3)

    try:
        data = r.json()
    except:
        return None, response_time

    return data, response_time


# ======================================================
# Metadata curation with season/month awareness
# ======================================================
def curate_metadata(raw_json, response_time, crop, state, params):
    if "data" not in raw_json:
        return {
            "status": "NO DATA",
            "crop": crop,
            "state": state,
            "response_time_sec": response_time,
            "params": params
        }

    records = raw_json["data"]

    year_period_pairs = []
    all_years = []
    all_months = []
    all_seasons = []
    all_marketing = []
    values = []

    for item in records:
        year = item.get("year")
        period = item.get("period_desc", "")
        marketing_year = item.get("reference_period_desc", "")

        # parse month/season
        month, season = parse_period(period)

        if year:
            all_years.append(int(year))
        if month:
            all_months.append(month)
        if season:
            all_seasons.append(season)
        if marketing_year:
            all_marketing.append(marketing_year)

        # actual numeric yield
        v = item.get("Value")
        if v and v not in ["(D)", "(Z)", "(NA)"]:
            try:
                values.append(float(v.replace(",", "")))
            except:
                pass

        # track multiperiod grouping
        year_period_pairs.append((year, period))

    years_available = sorted(set(all_years))
    pairs_unique = sorted(set(year_period_pairs))

    metadata = {
        "status": "OK",
        "crop": crop,
        "state": state,
        "n_records": len(records),

        "years_available": years_available,
        "year_period_combinations": pairs_unique,

        "months_detected": sorted(set(all_months)),
        "seasons_detected": sorted(set(all_seasons)),
        "marketing_year_descriptors": sorted(set(all_marketing)),

        "first_year": min(all_years) if all_years else None,
        "last_year": max(all_years) if all_years else None,

        "value_stats": {
            "count_valid": len(values),
            "min": min(values) if values else None,
            "max": max(values) if values else None,
            "mean": mean(values) if values else None,
        },

        "response_time_sec": response_time,
        "parameter_used": params,

        # data example
        "example_record_raw": records[0] if len(records) > 0 else None
    }

    return metadata


# ======================================================
# Query definitions (same crops, but now preserving full resolution)
# ======================================================
queries = [
    ("CORN", "IA"),
    ("SOYBEANS", "IL"),
    ("WHEAT", "KS")
]

base_params = {
    "source_desc": "SURVEY",
    "agg_level_desc": "STATE",
    "statisticcat_desc": "YIELD",
    "year__GE": 2000
}


# ======================================================
# Run
# ======================================================
os.makedirs("json_raw", exist_ok=True)
os.makedirs("metadata", exist_ok=True)
os.makedirs("processed", exist_ok=True)

combined = {}

for crop, state in queries:
    params = base_params.copy()
    params["commodity_desc"] = crop
    params["state_alpha"] = state

    raw_json, rt = nass_query(params)

    raw_path = f"json_raw/{state}_{crop}_raw.json"
    with open(raw_path, "w") as f:
        json.dump(raw_json, f, indent=2)

    metadata = curate_metadata(raw_json, rt, crop, state, params)

    meta_path = f"metadata/{state}_{crop}_meta.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    combined[f"{state}_{crop}"] = raw_json

with open("processed/combined_clean.json", "w") as f:
    json.dump(combined, f, indent=2)

print("DONE — multi-period JSON + metadata saved.")
