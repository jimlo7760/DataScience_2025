"""Resource Endowment Data Pipeline.

This module pulls monthly electric generation by fuel type for selected US
states using the EIA v2 API. The state-level totals are propagated to the
associated cities defined in :data:`CITY_STATE` so that each city record embeds
its state's monthly resource mix.

Outputs
-------
``data/curated/resource_endowment_by_state.json``
    Nested JSON document containing total generation by fuel and the full
    monthly resource report for each tracked state.
``data/curated/resource_endowment_by_state.parquet``
    Flat table of state-level totals (one row per state).
``data/curated/resource_endowment_state_monthly.parquet``
    Flat table of monthly generation by state.
``data/curated/resource_endowment_by_city.json``
    Nested JSON with the monthly resource report replicated for each city via
    its state mapping.
``data/curated/resource_endowment_by_city.parquet``
    Flat table of city-level totals (one row per city).
``data/curated/resource_endowment_city_monthly.parquet``
    Flat table of monthly generation by city (state mix duplicated per city).

Set the ``EIA_API_KEY`` environment variable before running the module.

Example
-------
>>> export EIA_API_KEY=...
>>> python resource_endowment_pipeline.py
"""

from __future__ import annotations

import json
import os
import pathlib
from typing import Dict, Iterable, List

import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = pathlib.Path("data")
CURATED_DIR = DATA_DIR / "curated"
CURATED_DIR.mkdir(parents=True, exist_ok=True)

EIA_API_KEY = os.getenv("EIA_API_KEY")
EIA_BASE = "https://api.eia.gov/v2/"
EIA_ROUTE = "electricity/electric-power-operational-data/data/"

CITY_STATE = {
    "NYC": "NY",
    "LosAngeles": "CA",
    "Seattle": "WA",
}

RESOURCE_FUELS = {
    "hydro": "WAT",
    "solar": "SUN",
    "wind": "WND",
    "coal": "COL",
    "natural_gas": "NG",
}

START_MONTH = "2014-01"
END_MONTH = "2024-12"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=1, max=10))
def eia_query(route: str, params: Dict[str, object]) -> pd.DataFrame:
    """Execute a paginated query against the EIA v2 API.

    Parameters
    ----------
    route:
        Path segment after ``https://api.eia.gov/v2/`` identifying the dataset.
    params:
        Query string parameters (without the API key). Pagination is handled via
        the ``offset`` parameter which is automatically injected.
    """

    if not EIA_API_KEY:
        raise RuntimeError("Set the EIA_API_KEY environment variable.")

    url = f"{EIA_BASE}{route}"
    frames: List[pd.DataFrame] = []
    offset = 0
    while True:
        paged_params = {**params, "api_key": EIA_API_KEY, "offset": offset}
        response = requests.get(url, params=paged_params, timeout=60)
        response.raise_for_status()
        payload = response.json()
        data = payload.get("response", {}).get("data", [])
        if not data:
            break
        frames.append(pd.DataFrame(data))
        total = payload.get("response", {}).get("total", 0)
        try:
            total = int(total)
        except Exception:
            total = 0
        offset += len(data)
        if offset >= total:
            break
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def normalize_generation_frame(df: pd.DataFrame, fuels: Iterable[str]) -> pd.DataFrame:
    """Filter and clean a raw EIA response DataFrame."""

    if df.empty:
        return df
    df = df[df["fueltypeid"].isin(fuels)].copy()
    if df.empty:
        return df
    df["generation"] = pd.to_numeric(df["generation"], errors="coerce").fillna(0.0)
    df["period"] = pd.to_datetime(df["period"])
    return df


def fetch_state_resource_endowment(state: str) -> Dict[str, object]:
    """Fetch monthly generation by fuel for a single state."""

    params = {
        "facets[location][]": state,
        "facets[fueltypeid][]": list(RESOURCE_FUELS.values()),
        "frequency": "monthly",
        "start": START_MONTH,
        "end": END_MONTH,
        "data[]": ["generation"],
        "sort[0][column]": "period",
        "sort[0][direction]": "asc",
        "length": 5000,
    }
    raw_df = eia_query(EIA_ROUTE, params=params)
    clean_df = normalize_generation_frame(raw_df, RESOURCE_FUELS.values())
    if clean_df.empty:
        raise RuntimeError(f"EIA API returned no data for state {state}. Verify access and route.")

    pivot = (
        clean_df
        .pivot_table(index="period", columns="fueltypeid", values="generation", aggfunc="sum", fill_value=0.0)
        .sort_index()
    )
    # Map fuel codes to friendly names and ensure all fuels present.
    rename_map = {v: k for k, v in RESOURCE_FUELS.items()}
    pivot = pivot.rename(columns=rename_map)
    for friendly_name in RESOURCE_FUELS:
        if friendly_name not in pivot.columns:
            pivot[friendly_name] = 0.0
    pivot = pivot[list(RESOURCE_FUELS.keys())]

    monthly_records = [
        {
            "period": period.strftime("%Y-%m"),
            **{fuel: float(value) for fuel, value in row.items()},
        }
        for period, row in pivot.iterrows()
    ]
    totals = {fuel: float(pivot[fuel].sum()) for fuel in RESOURCE_FUELS}
    return {
        "state": state,
        "resource_totals_mwh": totals,
        "monthly_resource_report": monthly_records,
    }


def build_state_payload() -> Dict[str, Dict[str, object]]:
    states = sorted(set(CITY_STATE.values()))
    payload = {}
    for state in states:
        print(f"[resource] fetching state data for {state} ...")
        payload[state] = fetch_state_resource_endowment(state)
    return payload


def save_state_resource_endowment(state_payload: Dict[str, Dict[str, object]]) -> None:
    records = list(state_payload.values())
    (CURATED_DIR / "resource_endowment_by_state.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2)
    )

    totals_records = [
        {"state": state, **info["resource_totals_mwh"]}
        for state, info in state_payload.items()
    ]
    totals_df = pd.DataFrame(totals_records)
    totals_df.to_parquet(CURATED_DIR / "resource_endowment_by_state.parquet", index=False)

    monthly_records = [
        {"state": state, **monthly}
        for state, info in state_payload.items()
        for monthly in info["monthly_resource_report"]
    ]
    monthly_df = pd.DataFrame(monthly_records)
    monthly_df.to_parquet(CURATED_DIR / "resource_endowment_state_monthly.parquet", index=False)
    print("State-level resource endowment saved.")


def save_city_resource_endowment(state_payload: Dict[str, Dict[str, object]]) -> None:
    city_records = []
    for city, state in CITY_STATE.items():
        info = state_payload[state]
        city_records.append(
            {
                "city": city,
                "state": state,
                "resource_totals_mwh": info["resource_totals_mwh"],
                "monthly_resource_report": info["monthly_resource_report"],
            }
        )

    (CURATED_DIR / "resource_endowment_by_city.json").write_text(
        json.dumps(city_records, ensure_ascii=False, indent=2)
    )

    totals_df = pd.DataFrame(
        [
            {"city": rec["city"], "state": rec["state"], **rec["resource_totals_mwh"]}
            for rec in city_records
        ]
    )
    totals_df.to_parquet(CURATED_DIR / "resource_endowment_by_city.parquet", index=False)

    monthly_records = [
        {
            "city": rec["city"],
            "state": rec["state"],
            **monthly,
        }
        for rec in city_records
        for monthly in rec["monthly_resource_report"]
    ]
    monthly_df = pd.DataFrame(monthly_records)
    monthly_df.to_parquet(CURATED_DIR / "resource_endowment_city_monthly.parquet", index=False)
    print("City-level resource endowment saved.")


def main() -> None:
    state_payload = build_state_payload()
    save_state_resource_endowment(state_payload)
    save_city_resource_endowment(state_payload)


if __name__ == "__main__":
    main()
