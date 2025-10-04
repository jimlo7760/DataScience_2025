import requests
import pandas as pd
import json
from typing import Optional, Dict, List

EIA_API_BASE_URL = "https://api.eia.gov/v2/"


CITY_SERIES_MAP = {
    "NYC": {
        "name": "New York City",
        "state": "NY",
        "series_id": "ELEC.SALES.NY-ALL.A"  # NY state annual electricity sales
    },
    "LA": {
        "name": "Los Angeles",
        "state": "CA",
        "series_id": "ELEC.SALES.CA-ALL.A"  # CA state annual electricity sales
    },
    "Seattle": {
        "name": "Seattle",
        "state": "WA",
        "series_id": "ELEC.SALES.WA-ALL.A"  # WA state annual electricity sales
    }
}


def get_energy_data(city: str, start_year: int, end_year: int, api_key: str,
                    sectors: Optional[List[str]] = None) -> Optional[pd.DataFrame]:
    """
    Fetch energy use data from EIA API for a specific city across all sectors.

    Args:
        city: City identifier (e.g., 'NYC', 'LA', 'Seattle')
        start_year: Starting year for data retrieval
        end_year: Ending year for data retrieval
        api_key: EIA API key (get free at https://www.eia.gov/opendata/register.php)
        sectors: List of sector IDs to fetch. If None, fetches all sectors.

    Returns:
        DataFrame containing energy data for all sectors, or None if request fails
    """

    if city not in CITY_SERIES_MAP:
        print(f"Error: City '{city}' not found in mapping. Available cities: {list(CITY_SERIES_MAP.keys())}")
        return None

    city_info = CITY_SERIES_MAP[city]
    endpoint = f"{EIA_API_BASE_URL}electricity/retail-sales/data/"

    # API request parameters
    params = {
        "api_key": api_key,
        "frequency": "annual",
        "data[0]": "sales",
        "facets[stateid][]": city_info["state"],
        "start": str(start_year),
        "end": str(end_year),
        "sort[0][column]": "period",
        "sort[0][direction]": "asc",
        "length": 5000  # Ensure we get all records
    }

    # Add specific sectors if requested
    if sectors:
        for i, sector in enumerate(sectors):
            params[f"facets[sectorid][{i}]"] = sector

    try:
        print(f"Fetching energy data for {city_info['name']} ({start_year}-{end_year})...")
        response = requests.get(endpoint, params=params)
        response.raise_for_status()

        data = response.json()

        if "response" in data and "data" in data["response"]:
            records = data["response"]["data"]

            if not records:
                print(f"No data found for {city}")
                return None

            df = pd.DataFrame(records)

            df['city'] = city_info['name']
            df['city_code'] = city

            sector_counts = df['sectorName'].value_counts()
            print(f"Successfully retrieved {len(df)} records for {city_info['name']}")
            print(f"  Sectors: {', '.join(sector_counts.index.tolist())}")
            print(f"  Records per sector: {dict(sector_counts)}")
            return df
        else:
            print(f"Unexpected API response format for {city}")
            return None

    except requests.exceptions.RequestException as e:
        print(f"Error fetching data for {city}: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON response for {city}: {e}")
        return None


def fetch_all_cities(cities: List[str], start_year: int, end_year: int, api_key: str,
                     sectors: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Fetch energy data for multiple cities and combine into a single DataFrame.

    Args:
        cities: List of city identifiers
        start_year: Starting year for data retrieval
        end_year: Ending year for data retrieval
        api_key: EIA API key
        sectors: List of sector IDs. If None, fetches all sectors.

    Returns:
        Combined DataFrame with data from all cities and all sectors
    """


    all_data = []
    all_data = []

    for city in cities:
        df = get_energy_data(city, start_year, end_year, api_key, sectors)
        if df is not None:
            all_data.append(df)

    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        return combined_df
    else:
        return pd.DataFrame()


if __name__ == "__main__":
    API_KEY = "ousQOkjUKFziMAvFFY0LJ1m8UR2YaNFdv7Gn1AjY"

    cities = ["NYC", "LA", "Seattle"]
    start_year = 2014
    end_year = 2024

    print(f"Fetching energy consumption of {', '.join(cities)} from {start_year}-{end_year}...\n")
    df = fetch_all_cities(cities, start_year, end_year, API_KEY, sectors=None)

    if not df.empty:
        print("\n" + "=" * 50)
        print("DATA SUMMARY")
        print("=" * 50)
        print(f"Total records retrieved: {len(df)}")
        print(f"Cities: {df['city'].unique().tolist()}")
        print(f"Years: {sorted(df['period'].unique())}")
        print(f"Sectors: {sorted(df['sectorName'].unique())}")

        # Save each city to a separate CSV file
        print("\n" + "=" * 50)
        print("SAVING INDIVIDUAL CITY FILES")
        print("=" * 50)

        for city_code in df['city_code'].unique():
            # Filter data for this city
            city_df = df[df['city_code'] == city_code].copy()
            city_name = city_df['city'].iloc[0]

            # Filter for "all sectors" only to get total consumption
            city_total = city_df[city_df['sectorName'] == 'all sectors'].copy()

            output_df = pd.DataFrame({
                'Year': city_total['period'],
                'Consumption': city_total['sales'],
                'City': city_total['city']
            })

            # Sort by year
            output_df = output_df.sort_values('Year').reset_index(drop=True)

            # Save to CSV
            filename = f"{city_code}_energy_2014_2024.csv"
            output_df.to_csv(filename, index=False)
            print(f"{city_name}: {filename} ({len(output_df)} years)")

        print("\n" + "=" * 50)
        print("All files saved successfully!")
        print("=" * 50)
    else:
        print("No data retrieved.")



"""
@version: Python 3.10
@created by: Jim Lu
@contact: jimlo7760@gmail.com
@time: 10/4/25
"""