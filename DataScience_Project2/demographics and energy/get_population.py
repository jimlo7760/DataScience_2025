import requests
import pandas as pd
from typing import Dict, List

CITY_FIPS = {
    'NYC': {'state': '36', 'place': '51000', 'name': 'New York City'},
    'LA': {'state': '06', 'place': '44000', 'name': 'Los Angeles'},
    'Seattle': {'state': '53', 'place': '63000', 'name': 'Seattle'}
}


def get_city_population(city: str, api_key: str, start_year: int = 2014, end_year: int = 2023) -> pd.DataFrame:
    """
    Fetch city population data from U.S. Census Bureau API using ACS 1-Year Estimates.

    Args:
        city: City identifier
        api_key: Census Bureau API key
        start_year: Starting year for data
        end_year: Ending year for data

    Returns:
        DataFrame with year and population data
    """


    city_info = CITY_FIPS[city]
    place_fips = city_info['place']
    state_fips = city_info['state']
    population_data = []

    for year in range(start_year, end_year + 1):
        try:
            url = f"https://api.census.gov/data/{year}/acs/acs1"
            params = {
                "get": "NAME,B01003_001E",  # B01003_001E is total population
                "for": f"place:{place_fips}",
                "in": f"state:{state_fips}",
                "key": api_key
            }

            response = requests.get(url, params=params)
            response.raise_for_status()

            data = response.json()

            if len(data) > 1:
                population = int(data[1][1])
                population_data.append({
                    'Year': year,
                    'Population': population
                })
                print(f" {year}: {population:,}")

        except requests.exceptions.RequestException as e:
            print(f" Error for {year}: {e}")
            continue

    return pd.DataFrame(population_data)


if __name__ == "__main__":

    API_KEY = "de27d4bcfb9cf7bc2d78a1ed7a6413b37207dcec"

    cities = ['NYC', 'LA', 'Seattle']

    all_data = []

    for city in cities:
        print(f"\n{'=' * 60}")
        print(f"Fetching {CITY_FIPS[city]['name']} population data...")
        print('=' * 60)

        df_pop = get_city_population(city, API_KEY, 2014, 2023)

        df_pop['City'] = CITY_FIPS[city]['name']
        all_data.append(df_pop)

        print(f"\n{CITY_FIPS[city]['name']} POPULATION DATA (2014-2023)")
        print("-" * 60)

        if not df_pop.empty:
            print(df_pop[['Year', 'Population']].to_string(index=False))

            # Save to individual CSV file
            output_file = f"{city.lower()}_population_2014_2023.csv"
            df_pop[['Year', 'Population', 'City']].to_csv(output_file, index=False)
            print(f"\n Data saved to {output_file}")
        else:
            print("Error: No data retrieved")

"""
@version: Python 3.10
@created by: Jim Lu
@contact: jimlo7760@gmail.com
@time: 10/2/25
"""