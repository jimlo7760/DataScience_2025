import requests
import pandas as pd
import time

API_KEY = "de27d4bcfb9cf7bc2d78a1ed7a6413b37207dcec"

# City configurations: (state FIPS, place FIPS, city name)
CITIES = {
    "NYC": ("36", "51000", "New York"),
    "LA": ("06", "44000", "Los Angeles"),
    "Seattle": ("53", "63000", "Seattle")
}

# Education level categories from ACS Table B15003
EDUCATION_VARS = {
    "B15003_017E": "High school graduate",
    "B15003_018E": "GED or alternative",
    "B15003_019E": "Some college, less than 1 year",
    "B15003_020E": "Some college, 1+ years, no degree",
    "B15003_021E": "Associate's degree",
    "B15003_022E": "Bachelor's degree",
    "B15003_023E": "Master's degree",
    "B15003_024E": "Professional degree",
    "B15003_025E": "Doctorate degree",
    "B15003_001E": "Total population 25+"
}


def fetch_acs_data(year: int, state_fips: str, place_fips: str) -> dict:
    """
    Fetch education data from ACS 5-year estimates for a specific place.

    Args:
        year: Year for data retrieval
        state_fips: State FIPS code
        place_fips: Place FIPS code

    Returns:
        Dictionary containing ACS data, or None if request fails
    """
    base_url = f"https://api.census.gov/data/{year}/acs/acs5"

    # Get all education variables
    vars_string = ",".join(EDUCATION_VARS.keys())

    params = {
        "get": vars_string,
        "for": f"place:{place_fips}",
        "in": f"state:{state_fips}",
        "key": API_KEY
    }

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()

        # Convert to dictionary (skip header row)
        if len(data) > 1:
            headers = data[0]
            values = data[1]
            return dict(zip(headers, values))
        return None
    except Exception as e:
        print(f"Error fetching data for year {year}: {e}")
        return None


def calculate_education_level(data_dict: dict) -> float:
    """
    Calculate weighted average education level from ACS data.

    Args:
        data_dict: Dictionary containing ACS education variables

    Returns:
        Weighted average education level in years, or None if calculation fails
    """
    if not data_dict:
        return None

    # Education level weights (simplified scoring system)
    weights = {
        "High school graduate": 12,
        "GED or alternative": 12,
        "Some college, less than 1 year": 13,
        "Some college, 1+ years, no degree": 14,
        "Associate's degree": 14,
        "Bachelor's degree": 16,
        "Master's degree": 18,
        "Professional degree": 19,
        "Doctorate degree": 21
    }

    total_weighted = 0
    total_population = 0

    for var_code, education_level in EDUCATION_VARS.items():
        if education_level == "Total population 25+":
            continue

        try:
            count = int(data_dict.get(var_code, 0))
            weight = weights[education_level]
            total_weighted += count * weight
            total_population += count
        except (ValueError, KeyError):
            continue

    if total_population > 0:
        return round(total_weighted / total_population, 2)
    return None


def main():
    """Main execution function to fetch and save education data for all cities."""

    print("Fetching ACS Education Data (2014-2023)...")
    print("=" * 50)


    # Process each city
    for city_key, (state_fips, place_fips, city_name) in CITIES.items():
        print(f"\nFetching data for {city_name}...")

        city_data = []

        # Fetch data for each year (2014-2023)
        for year in range(2014, 2025):
            print(f"  {year}...", end=" ")

            data = fetch_acs_data(year, state_fips, place_fips)

            if data:
                education_level = calculate_education_level(data)

                if education_level:
                    city_data.append({
                        "Year": year,
                        "Education level": education_level,
                        "City": city_name
                    })
                    print(f"✓ (Avg: {education_level} years)")
                else:
                    print("✗ (Could not calculate)")
            else:
                print("✗ (No data)")

            # Be respectful to the API
            time.sleep(0.5)

        # Save to CSV
        if city_data:
            df = pd.DataFrame(city_data)
            filename = f"{city_key}_education_2014_2023.csv"
            df.to_csv(filename, index=False)
            print(f"  Saved: {filename} ({len(city_data)} years)")
        else:
            print(f"  No data collected for {city_name}")

    print("\n" + "=" * 50)
    print("All files saved successfully!")
    print("=" * 50)


if __name__ == "__main__":
    main()


"""
@version: Python 3.10
@created by: Jim Lu
@contact: jimlo7760@gmail.com
@time: 10/4/25
"""