"""
Generate metadata.json from existing HDF5 attributes
Creates proper metadata for Multi-City Demographic and Energy Dataset
"""

import json
import h5py
from pathlib import Path
from datetime import datetime
import hashlib


def compute_checksums(filepath):
    """Compute MD5 and SHA256 checksums"""
    md5 = hashlib.md5()
    sha256 = hashlib.sha256()

    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            md5.update(chunk)
            sha256.update(chunk)

    return md5.hexdigest(), sha256.hexdigest()


def generate_metadata_json(hdf5_path, output_path):
    """
    Generate metadata.json using existing HDF5 metadata
    plus proper archival additions for multi-city dataset
    """

    # Read existing metadata from HDF5
    with h5py.File(hdf5_path, 'r') as hdf:
        # Extract all root attributes
        metadata_from_hdf5 = {}
        for key, value in hdf.attrs.items():
            # Convert numpy arrays and bytes to native Python types
            if hasattr(value, 'tolist'):
                metadata_from_hdf5[key] = value.tolist()
            elif isinstance(value, bytes):
                metadata_from_hdf5[key] = value.decode('utf-8')
            else:
                metadata_from_hdf5[key] = value

        # Get information about the structure
        cities = list(hdf.keys())
        cities = [c for c in cities if c in ['NYC', 'LA', 'Seattle']]

    # Compute checksums
    md5_hash, sha256_hash = compute_checksums(hdf5_path)
    file_size_kb = hdf5_path.stat().st_size / 1024

    # Build proper metadata structure for multi-city dataset
    metadata = {
        # Dataset information from HDF5 attributes
        "dataset_information": {
            "title": metadata_from_hdf5.get('dataset_name', 'Multi-City Demographic and Energy Dataset'),
            "creator": metadata_from_hdf5.get('creator', 'Jim Lu'),
            "creator_affiliation": metadata_from_hdf5.get('creator_affiliation', 'Rensselaer Polytechnic Institute'),
            "date_created": datetime.now().strftime('%Y-%m-%d'),
            "description": metadata_from_hdf5.get('description', 'Combined dataset for analyzing demographic trends, educational attainment, and energy consumption patterns across three major U.S. cities.'),
            "temporal_coverage": metadata_from_hdf5.get('temporal_coverage', '2014-2023'),
            "cities_covered": cities,
            "num_cities": len(cities),
            "data_types": [
                "Population (Census ACS 1-Year Estimates)",
                "Energy Consumption (EIA State Energy Data System)",
                "Educational Attainment (Census ACS 5-Year Estimates)"
            ]
        },

        # Data sources and methodology
        "data_sources": {
            "population": {
                "source": "U.S. Census Bureau American Community Survey (ACS) 1-Year Estimates",
                "variables": "B01003_001E (Total Population)",
                "api_endpoint": "https://api.census.gov/data/{year}/acs/acs1",
                "temporal_resolution": "Annual",
                "years_covered": "2014-2023"
            },
            "energy_consumption": {
                "source": "U.S. Energy Information Administration (EIA) State Energy Data System",
                "variables": "Electricity retail sales (all sectors combined)",
                "api_endpoint": "https://api.eia.gov/v2/electricity/retail-sales/data/",
                "temporal_resolution": "Annual",
                "years_covered": "2014-2023",
                "units": "Million kilowatt hours (GWh)"
            },
            "education": {
                "source": "U.S. Census Bureau American Community Survey (ACS) 5-Year Estimates",
                "table": "B15003 - Educational Attainment for the Population 25 Years and Over",
                "api_endpoint": "https://api.census.gov/data/{year}/acs/acs5",
                "temporal_resolution": "Annual (5-year rolling averages)",
                "years_covered": "2014-2023",
                "calculation_method": "Weighted average of education levels using equivalent years of schooling (12-21 years)",
                "units": "Years of education"
            }
        },

        # Geographic coverage
        "geographic_coverage": {
            "NYC": {
                "city_name": "New York City",
                "state": "New York",
                "state_code": "NY",
                "state_fips": "36",
                "place_fips": "51000"
            },
            "LA": {
                "city_name": "Los Angeles",
                "state": "California",
                "state_code": "CA",
                "state_fips": "06",
                "place_fips": "44000"
            },
            "Seattle": {
                "city_name": "Seattle",
                "state": "Washington",
                "state_code": "WA",
                "state_fips": "53",
                "place_fips": "63000"
            }
        },

        # Data quality and limitations
        "data_quality": {
            "limitations": [
                "ACS 1-Year Estimates have margins of error that should be considered",
                "ACS 5-Year Estimates represent rolling 5-year averages",
                "Energy data represents state-level totals, not city-specific consumption",
                "Education data uses simplified weighting scheme (12-21 years)",
                "2020 population data missing due to COVID-19 impacts on ACS data collection"
            ],
            "quality_notes": "All data retrieved from authoritative government sources (U.S. Census Bureau, EIA). Data processing scripts included in archive."
        },

        # Archival information
        "archival_information": {
            "archive_date": datetime.now().strftime('%Y-%m-%d'),
            "depositor": "Jim Lu",
            "depositor_affiliation": "Rensselaer Polytechnic Institute",
            "depositor_email": "jimlo7760@gmail.com",
            "license": "CC0 1.0 Universal (Public Domain) - Original government data is public domain",
            "recommended_citation": "Lu, Jim (2025). Multi-City Demographic and Energy Dataset (2014-2023). Dataset compiled from U.S. Census Bureau ACS and EIA State Energy Data System.",
            "purpose": "Academic research on relationships between demographics, education, and energy consumption in major U.S. cities"
        },

        # Keywords for discovery
        "keywords": [
            "population",
            "demographics",
            "energy consumption",
            "electricity",
            "education",
            "educational attainment",
            "urban data",
            "time series",
            "New York City",
            "Los Angeles",
            "Seattle",
            "Census",
            "ACS",
            "EIA",
            "multi-city analysis"
        ],

        # Technical file information
        "file_information": {
            "filename": hdf5_path.name,
            "format": "HDF5",
            "format_version": "HDF5 1.10+",
            "size_kb": round(file_size_kb, 2),
            "compression": "gzip level 9",
            "md5_checksum": md5_hash,
            "sha256_checksum": sha256_hash,
            "created_with": "Python 3.10, h5py, pandas, numpy"
        },

        # HDF5 structure documentation
        "hdf5_structure": {
            "description": "Hierarchical organization by city, then by data type",
            "root_level": {
                "groups": ["NYC", "LA", "Seattle", "summary_statistics"],
                "attributes": [
                    "dataset_name",
                    "description",
                    "creator",
                    "temporal_coverage",
                    "license",
                    "citation",
                    "keywords"
                ]
            },
            "city_level_groups": {
                "description": "Each city has groups for different data types",
                "subgroups": [
                    "population - Annual population estimates",
                    "energy_consumption - Annual electricity consumption",
                    "education_level - Weighted average years of education",
                    "education_detailed - Breakdown by education category",
                    "integrated_timeseries - Combined data for correlation analysis"
                ]
            },
            "dataset_attributes": {
                "description": "Each dataset includes metadata as HDF5 attributes",
                "common_attributes": [
                    "source - Data source name",
                    "units - Measurement units",
                    "variable - Variable description"
                ]
            }
        },

        # Related resources
        "related_resources": {
            "census_api": "https://www.census.gov/data/developers/data-sets.html",
            "eia_api": "https://www.eia.gov/opendata/",
            "census_acs_documentation": "https://www.census.gov/programs-surveys/acs/technical-documentation.html",
            "eia_seds_documentation": "https://www.eia.gov/state/seds/"
        },

        # Processing history
        "processing_history": {
            "conversion_date": datetime.now().strftime('%Y-%m-%d'),
            "conversion_tool": "Python 3.10 with h5py library",
            "original_formats": ["CSV files from API requests"],
            "processing_steps": [
                "1. Retrieved raw data from Census and EIA APIs using Python scripts",
                "2. Processed and cleaned data using pandas",
                "3. Calculated derived metrics (weighted education averages)",
                "4. Organized data hierarchically by city and data type",
                "5. Converted to HDF5 with metadata preservation",
                "6. Applied gzip compression level 9",
                "7. Generated integrated time series for analysis"
            ],
            "scripts_included": [
                "convert_multicity_to_hdf5.py - Main conversion script",
                "get_population.py - Population data retrieval",
                "get_energy.py - Energy data retrieval",
                "get_education.py - Education data retrieval"
            ]
        }
    }

    # Write to file with proper formatting
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"✓ Metadata JSON created: {output_path}")
    print(f"  Size: {output_path.stat().st_size / 1024:.2f} KB")
    print(f"  Sections: {len(metadata)}")
    print(f"\n  Dataset: Multi-City Demographic and Energy Dataset")
    print(f"  Cities: {', '.join(cities)}")
    print(f"  Temporal coverage: 2014-2023")
    print(f"  Data types: Population, Energy Consumption, Education")


def main():
    multicity_hdf5_path = Path('outputs/multicity_demographic_energy_dataset.h5')
    multicity_metadata_path = Path('outputs/multicity_demographic_energy_dataset.json')

    # Create output directory if needed
    multicity_metadata_path.parent.mkdir(parents=True, exist_ok=True)

    if not multicity_hdf5_path.exists():
        print(f"Error: {multicity_hdf5_path} not found.")
        print("Please run convert_multicity_to_hdf5.py first.")
        return

    generate_metadata_json(multicity_hdf5_path, multicity_metadata_path)
    print("\n" + "="*70)
    print("Metadata generation complete!")
    print("="*70)


if __name__ == '__main__':
    main()

"""
@version: Python 3.10
@created by: Jim Lu
@contact: jimlo7760@gmail.com
@time: 11/29/25
"""