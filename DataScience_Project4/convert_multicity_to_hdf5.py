#!/usr/bin/env python3
"""
Convert Multi-City Demographic and Energy Dataset to HDF5 Format

This script converts the demographic and energy data for NYC, LA, and Seattle
to HDF5 format, preserving all data, metadata, and relationships.

Following the same pattern as the Iris dataset conversion.
"""

import h5py
import numpy as np
import pandas as pd
from pathlib import Path
import json

# City information
CITIES = {
    'NYC': {
        'name': 'New York City',
        'state': 'New York',
        'state_code': 'NY',
        'state_fips': '36',
        'place_fips': '51000'
    },
    'LA': {
        'name': 'Los Angeles',
        'state': 'California',
        'state_code': 'CA',
        'state_fips': '06',
        'place_fips': '44000'
    },
    'Seattle': {
        'name': 'Seattle',
        'state': 'Washington',
        'state_code': 'WA',
        'state_fips': '53',
        'place_fips': '63000'
    }
}


def parse_metadata(metadata_path):
    """
    Parse the metadata.json file to extract dataset information.

    Returns:
        dict: Dictionary containing parsed metadata
    """
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    return metadata


def load_city_data(data_dir, city_code):
    """
    Load all data files for a specific city.

    Args:
        data_dir: Path to data directory
        city_code: City code (NYC, LA, Seattle)

    Returns:
        dict: Dictionary containing population, energy, and education DataFrames
    """
    city_data = {}

    # Load population data
    pop_file = data_dir / 'raw' / f'{city_code.lower()}_population_2014_2023.csv'
    if pop_file.exists():
        city_data['population'] = pd.read_csv(pop_file)

    # Load energy data
    energy_file = data_dir / 'raw' / f'{city_code}_energy_2014_2023.csv'
    if energy_file.exists():
        city_data['energy'] = pd.read_csv(energy_file)

    # Load education data (curated)
    edu_file = data_dir / 'curated' / f'{city_code}_education_2014_2023.csv'
    if edu_file.exists():
        city_data['education'] = pd.read_csv(edu_file)

    # Load raw education data
    raw_edu_file = data_dir / 'raw' / f'{city_code}_raw_education_2014_2023.csv'
    if raw_edu_file.exists():
        city_data['raw_education'] = pd.read_csv(raw_edu_file)

    return city_data


def create_hdf5_file(output_path, data_dir):
    """
    Create HDF5 file with all multi-city demographic and energy data.

    Args:
        output_path: Path where HDF5 file will be saved
        data_dir: Directory containing the data files
    """
    # Parse metadata
    print("Parsing metadata from metadata.json...")
    metadata = parse_metadata(data_dir / 'metadata.json')

    # Create HDF5 file
    print(f"Creating HDF5 file: {output_path}")
    with h5py.File(output_path, 'w') as hdf:

        # Add root-level metadata
        hdf.attrs['dataset_name'] = metadata['name']
        hdf.attrs['description'] = metadata['description']
        hdf.attrs['creator'] = metadata['creator']['name']
        hdf.attrs['creator_affiliation'] = metadata['creator']['affiliation']
        hdf.attrs['temporal_coverage'] = metadata['temporalCoverage']
        hdf.attrs['license'] = metadata['license']
        hdf.attrs['citation'] = metadata['citation']

        # Store keywords as array
        hdf.attrs['keywords'] = np.array(metadata['keywords'], dtype=h5py.string_dtype())

        # Process each city
        for city_code, city_info in CITIES.items():
            print(f"Loading {city_info['name']}...")

            # Load city data
            city_data = load_city_data(data_dir, city_code)

            if not city_data:
                print(f"  Warning: No data found for {city_info['name']}")
                continue

            # Create city group
            city_group = hdf.create_group(city_code)
            city_group.attrs['city_name'] = city_info['name']
            city_group.attrs['state'] = city_info['state']
            city_group.attrs['state_code'] = city_info['state_code']
            city_group.attrs['state_fips'] = city_info['state_fips']
            city_group.attrs['place_fips'] = city_info['place_fips']

            # Add population data
            if 'population' in city_data:
                pop_df = city_data['population']
                pop_group = city_group.create_group('population')
                pop_group.attrs['source'] = 'U.S. Census Bureau ACS 1-Year Estimates'
                pop_group.attrs['variable'] = 'B01003_001E (Total Population)'

                pop_group.create_dataset('year', data=pop_df['Year'].values,
                                         dtype=np.int32, compression='gzip', compression_opts=9)
                pop_dataset = pop_group.create_dataset('values', data=pop_df['Population'].values,
                                                       dtype=np.int64, compression='gzip', compression_opts=9)
                pop_dataset.attrs['units'] = 'persons'

            # Add energy data
            if 'energy' in city_data:
                energy_df = city_data['energy']
                energy_group = city_group.create_group('energy_consumption')
                energy_group.attrs['source'] = 'EIA State Energy Data System'
                energy_group.attrs['variable'] = 'Electricity sales (all sectors)'

                energy_group.create_dataset('year', data=energy_df['Year'].values,
                                            dtype=np.int32, compression='gzip', compression_opts=9)
                energy_dataset = energy_group.create_dataset('values', data=energy_df['Consumption'].values,
                                                             dtype=np.float64, compression='gzip', compression_opts=9)
                energy_dataset.attrs['units'] = 'million kilowatt hours'

            # Add education data
            if 'education' in city_data:
                edu_df = city_data['education']
                edu_group = city_group.create_group('education_level')
                edu_group.attrs['source'] = 'U.S. Census Bureau ACS 5-Year Estimates'
                edu_group.attrs['table'] = 'B15003 (Educational Attainment)'
                edu_group.attrs['calculation_method'] = 'Weighted average (12-21 years)'

                edu_group.create_dataset('year', data=edu_df['Year'].values,
                                         dtype=np.int32, compression='gzip', compression_opts=9)
                edu_dataset = edu_group.create_dataset('values', data=edu_df['Education level'].values,
                                                       dtype=np.float64, compression='gzip', compression_opts=9)
                edu_dataset.attrs['units'] = 'years'
                edu_dataset.attrs['range'] = '12-21 years'

            # Add detailed education data
            if 'raw_education' in city_data:
                raw_edu_df = city_data['raw_education']
                detail_group = city_group.create_group('education_detailed')
                detail_group.attrs['source'] = 'U.S. Census Bureau ACS 5-Year Estimates'
                detail_group.attrs['description'] = 'Detailed breakdown by education level'

                detail_group.create_dataset('year', data=raw_edu_df['Year'].values,
                                            dtype=np.int32, compression='gzip', compression_opts=9)

                # Store each education category
                categories = {
                    'high_school_graduate': 'High school graduate',
                    'bachelors_degree': 'Bachelor\'s degree',
                    'masters_degree': 'Master\'s degree',
                    'doctorate_degree': 'Doctorate degree'
                }

                for key, label in categories.items():
                    if label in raw_edu_df.columns:
                        dataset = detail_group.create_dataset(
                            key, data=raw_edu_df[label].values,
                            dtype=np.int64, compression='gzip', compression_opts=9)
                        dataset.attrs['category'] = label
                        dataset.attrs['units'] = 'persons'

            # Create integrated time series for correlation analysis
            if all(k in city_data for k in ['population', 'energy', 'education']):
                ts_group = city_group.create_group('integrated_timeseries')
                ts_group.attrs['description'] = 'Combined time series for correlation analysis'

                # Merge data on Year
                merged = city_data['population'].merge(
                    city_data['energy'], on='Year', suffixes=('_pop', '_energy'))
                merged = merged.merge(city_data['education'], on='Year')

                ts_group.create_dataset('year', data=merged['Year'].values,
                                        dtype=np.int32, compression='gzip', compression_opts=9)

                pop_ds = ts_group.create_dataset('population', data=merged['Population'].values,
                                                 dtype=np.int64, compression='gzip', compression_opts=9)
                pop_ds.attrs['units'] = 'persons'

                energy_ds = ts_group.create_dataset('energy_consumption',
                                                    data=merged['Consumption'].values,
                                                    dtype=np.float64, compression='gzip', compression_opts=9)
                energy_ds.attrs['units'] = 'million kilowatt hours'

                edu_ds = ts_group.create_dataset('education_level',
                                                 data=merged['Education level'].values,
                                                 dtype=np.float64, compression='gzip', compression_opts=9)
                edu_ds.attrs['units'] = 'years'

        # Create summary statistics group
        stats_group = hdf.create_group('summary_statistics')

        for city_code in CITIES.keys():
            city_data = load_city_data(data_dir, city_code)

            if all(k in city_data for k in ['population', 'energy', 'education']):
                city_stats = stats_group.create_group(city_code)
                city_stats.attrs['city_name'] = CITIES[city_code]['name']

                # Population statistics
                pop_values = city_data['population']['Population'].values
                pop_stats = city_stats.create_dataset(
                    'population_stats',
                    data=np.array([np.min(pop_values), np.max(pop_values),
                                   np.mean(pop_values), np.std(pop_values, ddof=1)]),
                    dtype=np.float64)
                pop_stats.attrs['statistics'] = np.array(['min', 'max', 'mean', 'std_dev'],
                                                         dtype=h5py.string_dtype())

                # Energy statistics
                energy_values = city_data['energy']['Consumption'].values
                energy_stats = city_stats.create_dataset(
                    'energy_stats',
                    data=np.array([np.min(energy_values), np.max(energy_values),
                                   np.mean(energy_values), np.std(energy_values, ddof=1)]),
                    dtype=np.float64)
                energy_stats.attrs['statistics'] = np.array(['min', 'max', 'mean', 'std_dev'],
                                                            dtype=h5py.string_dtype())

                # Education statistics
                edu_values = city_data['education']['Education level'].values
                edu_stats = city_stats.create_dataset(
                    'education_stats',
                    data=np.array([np.min(edu_values), np.max(edu_values),
                                   np.mean(edu_values), np.std(edu_values, ddof=1)]),
                    dtype=np.float64)
                edu_stats.attrs['statistics'] = np.array(['min', 'max', 'mean', 'std_dev'],
                                                         dtype=h5py.string_dtype())

        # Add data quality information
        quality_group = hdf.create_group('data_quality')
        quality_group.attrs['limitations'] = np.array(metadata['limitations'], dtype=h5py.string_dtype())

    print(f"✓ HDF5 file created successfully: {output_path}")
    print(f"  - Cities: {len(CITIES)}")
    print(f"  - Time period: 2014-2023")


def print_hdf5_structure(filepath):
    """
    Print the structure of the created HDF5 file.
    """
    print("\n" + "=" * 60)
    print("HDF5 File Structure")
    print("=" * 60)

    with h5py.File(filepath, 'r') as hdf:
        print(f"\nRoot attributes:")
        for key, value in hdf.attrs.items():
            if isinstance(value, np.ndarray):
                print(f"  {key}: [{len(value)} items]")
            else:
                value_str = str(value)
                if len(value_str) > 80:
                    value_str = value_str[:80] + "..."
                print(f"  {key}: {value_str}")

        def print_group(name, obj):
            indent = "  " * (name.count('/'))
            if isinstance(obj, h5py.Group):
                print(f"\n{indent}Group: {name}")
                if obj.attrs:
                    print(f"{indent}  Attributes:")
                    for key, value in obj.attrs.items():
                        if isinstance(value, np.ndarray) and len(value) > 3:
                            print(f"{indent}    {key}: [{len(value)} items]")
                        else:
                            print(f"{indent}    {key}: {value}")
            elif isinstance(obj, h5py.Dataset):
                print(f"{indent}Dataset: {name}")
                print(f"{indent}  Shape: {obj.shape}")
                print(f"{indent}  Dtype: {obj.dtype}")
                if obj.attrs:
                    print(f"{indent}  Attributes:")
                    for key, value in obj.attrs.items():
                        print(f"{indent}    {key}: {value}")

        hdf.visititems(print_group)

    print("\n" + "=" * 60)


def main():
    """Main execution function."""
    # Define paths
    data_dir = Path('../DataScience_Project2/demographics and energy/')
    output_path = Path('outputs/multicity_demographic_energy_dataset.h5')

    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create HDF5 file
    create_hdf5_file(output_path, data_dir)

    # Print structure
    print_hdf5_structure(output_path)

    print(f"\n Conversion complete!")
    print(f"  Output file: {output_path}")
    print(f"  File size: {output_path.stat().st_size / 1024:.2f} KB")


if __name__ == '__main__':
    main()

"""
@version: Python 3.10
@created by: Jim Lu
@contact: jimlo7760@gmail.com
@time: 11/27/25
"""
