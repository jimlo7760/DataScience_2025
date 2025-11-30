#!/usr/bin/env python3
"""
Verify Multi-City Demographic and Energy HDF5 File
Checks the integrity and structure of the multicity_demographic_energy_dataset.h5 file
"""

import h5py
import numpy as np
from pathlib import Path
import sys


def verify_hdf5_file(filepath):
    """Verify the multi-city dataset HDF5 file structure and integrity."""
    print("=" * 70)
    print("Multi-City Dataset HDF5 File Verification Report")
    print("=" * 70)
    print(f"\nFile: {filepath}")
    print(f"Size: {filepath.stat().st_size / 1024:.2f} KB\n")

    passed = True

    try:
        with h5py.File(filepath, 'r') as hdf:

            print("PASS: File is readable and valid HDF5 format")

            # Root attributes check
            print("\n" + "-" * 70)
            print("Root Attributes")
            print("-" * 70)
            required_attrs = ['dataset_name', 'description', 'creator',
                              'temporal_coverage', 'license']

            for attr in required_attrs:
                if attr in hdf.attrs:
                    value = hdf.attrs[attr]
                    if isinstance(value, bytes):
                        value = value.decode('utf-8')
                    # Truncate long values for display
                    value_str = str(value)
                    if len(value_str) > 60:
                        value_str = value_str[:60] + "..."
                    print(f"  PASS: {attr}: {value_str}")
                else:
                    print(f"  ERROR: Missing attribute '{attr}'")
                    passed = False

            # Groups structure - check for cities
            print("\n" + "-" * 70)
            print("City Groups Structure")
            print("-" * 70)
            expected_cities = ['NYC', 'LA', 'Seattle']

            for city in expected_cities:
                if city in hdf:
                    print(f"  PASS: Found city group: {city}")
                    city_group = hdf[city]

                    # Check city attributes
                    if 'city_name' in city_group.attrs:
                        city_name = city_group.attrs['city_name']
                        if isinstance(city_name, bytes):
                            city_name = city_name.decode('utf-8')
                        print(f"    City name: {city_name}")
                    else:
                        print(f"    WARNING: Missing 'city_name' attribute")
                else:
                    print(f"  ERROR: Missing city group '{city}'")
                    passed = False

            # Check for summary statistics group
            if 'summary_statistics' in hdf:
                print(f"  PASS: Found summary_statistics group")
            else:
                print(f"  WARNING: Missing 'summary_statistics' group")

            # Detailed check for each city
            for city in expected_cities:
                if city not in hdf:
                    continue

                print("\n" + "-" * 70)
                print(f"{city} - Detailed Structure Check")
                print("-" * 70)
                city_group = hdf[city]

                # Expected subgroups
                expected_subgroups = ['population', 'energy_consumption',
                                      'education_level', 'integrated_timeseries']

                for subgroup_name in expected_subgroups:
                    if subgroup_name in city_group:
                        print(f"  PASS: {subgroup_name}")
                        subgroup = city_group[subgroup_name]

                        # Check for 'year' dataset
                        if 'year' in subgroup:
                            year_data = subgroup['year'][:]
                            print(f"    - year: shape={year_data.shape}, "
                                  f"range=[{year_data.min()}-{year_data.max()}]")

                            # Verify years are reasonable (2014-2023)
                            if year_data.min() >= 2014 and year_data.max() <= 2023:
                                print(f"      PASS: Year range valid (2014-2023)")
                            else:
                                print(f"      ERROR: Invalid year range")
                                passed = False
                        else:
                            print(f"    ERROR: Missing 'year' dataset")
                            passed = False

                        # Skip 'values' check for integrated_timeseries (uses separate datasets)
                        if subgroup_name == 'integrated_timeseries':
                            # Don't check for 'values' - this group has separate datasets
                            pass
                        else:
                            # For other groups, check for 'values' dataset
                            if 'values' in subgroup:
                                values_data = subgroup['values'][:]
                                print(f"    - values: shape={values_data.shape}, "
                                      f"dtype={values_data.dtype}")
                                print(f"      range=[{values_data.min():.2f}, {values_data.max():.2f}], "
                                      f"mean={values_data.mean():.2f}")

                                # Check for data quality issues
                                if np.any(np.isnan(values_data)):
                                    print(f"      ERROR: Contains NaN values")
                                    passed = False
                                else:
                                    print(f"      PASS: No NaN values")

                                if np.any(np.isinf(values_data)):
                                    print(f"      ERROR: Contains infinite values")
                                    passed = False
                                else:
                                    print(f"      PASS: No infinite values")

                                # Check for negative values (shouldn't exist for population/energy/education)
                                if np.any(values_data < 0):
                                    print(f"      WARNING: Contains negative values")

                                # Check units attribute
                                if 'units' in subgroup['values'].attrs:
                                    units = subgroup['values'].attrs['units']
                                    if isinstance(units, bytes):
                                        units = units.decode('utf-8')
                                    print(f"      units: {units}")
                            else:
                                print(f"    ERROR: Missing 'values' dataset")
                                passed = False
                    else:
                        if subgroup_name == 'integrated_timeseries':
                            print(f"  INFO: {subgroup_name} (optional, not found)")
                        else:
                            print(f"  ERROR: Missing required subgroup '{subgroup_name}'")
                            passed = False

                # Check for education_detailed (optional)
                if 'education_detailed' in city_group:
                    print(f"  PASS: education_detailed (optional)")
                    edu_detail = city_group['education_detailed']

                    # Check for common education categories
                    categories = ['high_school_graduate', 'bachelors_degree',
                                  'masters_degree', 'doctorate_degree']
                    found_categories = [cat for cat in categories if cat in edu_detail]
                    if found_categories:
                        print(f"    Categories found: {', '.join(found_categories)}")

                # Check integrated_timeseries if present
                if 'integrated_timeseries' in city_group:
                    print(f"\n  Integrated Time Series:")
                    its = city_group['integrated_timeseries']

                    expected_datasets = ['year', 'population', 'energy_consumption',
                                         'education_level']
                    for ds_name in expected_datasets:
                        if ds_name in its:
                            ds = its[ds_name]
                            print(f"    PASS: {ds_name}: shape={ds.shape}")
                        else:
                            print(f"    ERROR: Missing '{ds_name}' in integrated_timeseries")
                            passed = False

            # Summary statistics verification
            print("\n" + "-" * 70)
            print("Summary Statistics")
            print("-" * 70)
            if 'summary_statistics' in hdf:
                stats_group = hdf['summary_statistics']

                for city in expected_cities:
                    if city in stats_group:
                        print(f"  PASS: {city} statistics")
                        city_stats = stats_group[city]

                        # Check for expected statistics datasets
                        expected_stats = ['population_stats', 'energy_stats',
                                          'education_stats']
                        for stat_name in expected_stats:
                            if stat_name in city_stats:
                                stat_data = city_stats[stat_name][:]
                                print(f"    - {stat_name}: {stat_data}")

                                # Verify we have 4 values (min, max, mean, std)
                                if len(stat_data) == 4:
                                    print(f"      PASS: Contains min, max, mean, std_dev")
                                else:
                                    print(f"      WARNING: Expected 4 values, got {len(stat_data)}")
                            else:
                                print(f"    WARNING: Missing '{stat_name}'")
                    else:
                        print(f"  WARNING: No statistics for {city}")

            # Compression check
            print("\n" + "-" * 70)
            print("Compression Settings")
            print("-" * 70)
            compression_checked = False
            for city in expected_cities:
                if city in hdf and 'population' in hdf[city]:
                    if 'values' in hdf[city]['population']:
                        ds = hdf[city]['population']['values']
                        if ds.compression:
                            print(f"  PASS: Compression: {ds.compression}")
                            print(f"  PASS: Compression level: {ds.compression_opts}")
                            compression_checked = True
                            break
                        else:
                            print(f"  WARNING: No compression applied")
                            compression_checked = True
                            break

            if not compression_checked:
                print(f"  WARNING: Could not verify compression settings")

            # Data consistency checks
            print("\n" + "-" * 70)
            print("Data Consistency Checks")
            print("-" * 70)

            consistency_warnings = 0
            for city in expected_cities:
                if city not in hdf:
                    continue

                city_group = hdf[city]

                # Check if all time series have the same years
                years_dict = {}
                for data_type in ['population', 'energy_consumption', 'education_level']:
                    if data_type in city_group and 'year' in city_group[data_type]:
                        years_dict[data_type] = city_group[data_type]['year'][:]

                if len(years_dict) > 1:
                    # Compare years across data types
                    first_years = list(years_dict.values())[0]
                    consistent = True
                    for data_type, years in years_dict.items():
                        if not np.array_equal(first_years, years):
                            print(f"  INFO: {city} - {data_type} has different years (expected due to data sources)")
                            consistency_warnings += 1
                            consistent = False

                    if consistent:
                        print(f"  PASS: {city} - All data types have consistent years")

            if consistency_warnings > 0:
                print(f"\n  Note: Year inconsistencies are expected due to:")
                print(f"    - Population data missing 2020 (Census collection issues)")
                print(f"    - Different data collection schedules across sources")

            # Final data integrity summary
            print("\n" + "-" * 70)
            print("Overall Data Integrity")
            print("-" * 70)

            total_datasets = 0
            valid_datasets = 0

            for city in expected_cities:
                if city not in hdf:
                    continue

                city_group = hdf[city]
                for subgroup_name in ['population', 'energy_consumption', 'education_level']:
                    if subgroup_name in city_group:
                        if 'values' in city_group[subgroup_name]:
                            total_datasets += 1
                            values = city_group[subgroup_name]['values'][:]
                            if not np.any(np.isnan(values)) and not np.any(np.isinf(values)):
                                valid_datasets += 1

            print(f"  Total datasets checked: {total_datasets}")
            print(f"  Valid datasets (no NaN/Inf): {valid_datasets}")

            if total_datasets == valid_datasets:
                print(f"  PASS: All datasets passed integrity checks")
            else:
                print(f"  ERROR: {total_datasets - valid_datasets} datasets have issues")
                passed = False

            # Summary
            print("\n" + "=" * 70)
            if passed:
                print("ALL CHECKS PASSED - File is valid")
            else:
                print("SOME CHECKS FAILED - Review errors above")
            print("=" * 70)

    except Exception as e:
        print(f"\nERROR: Failed to read HDF5 file")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        passed = False

    return passed


def main():
    """Main execution function."""
    filepath = Path('outputs/multicity_demographic_energy_dataset.h5')

    if not filepath.exists():
        print(f"Error: File not found at {filepath}")
        print(f"\nPlease run 'python convert_multicity_to_hdf5.py' first to create the file.")
        sys.exit(1)

    success = verify_hdf5_file(filepath)

    if success:
        print("\nVerification complete - no issues found")
    else:
        print("\nVerification complete - issues detected")

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()

"""
@version: Python 3.10
@created by: Jim Lu
@contact: jimlo7760@gmail.com
@time: 11/29/25
"""