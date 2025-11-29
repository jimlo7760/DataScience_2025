#!/usr/bin/env python3
"""
Verify Iris HDF5 File
Checks the integrity and structure of the iris_archive.h5 file
"""

import h5py
import numpy as np
from pathlib import Path
import sys


def verify_hdf5_file(filepath):
    """Verify the iris dataset HDF5 file structure and integrity."""
    print("="*70)
    print("HDF5 File Verification Report")
    print("="*70)
    print(f"\nFile: {filepath}")
    print(f"Size: {filepath.stat().st_size / 1024:.2f} KB\n")

    passed = True

    try:
        with h5py.File(filepath, 'r') as hdf:

            print("File is readable and valid HDF5 format")

            # Root attributes check
            print("\n" + "-"*70)
            print("Root Attributes")
            print("-"*70)
            required_attrs = ['title', 'creator', 'donor', 'num_instances',
                            'num_attributes', 'num_classes']

            for attr in required_attrs:
                if attr in hdf.attrs:
                    value = hdf.attrs[attr]
                    print(f"  {attr}: {value}")
                else:
                    print(f"  ERROR: Missing attribute '{attr}'")
                    passed = False

            # Groups structure
            print("\n" + "-"*70)
            print("Groups Structure")
            print("-"*70)
            expected_groups = ['original_data', 'corrected_data',
                              'summary_statistics', 'by_class']

            for group_name in expected_groups:
                if group_name in hdf:
                    print(f"  Found group: {group_name}")
                else:
                    print(f"  ERROR: Missing group '{group_name}'")
                    passed = False

            # Original data check
            print("\n" + "-"*70)
            print("Original Data")
            print("-"*70)
            if 'original_data' in hdf:
                orig = hdf['original_data']
                if 'features' in orig:
                    features = orig['features']
                    print(f"  Features shape: {features.shape}")
                    print(f"  Features dtype: {features.dtype}")

                    if features.shape == (150, 4):
                        print(f"  Shape OK (150 samples, 4 features)")
                    else:
                        print(f"  ERROR: Unexpected shape {features.shape}")
                        passed = False
                else:
                    print(f"  ERROR: Missing 'features' dataset")
                    passed = False

                if 'class_labels' in orig:
                    labels = orig['class_labels']
                    print(f"  Labels shape: {labels.shape}")
                    unique_classes = np.unique(labels)
                    print(f"  Unique classes ({len(unique_classes)}): {[c.decode() if isinstance(c, bytes) else c for c in unique_classes]}")
                else:
                    print(f"  ERROR: Missing 'class_labels' dataset")
                    passed = False

            # Corrected data check
            print("\n" + "-"*70)
            print("Corrected Data")
            print("-"*70)
            if 'corrected_data' in hdf:
                corr = hdf['corrected_data']
                if 'features' in corr:
                    features = corr['features']
                    print(f"  Features shape: {features.shape}")
                    print(f"  Features dtype: {features.dtype}")

                    print(f"\n  Feature statistics:")
                    feature_names = ['sepal_length', 'sepal_width',
                                   'petal_length', 'petal_width']
                    for i, name in enumerate(feature_names):
                        col = features[:, i]
                        print(f"    {name}: min={col.min():.2f}, max={col.max():.2f}, mean={col.mean():.2f}")
                else:
                    print(f"  ERROR: Missing 'features' dataset")
                    passed = False

                if 'class_labels' in corr:
                    print(f"  Class labels present")
                else:
                    print(f"  ERROR: Missing 'class_labels' dataset")
                    passed = False

            # Summary statistics check
            print("\n" + "-"*70)
            print("Summary Statistics")
            print("-"*70)
            if 'summary_statistics' in hdf:
                stats = hdf['summary_statistics']
                if 'feature_statistics' in stats:
                    stats_data = stats['feature_statistics'][:]
                    print(f"  Statistics shape: {stats_data.shape}")
                    print(f"  Contains: min, max, mean, std for 4 features")
                else:
                    print(f"  ERROR: Missing 'feature_statistics' dataset")
                    passed = False

            # By-class grouping check
            print("\n" + "-"*70)
            print("By-Class Grouping")
            print("-"*70)
            if 'by_class' in hdf:
                by_class = hdf['by_class']
                expected_classes = ['Iris_setosa', 'Iris_versicolor', 'Iris_virginica']

                for class_name in expected_classes:
                    if class_name in by_class:
                        class_group = by_class[class_name]
                        if 'features' in class_group:
                            num_samples = class_group['features'].shape[0]
                            print(f"  {class_name}: {num_samples} samples")

                            if num_samples != 50:
                                print(f"    WARNING: Expected 50 samples, got {num_samples}")
                        else:
                            print(f"  ERROR: {class_name} missing 'features' dataset")
                            passed = False
                    else:
                        print(f"  ERROR: Missing class '{class_name}'")
                        passed = False

            # Compression info
            print("\n" + "-"*70)
            print("Compression Settings")
            print("-"*70)
            if 'corrected_data' in hdf and 'features' in hdf['corrected_data']:
                ds = hdf['corrected_data']['features']
                if ds.compression:
                    print(f"  Compression: {ds.compression}")
                    print(f"  Compression opts: {ds.compression_opts}")
                else:
                    print(f"  No compression applied")

            # Data integrity
            print("\n" + "-"*70)
            print("Data Integrity Checks")
            print("-"*70)
            if 'corrected_data' in hdf and 'features' in hdf['corrected_data']:
                features = hdf['corrected_data']['features'][:]

                if not np.any(np.isnan(features)):
                    print(f"  No NaN values")
                else:
                    print(f"  ERROR: NaN values found")
                    passed = False

                if not np.any(np.isinf(features)):
                    print(f"  No infinite values")
                else:
                    print(f"  ERROR: Infinite values found")
                    passed = False

                # reasonable range check for iris data
                if np.all((features >= 0) & (features <= 10)):
                    print(f"  All values in reasonable range (0-10 cm)")
                else:
                    print(f"  WARNING: Some values outside typical range")

            # Summary
            print("\n" + "="*70)
            if passed:
                print("ALL CHECKS PASSED - File is valid")
            else:
                print("SOME CHECKS FAILED - Review errors above")
            print("="*70)

    except Exception as e:
        print(f"\nERROR: Failed to read HDF5 file")
        print(f"  {type(e).__name__}: {e}")
        passed = False

    return passed


def main():
    """Main execution function."""
    filepath = Path('../outputs/iris_dataset.h5')

    if not filepath.exists():
        print(f"Error: File not found at {filepath}")
        print(f"\nPlease run 'python convert_iris_to_hdf5.py' first to create the file.")
        sys.exit(1)

    success = verify_hdf5_file(filepath)

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()