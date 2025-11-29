"""
Generate metadata.json from existing HDF5 attributes
Stays minimal and respects original metadata structure
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
    Generate metadata.json using ONLY existing HDF5 metadata
    plus minimal archival additions
    """

    # Read existing metadata from HDF5
    with h5py.File(hdf5_path, 'r') as hdf:
        # Extract all root attributes (your parse_iris_names data)
        metadata_from_hdf5 = {}
        for key, value in hdf.attrs.items():
            # Convert numpy arrays and bytes to native Python types
            if hasattr(value, 'tolist'):
                metadata_from_hdf5[key] = value.tolist()
            elif isinstance(value, bytes):
                metadata_from_hdf5[key] = value.decode('utf-8')
            else:
                metadata_from_hdf5[key] = value

    # Compute checksums
    md5_hash, sha256_hash = compute_checksums(hdf5_path)
    file_size_kb = hdf5_path.stat().st_size / 1024

    # Build minimal metadata structure
    # Using ONLY what exists + essential archival fields
    metadata = {
        # Original metadata from parse_iris_names (preserved as-is)
        "dataset_information": {
            "title": metadata_from_hdf5.get('title'),
            "creator": metadata_from_hdf5.get('creator'),
            "donor": metadata_from_hdf5.get('donor'),
            "date": metadata_from_hdf5.get('date'),
            "description": metadata_from_hdf5.get('description'),
            "num_instances": metadata_from_hdf5.get('num_instances'),
            "num_attributes": metadata_from_hdf5.get('num_attributes'),
            "num_classes": metadata_from_hdf5.get('num_classes'),
            "instances_per_class": metadata_from_hdf5.get('instances_per_class'),
            "attribute_names": metadata_from_hdf5.get('attribute_names'),
            "class_names": metadata_from_hdf5.get('class_names'),
            "missing_values": metadata_from_hdf5.get('missing_values'),
            "class_distribution": metadata_from_hdf5.get('class_distribution')
        },

        # Known data quality issues (from parse_iris_names)
        "data_quality": {
            "note_sample_35": metadata_from_hdf5.get('note_sample_35'),
            "note_sample_38": metadata_from_hdf5.get('note_sample_38')
        },

        "archival_information": {
            "doi": "10.24432/C56C76",
            "archive_date": datetime.now().strftime('%Y-%m-%d'),
            "depositor": "Jim Lu",
            "depositor_affiliation": "Rensselaer Polytechnic Institute",
            "depositor_email": "jimlo7760@gmail.com",
            "license": "CC0 1.0 Universal (Public Domain)",
            "recommended_citation": "Fisher, R.A. (1936). Iris Plants Database. UCI Machine Learning Repository. https://doi.org/10.24432/C56C76"
        },

        "keywords": [
            "iris",
            "classification",
            "machine learning",
            "pattern recognition",
            "Fisher"
        ],

        # Technical file information
        "file_information": {
            "filename": hdf5_path.name,
            "format": "HDF5",
            "size_kb": round(file_size_kb, 2),
            "compression": "gzip level 9",
            "md5_checksum": md5_hash,
            "sha256_checksum": sha256_hash
        },

        # HDF5 structure (what's actually in the file)
        "hdf5_structure": {
            "groups": [
                "original_data - from iris.data",
                "corrected_data - from bezdekIris.data",
                "summary_statistics - computed statistics",
                "by_class - data organized by species"
            ]
        }
    }

    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"Metadata JSON created: {output_path}")
    print(f"  Size: {output_path.stat().st_size / 1024:.2f} KB")
    print(f"  Sections: {len(metadata)}")
    print(f"\nMetadata respects original parse_iris_names structure")


def main():
    multi_city_hdf5_path = Path('outputs/multicity_demographic_energy_dataset.h5')
    multi_city_metadata_path = Path('outputs/.json')

    iris_hdf5_path = Path('outputs/iris_dataset.h5')
    iris_metadata_path = Path('outputs/iris_metadata.json')

    if not multi_city_hdf5_path.exists():
        print(f"Error: {multi_city_hdf5_path} not found.")
        return

    if not iris_hdf5_path.exists():
        print(f"Error: {iris_hdf5_path} not found.")
        return

    generate_metadata_json(multi_city_hdf5_path, multi_city_metadata_path)
    generate_metadata_json(iris_hdf5_path, iris_metadata_path)



if __name__ == '__main__':
    main()
"""
@version: Python 3.10
@created by: Jim Lu
@contact: jimlo7760@gmail.com
@time: 11/19/25
"""
