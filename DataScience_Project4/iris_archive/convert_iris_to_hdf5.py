#!/usr/bin/env python3
"""
Convert Iris Dataset to HDF5 Format

This script converts the classic Iris dataset files to HDF5 format,
preserving all data, metadata, and relationships.
"""

import h5py
import numpy as np
import pandas as pd
from pathlib import Path

def parse_iris_names(filepath):
    """
    Parse the iris.names file to extract metadata.
    
    Returns:
        dict: Dictionary containing parsed metadata
    """
    with open(filepath, 'r') as f:
        content = f.read()
    
    metadata = {
        'title': 'Iris Plants Database',
        'creator': 'R.A. Fisher',
        'donor': 'Michael Marshall (MARSHALL%PLU@io.arc.nasa.gov)',
        'date': 'July, 1988',
        'num_instances': 150,
        'num_attributes': 4,
        'num_classes': 3,
        'instances_per_class': 50,
        'attribute_names': [
            'sepal length in cm',
            'sepal width in cm', 
            'petal length in cm',
            'petal width in cm'
        ],
        'class_names': [
            'Iris-setosa',
            'Iris-versicolor', 
            'Iris-virginica'
        ],
        'missing_values': 'None',
        'description': 'This is perhaps the best known database to be found in the pattern recognition literature.',
        'note_sample_35': '4.9,3.1,1.5,0.2,Iris-setosa (error in fourth feature)',
        'note_sample_38': '4.9,3.6,1.4,0.1,Iris-setosa (errors in second and third features)',
        'class_distribution': '33.3% for each of 3 classes'
    }
    
    return metadata

def load_iris_data(filepath):
    """
    Load iris data from CSV file.
    
    Returns:
        tuple: (features array, class labels array)
    """
    # Read the CSV file
    df = pd.read_csv(filepath, header=None, 
                     names=['sepal_length', 'sepal_width', 'petal_length', 
                            'petal_width', 'class'])
    
    # Split features and labels
    features = df.iloc[:, :4].values.astype(np.float64)
    labels = df.iloc[:, 4].values.astype(str)
    
    return features, labels

def create_hdf5_file(output_path, project_dir):
    """
    Create HDF5 file with all iris dataset information.
    
    Args:
        output_path: Path where HDF5 file will be saved
        project_dir: Directory containing the iris data files
    """
    # Parse metadata
    print("Parsing metadata from iris.names...")
    metadata = parse_iris_names(project_dir / 'iris.names')
    
    # Load both data files
    print("Loading iris.data...")
    features_original, labels_original = load_iris_data(project_dir / 'iris.data')
    
    print("Loading bezdekIris.data (corrected version)...")
    features_bezdek, labels_bezdek = load_iris_data(project_dir / 'bezdekIris.data')
    
    # Create HDF5 file
    print(f"Creating HDF5 file: {output_path}")
    with h5py.File(output_path, 'w') as hdf:
        
        # Add root-level metadata
        hdf.attrs['title'] = metadata['title']
        hdf.attrs['creator'] = metadata['creator']
        hdf.attrs['donor'] = metadata['donor']
        hdf.attrs['date'] = metadata['date']
        hdf.attrs['description'] = metadata['description']
        hdf.attrs['num_instances'] = metadata['num_instances']
        hdf.attrs['num_attributes'] = metadata['num_attributes']
        hdf.attrs['num_classes'] = metadata['num_classes']
        hdf.attrs['instances_per_class'] = metadata['instances_per_class']
        hdf.attrs['missing_values'] = metadata['missing_values']
        hdf.attrs['class_distribution'] = metadata['class_distribution']
        
        # Store attribute names
        hdf.attrs['attribute_names'] = np.array(metadata['attribute_names'], dtype=h5py.string_dtype())
        hdf.attrs['class_names'] = np.array(metadata['class_names'], dtype=h5py.string_dtype())
        
        # Create group for original data (from iris.data)
        original_group = hdf.create_group('original_data')
        original_group.attrs['source_file'] = 'iris.data'
        original_group.attrs['note'] = 'Original data as published, contains known errors in samples 35 and 38'
        original_group.attrs['note_sample_35'] = metadata['note_sample_35']
        original_group.attrs['note_sample_38'] = metadata['note_sample_38']
        
        # Store original features with column names
        original_features = original_group.create_dataset(
            'features',
            data=features_original,
            dtype=np.float64,
            compression='gzip',
            compression_opts=9
        )
        original_features.attrs['column_0'] = 'sepal_length_cm'
        original_features.attrs['column_1'] = 'sepal_width_cm'
        original_features.attrs['column_2'] = 'petal_length_cm'
        original_features.attrs['column_3'] = 'petal_width_cm'
        original_features.attrs['units'] = 'centimeters'
        
        # Store original class labels
        original_labels = original_group.create_dataset(
            'class_labels',
            data=labels_original.astype('S20'),  # Fixed-length string
            compression='gzip',
            compression_opts=9
        )
        original_labels.attrs['description'] = 'Iris species class labels'
        
        # Create group for corrected data (from bezdekIris.data)
        corrected_group = hdf.create_group('corrected_data')
        corrected_group.attrs['source_file'] = 'bezdekIris.data'
        corrected_group.attrs['note'] = 'Corrected data with fixes for samples 35 and 38'
        
        # Store corrected features
        corrected_features = corrected_group.create_dataset(
            'features',
            data=features_bezdek,
            dtype=np.float64,
            compression='gzip',
            compression_opts=9
        )
        corrected_features.attrs['column_0'] = 'sepal_length_cm'
        corrected_features.attrs['column_1'] = 'sepal_width_cm'
        corrected_features.attrs['column_2'] = 'petal_length_cm'
        corrected_features.attrs['column_3'] = 'petal_width_cm'
        corrected_features.attrs['units'] = 'centimeters'
        
        # Store corrected class labels
        corrected_labels = corrected_group.create_dataset(
            'class_labels',
            data=labels_bezdek.astype('S20'),
            compression='gzip',
            compression_opts=9
        )
        corrected_labels.attrs['description'] = 'Iris species class labels'
        
        # Create summary statistics group
        stats_group = hdf.create_group('summary_statistics')
        
        # Calculate and store statistics for corrected data
        feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        stats_data = []
        
        for i, name in enumerate(feature_names):
            feature_col = features_bezdek[:, i]
            stats_data.append([
                np.min(feature_col),
                np.max(feature_col),
                np.mean(feature_col),
                np.std(feature_col, ddof=1)  # Sample standard deviation
            ])
        
        stats_array = np.array(stats_data, dtype=np.float64)
        stats_dataset = stats_group.create_dataset(
            'feature_statistics',
            data=stats_array
        )
        stats_dataset.attrs['row_0'] = 'sepal_length'
        stats_dataset.attrs['row_1'] = 'sepal_width'
        stats_dataset.attrs['row_2'] = 'petal_length'
        stats_dataset.attrs['row_3'] = 'petal_width'
        stats_dataset.attrs['column_0'] = 'min'
        stats_dataset.attrs['column_1'] = 'max'
        stats_dataset.attrs['column_2'] = 'mean'
        stats_dataset.attrs['column_3'] = 'std_dev'
        
        # Create class-separated datasets for easier analysis
        classes_group = hdf.create_group('by_class')
        
        for class_name in metadata['class_names']:
            # Filter corrected data by class
            class_mask = labels_bezdek == class_name
            class_features = features_bezdek[class_mask]
            
            class_subgroup = classes_group.create_group(class_name.replace('-', '_'))
            class_subgroup.attrs['class_name'] = class_name
            class_subgroup.attrs['num_samples'] = np.sum(class_mask)
            
            class_dataset = class_subgroup.create_dataset(
                'features',
                data=class_features,
                dtype=np.float64,
                compression='gzip',
                compression_opts=9
            )
            class_dataset.attrs['column_0'] = 'sepal_length_cm'
            class_dataset.attrs['column_1'] = 'sepal_width_cm'
            class_dataset.attrs['column_2'] = 'petal_length_cm'
            class_dataset.attrs['column_3'] = 'petal_width_cm'
    
    print(f"✓ HDF5 file created successfully: {output_path}")
    print(f"  - Original data: {len(features_original)} samples")
    print(f"  - Corrected data: {len(features_bezdek)} samples")
    print(f"  - Classes: {', '.join(metadata['class_names'])}")

def print_hdf5_structure(filepath):
    """
    Print the structure of the created HDF5 file.
    """
    print("\n" + "="*60)
    print("HDF5 File Structure")
    print("="*60)
    
    with h5py.File(filepath, 'r') as hdf:
        print(f"\nRoot attributes:")
        for key, value in hdf.attrs.items():
            print(f"  {key}: {value}")
        
        def print_group(name, obj):
            indent = "  " * (name.count('/'))
            if isinstance(obj, h5py.Group):
                print(f"\n{indent}Group: {name}")
                if obj.attrs:
                    print(f"{indent}  Attributes:")
                    for key, value in obj.attrs.items():
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
    
    print("\n" + "="*60)

def main():
    """Main execution function."""
    # Define paths
    project_dir = Path('iris')
    output_path = Path('outputs/iris_dataset.h5')
    
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create HDF5 file
    create_hdf5_file(output_path, project_dir)
    
    # Print structure
    print_hdf5_structure(output_path)

    print(f"  Output file: {output_path}")
    print(f"  File size: {output_path.stat().st_size / 1024:.2f} KB")

if __name__ == '__main__':
    main()
