"""
Data Loading Module
Handles loading and initial processing of the California Housing dataset
"""

import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
import os
import yaml


def load_config():
    """Load configuration from YAML file"""
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_california_housing_data(save_to_csv=True):
    """
    Load California Housing dataset from sklearn
    
    Parameters:
    -----------
    save_to_csv : bool
        Whether to save the dataset to CSV files
    
    Returns:
    --------
    pd.DataFrame : Complete dataset with features and target
    """
    print("Loading California Housing dataset...")
    
    # Fetch the dataset
    housing = fetch_california_housing(as_frame=True)
    
    # Combine features and target
    df = housing.frame
    
    # Add feature descriptions
    feature_descriptions = {
        'MedInc': 'Median income in block group',
        'HouseAge': 'Median house age in block group',
        'AveRooms': 'Average number of rooms per household',
        'AveBedrms': 'Average number of bedrooms per household',
        'Population': 'Block group population',
        'AveOccup': 'Average number of household members',
        'Latitude': 'Block group latitude',
        'Longitude': 'Block group longitude',
        'MedHouseVal': 'Median house value (in $100,000s)'
    }
    
    print(f"\nDataset loaded successfully!")
    print(f"Shape: {df.shape}")
    print(f"Features: {list(df.columns)}")
    
    # Save to CSV if requested
    if save_to_csv:
        config = load_config()
        raw_data_path = os.path.join(
            os.path.dirname(__file__), '..', 
            config['paths']['data_raw'], 
            'california_housing.csv'
        )
        os.makedirs(os.path.dirname(raw_data_path), exist_ok=True)
        df.to_csv(raw_data_path, index=False)
        print(f"\nData saved to: {raw_data_path}")
    
    return df, feature_descriptions


def get_data_summary(df):
    """
    Generate comprehensive data summary
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    
    Returns:
    --------
    dict : Summary statistics
    """
    summary = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'duplicates': df.duplicated().sum(),
        'memory_usage': df.memory_usage(deep=True).sum() / 1024**2,  # MB
        'numeric_summary': df.describe().to_dict()
    }
    
    return summary


def validate_data(df):
    """
    Validate data quality
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    
    Returns:
    --------
    dict : Validation results
    """
    validation_results = {
        'has_missing': df.isnull().any().any(),
        'has_duplicates': df.duplicated().any(),
        'has_infinite': np.isinf(df.select_dtypes(include=[np.number])).any().any(),
        'negative_values': (df.select_dtypes(include=[np.number]) < 0).any().to_dict(),
        'zero_variance': (df.var() == 0).to_dict()
    }
    
    return validation_results


if __name__ == "__main__":
    # Load and save data
    df, descriptions = load_california_housing_data(save_to_csv=True)
    
    # Print summary
    print("\n" + "="*50)
    print("DATA SUMMARY")
    print("="*50)
    summary = get_data_summary(df)
    print(f"\nShape: {summary['shape']}")
    print(f"Memory Usage: {summary['memory_usage']:.2f} MB")
    print(f"Missing Values: {summary['missing_values']}")
    print(f"Duplicates: {summary['duplicates']}")
    
    # Validate data
    print("\n" + "="*50)
    print("DATA VALIDATION")
    print("="*50)
    validation = validate_data(df)
    print(f"Has Missing Values: {validation['has_missing']}")
    print(f"Has Duplicates: {validation['has_duplicates']}")
    print(f"Has Infinite Values: {validation['has_infinite']}")
