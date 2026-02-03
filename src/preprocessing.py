"""
Data Preprocessing Module
Handles data cleaning, outlier detection, and preprocessing pipeline
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
import yaml
import os


def load_config():
    """Load configuration from YAML file"""
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def detect_outliers_iqr(df, columns, threshold=1.5):
    """
    Detect outliers using IQR method
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    columns : list
        Columns to check for outliers
    threshold : float
        IQR multiplier (default: 1.5)
    
    Returns:
    --------
    pd.DataFrame : Boolean dataframe indicating outliers
    """
    outliers = pd.DataFrame(index=df.index)
    
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        outliers[col] = (df[col] < lower_bound) | (df[col] > upper_bound)
    
    return outliers


def handle_outliers(df, method='clip', threshold=1.5):
    """
    Handle outliers in the dataset
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    method : str
        Method to handle outliers ('clip', 'remove', or 'keep')
    threshold : float
        IQR multiplier
    
    Returns:
    --------
    pd.DataFrame : Processed dataframe
    """
    df_processed = df.copy()
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
    
    # Exclude target variable if present
    if 'MedHouseVal' in numeric_cols:
        numeric_cols.remove('MedHouseVal')
    
    if method == 'clip':
        for col in numeric_cols:
            Q1 = df_processed[col].quantile(0.25)
            Q3 = df_processed[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            df_processed[col] = df_processed[col].clip(lower_bound, upper_bound)
        
        print(f"Outliers clipped for {len(numeric_cols)} columns")
    
    elif method == 'remove':
        outliers = detect_outliers_iqr(df_processed, numeric_cols, threshold)
        outlier_mask = outliers.any(axis=1)
        df_processed = df_processed[~outlier_mask]
        print(f"Removed {outlier_mask.sum()} rows with outliers")
    
    else:
        print("Keeping outliers")
    
    return df_processed


def split_data(df, target_col='MedHouseVal', test_size=0.2, random_state=42):
    """
    Split data into train and test sets
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    target_col : str
        Target column name
    test_size : float
        Proportion of test set
    random_state : int
        Random seed
    
    Returns:
    --------
    tuple : X_train, X_test, y_train, y_test
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"\nData split completed:")
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test


def scale_features(X_train, X_test, method='standard'):
    """
    Scale features using specified method
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    X_test : pd.DataFrame
        Test features
    method : str
        Scaling method ('standard' or 'robust')
    
    Returns:
    --------
    tuple : X_train_scaled, X_test_scaled, scaler
    """
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError("Method must be 'standard' or 'robust'")
    
    # Fit on training data only
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )
    
    print(f"\nFeatures scaled using {method} scaler")
    
    return X_train_scaled, X_test_scaled, scaler


def preprocess_pipeline(df, outlier_method='clip', scale_method='robust', 
                       test_size=0.2, random_state=42):
    """
    Complete preprocessing pipeline
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    outlier_method : str
        Method to handle outliers
    scale_method : str
        Scaling method
    test_size : float
        Test set proportion
    random_state : int
        Random seed
    
    Returns:
    --------
    dict : Dictionary containing processed data and artifacts
    """
    print("="*50)
    print("PREPROCESSING PIPELINE")
    print("="*50)
    
    # Handle outliers
    df_processed = handle_outliers(df, method=outlier_method)
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(
        df_processed, test_size=test_size, random_state=random_state
    )
    
    # Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(
        X_train, X_test, method=scale_method
    )
    
    print("\n" + "="*50)
    print("PREPROCESSING COMPLETED")
    print("="*50)
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'X_train_scaled': X_train_scaled,
        'X_test_scaled': X_test_scaled,
        'scaler': scaler
    }


if __name__ == "__main__":
    from data_loader import load_california_housing_data
    
    # Load data
    df, _ = load_california_housing_data(save_to_csv=False)
    
    # Run preprocessing pipeline
    processed_data = preprocess_pipeline(df)
    
    print(f"\nProcessed data keys: {list(processed_data.keys())}")
