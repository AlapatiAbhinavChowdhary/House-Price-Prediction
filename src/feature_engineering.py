"""
Feature Engineering Module
Creates advanced features for improved model performance
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import PolynomialFeatures
import yaml
import os


def load_config():
    """Load configuration from YAML file"""
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_interaction_features(df):
    """
    Create interaction features
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    
    Returns:
    --------
    pd.DataFrame : Dataframe with new features
    """
    df_new = df.copy()
    
    # Rooms per household
    df_new['RoomsPerHousehold'] = df_new['AveRooms'] * df_new['AveOccup']
    
    # Bedrooms ratio
    df_new['BedroomsRatio'] = df_new['AveBedrms'] / df_new['AveRooms']
    
    # Population per household
    df_new['PopulationPerHousehold'] = df_new['Population'] / df_new['AveOccup']
    
    # Income per room
    df_new['IncomePerRoom'] = df_new['MedInc'] / df_new['AveRooms']
    
    # Income per person
    df_new['IncomePerPerson'] = df_new['MedInc'] / df_new['AveOccup']
    
    # House age categories
    df_new['HouseAgeCategory'] = pd.cut(
        df_new['HouseAge'], 
        bins=[0, 10, 20, 30, 40, 100],
        labels=['New', 'Recent', 'Moderate', 'Old', 'Very Old']
    )
    
    # Income categories
    df_new['IncomeCategory'] = pd.cut(
        df_new['MedInc'],
        bins=[0, 2, 4, 6, 8, 100],
        labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
    )
    
    print(f"Created {len(df_new.columns) - len(df.columns)} interaction features")
    
    return df_new


def create_location_clusters(df, n_clusters=5):
    """
    Create location-based clusters using K-means
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    n_clusters : int
        Number of clusters
    
    Returns:
    --------
    pd.DataFrame : Dataframe with cluster labels
    """
    df_new = df.copy()
    
    # Extract location features
    location_features = df_new[['Latitude', 'Longitude']].values
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df_new['LocationCluster'] = kmeans.fit_predict(location_features)
    
    # Calculate cluster statistics
    cluster_stats = df_new.groupby('LocationCluster').agg({
        'MedHouseVal': 'mean' if 'MedHouseVal' in df_new.columns else 'count',
        'MedInc': 'mean',
        'Population': 'mean'
    }).round(2)
    
    print(f"\nCreated {n_clusters} location clusters")
    print("\nCluster Statistics:")
    print(cluster_stats)
    
    return df_new, kmeans


def create_distance_features(df, center_lat=37.7749, center_lon=-122.4194):
    """
    Create distance features from a central point (San Francisco)
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    center_lat : float
        Latitude of center point
    center_lon : float
        Longitude of center point
    
    Returns:
    --------
    pd.DataFrame : Dataframe with distance features
    """
    df_new = df.copy()
    
    # Haversine distance formula
    def haversine_distance(lat1, lon1, lat2, lon2):
        R = 6371  # Earth's radius in kilometers
        
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        return R * c
    
    # Distance from center
    df_new['DistanceFromCenter'] = haversine_distance(
        df_new['Latitude'], df_new['Longitude'],
        center_lat, center_lon
    )
    
    # Distance from coast (approximate using longitude)
    df_new['DistanceFromCoast'] = np.abs(df_new['Longitude'] + 120)
    
    print("Created distance-based features")
    
    return df_new


def create_polynomial_features(df, degree=2, include_bias=False):
    """
    Create polynomial features
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    degree : int
        Polynomial degree
    include_bias : bool
        Whether to include bias term
    
    Returns:
    --------
    pd.DataFrame : Dataframe with polynomial features
    tuple : (transformed_df, poly_transformer, feature_names)
    """
    # Select only numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Exclude target if present
    if 'MedHouseVal' in numeric_cols:
        numeric_cols.remove('MedHouseVal')
    
    # Limit to key features to avoid explosion
    key_features = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms']
    key_features = [f for f in key_features if f in numeric_cols]
    
    poly = PolynomialFeatures(degree=degree, include_bias=include_bias)
    poly_features = poly.fit_transform(df[key_features])
    
    # Get feature names
    feature_names = poly.get_feature_names_out(key_features)
    
    # Create dataframe
    poly_df = pd.DataFrame(
        poly_features,
        columns=feature_names,
        index=df.index
    )
    
    # Combine with original dataframe (excluding original key features to avoid duplication)
    df_combined = pd.concat([
        df.drop(columns=key_features),
        poly_df
    ], axis=1)
    
    print(f"Created polynomial features (degree={degree})")
    print(f"New feature count: {len(feature_names)}")
    
    return df_combined, poly, feature_names


def engineer_features(df, use_clusters=True, use_distance=True, 
                     use_polynomial=False, n_clusters=5):
    """
    Complete feature engineering pipeline
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    use_clusters : bool
        Whether to create location clusters
    use_distance : bool
        Whether to create distance features
    use_polynomial : bool
        Whether to create polynomial features
    n_clusters : int
        Number of location clusters
    
    Returns:
    --------
    dict : Dictionary containing engineered data and artifacts
    """
    print("="*50)
    print("FEATURE ENGINEERING PIPELINE")
    print("="*50)
    
    df_engineered = df.copy()
    artifacts = {}
    
    # Create interaction features
    df_engineered = create_interaction_features(df_engineered)
    
    # Create location clusters
    if use_clusters:
        df_engineered, kmeans = create_location_clusters(df_engineered, n_clusters)
        artifacts['kmeans'] = kmeans
    
    # Create distance features
    if use_distance:
        config = load_config()
        center_lat = config['features']['distance_center_lat']
        center_lon = config['features']['distance_center_lon']
        df_engineered = create_distance_features(df_engineered, center_lat, center_lon)
    
    # Create polynomial features (optional - can increase dimensionality significantly)
    if use_polynomial:
        config = load_config()
        degree = config['features']['polynomial_degree']
        df_engineered, poly, poly_names = create_polynomial_features(df_engineered, degree)
        artifacts['polynomial'] = poly
        artifacts['poly_feature_names'] = poly_names
    
    # Convert categorical features to numeric
    categorical_cols = df_engineered.select_dtypes(include=['category', 'object']).columns
    if len(categorical_cols) > 0:
        df_engineered = pd.get_dummies(df_engineered, columns=categorical_cols, drop_first=True)
        print(f"\nEncoded {len(categorical_cols)} categorical features")
    
    print("\n" + "="*50)
    print("FEATURE ENGINEERING COMPLETED")
    print("="*50)
    print(f"Original features: {len(df.columns)}")
    print(f"Engineered features: {len(df_engineered.columns)}")
    print(f"New features added: {len(df_engineered.columns) - len(df.columns)}")
    
    artifacts['engineered_df'] = df_engineered
    artifacts['feature_names'] = list(df_engineered.columns)
    
    return artifacts


if __name__ == "__main__":
    from data_loader import load_california_housing_data
    
    # Load data
    df, _ = load_california_housing_data(save_to_csv=False)
    
    # Run feature engineering
    results = engineer_features(df, use_polynomial=False)
    
    print(f"\nFinal shape: {results['engineered_df'].shape}")
    print(f"\nNew features: {results['feature_names']}")
