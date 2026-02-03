"""
Model Training Module
Implements multiple regression models for house price prediction
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor, StackingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
import yaml
import os
import joblib


def load_config():
    """Load configuration from YAML file"""
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


class HousePriceModels:
    """Class to manage multiple regression models"""
    
    def __init__(self):
        self.models = {}
        self.trained_models = {}
        self.best_params = {}
        self.config = load_config()
    
    def initialize_models(self):
        """Initialize all models"""
        
        # Baseline models
        self.models['Linear Regression'] = LinearRegression()
        self.models['Ridge'] = Ridge(random_state=42)
        self.models['Lasso'] = Lasso(random_state=42)
        self.models['ElasticNet'] = ElasticNet(random_state=42)
        
        # Tree-based models
        self.models['Decision Tree'] = DecisionTreeRegressor(random_state=42)
        self.models['Random Forest'] = RandomForestRegressor(random_state=42, n_jobs=-1)
        self.models['Gradient Boosting'] = GradientBoostingRegressor(random_state=42)
        self.models['XGBoost'] = XGBRegressor(random_state=42, n_jobs=-1)
        self.models['LightGBM'] = LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1)
        
        print(f"Initialized {len(self.models)} models")
        return self.models
    
    def train_model(self, model_name, X_train, y_train, cv=5):
        """
        Train a single model
        
        Parameters:
        -----------
        model_name : str
            Name of the model
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training target
        cv : int
            Number of cross-validation folds
        
        Returns:
        --------
        dict : Training results
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        
        print(f"\nTraining {model_name}...")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Cross-validation
        cv_scores = cross_val_score(
            model, X_train, y_train, 
            cv=cv, 
            scoring='neg_root_mean_squared_error',
            n_jobs=-1
        )
        
        # Store trained model
        self.trained_models[model_name] = model
        
        results = {
            'model': model,
            'cv_scores': -cv_scores,  # Convert to positive
            'cv_mean': -cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
        
        print(f"CV RMSE: {results['cv_mean']:.4f} (+/- {results['cv_std']:.4f})")
        
        return results
    
    def train_all_models(self, X_train, y_train, cv=5):
        """
        Train all models
        
        Parameters:
        -----------
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training target
        cv : int
            Number of cross-validation folds
        
        Returns:
        --------
        dict : Results for all models
        """
        print("="*50)
        print("TRAINING ALL MODELS")
        print("="*50)
        
        results = {}
        
        for model_name in self.models.keys():
            results[model_name] = self.train_model(model_name, X_train, y_train, cv)
        
        print("\n" + "="*50)
        print("TRAINING COMPLETED")
        print("="*50)
        
        return results
    
    def hyperparameter_tuning(self, model_name, X_train, y_train, method='grid', cv=5):
        """
        Perform hyperparameter tuning
        
        Parameters:
        -----------
        model_name : str
            Name of the model
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training target
        method : str
            'grid' or 'random'
        cv : int
            Number of cross-validation folds
        
        Returns:
        --------
        dict : Tuning results
        """
        print(f"\nHyperparameter tuning for {model_name}...")
        
        # Define parameter grids
        param_grids = {
            'Ridge': {
                'alpha': [0.1, 1.0, 10.0, 100.0]
            },
            'Lasso': {
                'alpha': [0.001, 0.01, 0.1, 1.0]
            },
            'Random Forest': {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5]
            },
            'XGBoost': {
                'n_estimators': [100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1]
            },
            'LightGBM': {
                'n_estimators': [100, 200],
                'max_depth': [3, 5],
                'learning_rate': [0.01, 0.1]
            }
        }
        
        if model_name not in param_grids:
            print(f"No parameter grid defined for {model_name}")
            return None
        
        base_model = self.models[model_name]
        param_grid = param_grids[model_name]
        
        if method == 'grid':
            search = GridSearchCV(
                base_model, param_grid, cv=cv,
                scoring='neg_root_mean_squared_error',
                n_jobs=-1, verbose=1
            )
        else:
            search = RandomizedSearchCV(
                base_model, param_grid, cv=cv,
                scoring='neg_root_mean_squared_error',
                n_jobs=-1, verbose=1, n_iter=10
            )
        
        search.fit(X_train, y_train)
        
        # Store best model and parameters
        self.trained_models[f"{model_name} (Tuned)"] = search.best_estimator_
        self.best_params[model_name] = search.best_params_
        
        print(f"Best parameters: {search.best_params_}")
        print(f"Best CV RMSE: {-search.best_score_:.4f}")
        
        return {
            'best_model': search.best_estimator_,
            'best_params': search.best_params_,
            'best_score': -search.best_score_,
            'cv_results': search.cv_results_
        }
    
    def create_ensemble(self, X_train, y_train, ensemble_type='voting'):
        """
        Create ensemble model
        
        Parameters:
        -----------
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training target
        ensemble_type : str
            'voting' or 'stacking'
        
        Returns:
        --------
        model : Trained ensemble model
        """
        print(f"\nCreating {ensemble_type} ensemble...")
        
        # Select best performing models
        base_models = [
            ('rf', RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)),
            ('xgb', XGBRegressor(n_estimators=200, random_state=42, n_jobs=-1)),
            ('lgbm', LGBMRegressor(n_estimators=200, random_state=42, n_jobs=-1, verbose=-1))
        ]
        
        if ensemble_type == 'voting':
            ensemble = VotingRegressor(estimators=base_models, n_jobs=-1)
        else:
            ensemble = StackingRegressor(
                estimators=base_models,
                final_estimator=Ridge(),
                n_jobs=-1
            )
        
        ensemble.fit(X_train, y_train)
        
        self.trained_models[f'Ensemble ({ensemble_type.capitalize()})'] = ensemble
        
        print(f"{ensemble_type.capitalize()} ensemble created and trained")
        
        return ensemble
    
    def get_feature_importance(self, model_name, feature_names):
        """
        Get feature importance for tree-based models
        
        Parameters:
        -----------
        model_name : str
            Name of the model
        feature_names : list
            List of feature names
        
        Returns:
        --------
        pd.DataFrame : Feature importance dataframe
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not trained")
        
        model = self.trained_models[model_name]
        
        # Check if model has feature_importances_
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        else:
            print(f"{model_name} does not have feature importances")
            return None
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def save_model(self, model_name, filepath):
        """Save trained model"""
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not trained")
        
        joblib.dump(self.trained_models[model_name], filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath, model_name):
        """Load trained model"""
        model = joblib.load(filepath)
        self.trained_models[model_name] = model
        print(f"Model loaded from {filepath}")
        return model


if __name__ == "__main__":
    from data_loader import load_california_housing_data
    from preprocessing import preprocess_pipeline
    
    # Load and preprocess data
    df, _ = load_california_housing_data(save_to_csv=False)
    processed_data = preprocess_pipeline(df)
    
    # Initialize models
    model_manager = HousePriceModels()
    model_manager.initialize_models()
    
    # Train all models
    results = model_manager.train_all_models(
        processed_data['X_train_scaled'],
        processed_data['y_train']
    )
    
    # Print summary
    print("\n" + "="*50)
    print("MODEL COMPARISON (CV RMSE)")
    print("="*50)
    for model_name, result in results.items():
        print(f"{model_name:20s}: {result['cv_mean']:.4f} (+/- {result['cv_std']:.4f})")
