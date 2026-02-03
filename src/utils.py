"""
Utility Functions
Helper functions used across the project
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os


def calculate_metrics(y_true, y_pred):
    """
    Calculate regression metrics
    
    Parameters:
    -----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
    
    Returns:
    --------
    dict : Dictionary of metrics
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'MAPE': mape
    }


def plot_actual_vs_predicted(y_true, y_pred, title="Actual vs Predicted", save_path=None):
    """
    Plot actual vs predicted values
    
    Parameters:
    -----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
    title : str
        Plot title
    save_path : str, optional
        Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5, edgecolors='k', linewidth=0.5)
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    plt.xlabel('Actual Values', fontsize=12)
    plt.ylabel('Predicted Values', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_residuals(y_true, y_pred, title="Residual Plot", save_path=None):
    """
    Plot residuals
    
    Parameters:
    -----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
    title : str
        Plot title
    save_path : str, optional
        Path to save the plot
    """
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Residual plot
    axes[0].scatter(y_pred, residuals, alpha=0.5, edgecolors='k', linewidth=0.5)
    axes[0].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[0].set_xlabel('Predicted Values', fontsize=12)
    axes[0].set_ylabel('Residuals', fontsize=12)
    axes[0].set_title('Residual Plot', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Residual distribution
    axes[1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('Residuals', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('Residual Distribution', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_feature_importance(importance_df, top_n=15, title="Feature Importance", save_path=None):
    """
    Plot feature importance
    
    Parameters:
    -----------
    importance_df : pd.DataFrame
        DataFrame with 'feature' and 'importance' columns
    top_n : int
        Number of top features to display
    title : str
        Plot title
    save_path : str, optional
        Path to save the plot
    """
    # Sort and select top N
    importance_df = importance_df.sort_values('importance', ascending=False).head(top_n)
    
    plt.figure(figsize=(10, 8))
    sns.barplot(data=importance_df, y='feature', x='importance', palette='viridis')
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def save_model(model, filename, models_dir='models/saved_models'):
    """
    Save model to disk
    
    Parameters:
    -----------
    model : sklearn model
        Trained model
    filename : str
        Filename for the model
    models_dir : str
        Directory to save models
    """
    os.makedirs(models_dir, exist_ok=True)
    filepath = os.path.join(models_dir, filename)
    joblib.dump(model, filepath)
    print(f"Model saved to: {filepath}")


def load_model(filename, models_dir='models/saved_models'):
    """
    Load model from disk
    
    Parameters:
    -----------
    filename : str
        Filename of the model
    models_dir : str
        Directory where models are saved
    
    Returns:
    --------
    model : sklearn model
        Loaded model
    """
    filepath = os.path.join(models_dir, filename)
    model = joblib.load(filepath)
    print(f"Model loaded from: {filepath}")
    return model


def format_currency(value):
    """
    Format value as currency (in $100,000s)
    
    Parameters:
    -----------
    value : float
        Value to format
    
    Returns:
    --------
    str : Formatted currency string
    """
    return f"${value * 100000:,.0f}"


def print_metrics(metrics, model_name="Model"):
    """
    Pretty print metrics
    
    Parameters:
    -----------
    metrics : dict
        Dictionary of metrics
    model_name : str
        Name of the model
    """
    print(f"\n{'='*50}")
    print(f"{model_name} Performance Metrics")
    print(f"{'='*50}")
    for metric, value in metrics.items():
        if metric == 'MAPE':
            print(f"{metric:10s}: {value:.2f}%")
        else:
            print(f"{metric:10s}: {value:.4f}")
    print(f"{'='*50}\n")
