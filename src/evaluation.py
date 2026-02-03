"""
Model Evaluation Module
Comprehensive evaluation metrics and visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import learning_curve
import os


def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Evaluate model performance
    
    Parameters:
    -----------
    model : sklearn model
        Trained model
    X_test : pd.DataFrame
        Test features
    y_test : pd.Series
        Test target
    model_name : str
        Name of the model
    
    Returns:
    --------
    dict : Evaluation metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    
    # Adjusted R2
    n = len(y_test)
    p = X_test.shape[1]
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    
    metrics = {
        'Model': model_name,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'Adjusted R2': adj_r2,
        'MAPE': mape
    }
    
    return metrics, y_pred


def evaluate_all_models(trained_models, X_test, y_test):
    """
    Evaluate all trained models
    
    Parameters:
    -----------
    trained_models : dict
        Dictionary of trained models
    X_test : pd.DataFrame
        Test features
    y_test : pd.Series
        Test target
    
    Returns:
    --------
    pd.DataFrame : Comparison dataframe
    """
    results = []
    predictions = {}
    
    for model_name, model in trained_models.items():
        metrics, y_pred = evaluate_model(model, X_test, y_test, model_name)
        results.append(metrics)
        predictions[model_name] = y_pred
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('RMSE')
    
    return results_df, predictions


def plot_model_comparison(results_df, save_path=None):
    """
    Plot model comparison
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        Results dataframe
    save_path : str, optional
        Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # RMSE comparison
    axes[0, 0].barh(results_df['Model'], results_df['RMSE'], color='skyblue', edgecolor='black')
    axes[0, 0].set_xlabel('RMSE', fontsize=12)
    axes[0, 0].set_title('Root Mean Squared Error', fontsize=14, fontweight='bold')
    axes[0, 0].invert_yaxis()
    axes[0, 0].grid(axis='x', alpha=0.3)
    
    # MAE comparison
    axes[0, 1].barh(results_df['Model'], results_df['MAE'], color='lightcoral', edgecolor='black')
    axes[0, 1].set_xlabel('MAE', fontsize=12)
    axes[0, 1].set_title('Mean Absolute Error', fontsize=14, fontweight='bold')
    axes[0, 1].invert_yaxis()
    axes[0, 1].grid(axis='x', alpha=0.3)
    
    # R2 comparison
    axes[1, 0].barh(results_df['Model'], results_df['R2'], color='lightgreen', edgecolor='black')
    axes[1, 0].set_xlabel('R² Score', fontsize=12)
    axes[1, 0].set_title('R² Score', fontsize=14, fontweight='bold')
    axes[1, 0].invert_yaxis()
    axes[1, 0].grid(axis='x', alpha=0.3)
    
    # MAPE comparison
    axes[1, 1].barh(results_df['Model'], results_df['MAPE'], color='plum', edgecolor='black')
    axes[1, 1].set_xlabel('MAPE (%)', fontsize=12)
    axes[1, 1].set_title('Mean Absolute Percentage Error', fontsize=14, fontweight='bold')
    axes[1, 1].invert_yaxis()
    axes[1, 1].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_predictions(y_test, predictions_dict, top_n=3, save_path=None):
    """
    Plot predictions for top N models
    
    Parameters:
    -----------
    y_test : pd.Series
        Test target
    predictions_dict : dict
        Dictionary of predictions
    top_n : int
        Number of top models to plot
    save_path : str, optional
        Path to save the plot
    """
    n_models = min(top_n, len(predictions_dict))
    fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
    
    if n_models == 1:
        axes = [axes]
    
    for idx, (model_name, y_pred) in enumerate(list(predictions_dict.items())[:n_models]):
        ax = axes[idx]
        
        ax.scatter(y_test, y_pred, alpha=0.5, edgecolors='k', linewidth=0.5)
        
        # Perfect prediction line
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        ax.set_xlabel('Actual Values', fontsize=12)
        ax.set_ylabel('Predicted Values', fontsize=12)
        ax.set_title(f'{model_name}', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_residuals_analysis(y_test, y_pred, model_name="Model", save_path=None):
    """
    Comprehensive residual analysis
    
    Parameters:
    -----------
    y_test : pd.Series
        Test target
    y_pred : array-like
        Predictions
    model_name : str
        Name of the model
    save_path : str, optional
        Path to save the plot
    """
    residuals = y_test - y_pred
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Residual plot
    axes[0, 0].scatter(y_pred, residuals, alpha=0.5, edgecolors='k', linewidth=0.5)
    axes[0, 0].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[0, 0].set_xlabel('Predicted Values', fontsize=12)
    axes[0, 0].set_ylabel('Residuals', fontsize=12)
    axes[0, 0].set_title('Residual Plot', fontsize=14, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Residual distribution
    axes[0, 1].hist(residuals, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
    axes[0, 1].set_xlabel('Residuals', fontsize=12)
    axes[0, 1].set_ylabel('Frequency', fontsize=12)
    axes[0, 1].set_title('Residual Distribution', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Q-Q plot
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot', fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Scale-Location plot
    standardized_residuals = (residuals - residuals.mean()) / residuals.std()
    axes[1, 1].scatter(y_pred, np.sqrt(np.abs(standardized_residuals)), alpha=0.5, edgecolors='k', linewidth=0.5)
    axes[1, 1].set_xlabel('Predicted Values', fontsize=12)
    axes[1, 1].set_ylabel('√|Standardized Residuals|', fontsize=12)
    axes[1, 1].set_title('Scale-Location Plot', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    fig.suptitle(f'Residual Analysis - {model_name}', fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_learning_curve(model, X, y, model_name="Model", cv=5, save_path=None):
    """
    Plot learning curve
    
    Parameters:
    -----------
    model : sklearn model
        Model to evaluate
    X : pd.DataFrame
        Features
    y : pd.Series
        Target
    model_name : str
        Name of the model
    cv : int
        Number of cross-validation folds
    save_path : str, optional
        Path to save the plot
    """
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=cv, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='neg_root_mean_squared_error'
    )
    
    train_scores_mean = -np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    val_scores_mean = -np.mean(val_scores, axis=1)
    val_scores_std = np.std(val_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training score')
    plt.plot(train_sizes, val_scores_mean, 'o-', color='g', label='Cross-validation score')
    
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color='r')
    plt.fill_between(train_sizes, val_scores_mean - val_scores_std,
                     val_scores_mean + val_scores_std, alpha=0.1, color='g')
    
    plt.xlabel('Training Set Size', fontsize=12)
    plt.ylabel('RMSE', fontsize=12)
    plt.title(f'Learning Curve - {model_name}', fontsize=14, fontweight='bold')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def create_evaluation_report(results_df, best_model_name, y_test, y_pred):
    """
    Create comprehensive evaluation report
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        Results dataframe
    best_model_name : str
        Name of the best model
    y_test : pd.Series
        Test target
    y_pred : array-like
        Predictions
    
    Returns:
    --------
    str : Formatted report
    """
    report = []
    report.append("="*70)
    report.append("MODEL EVALUATION REPORT")
    report.append("="*70)
    report.append("")
    
    report.append("TOP 5 MODELS:")
    report.append("-"*70)
    for idx, row in results_df.head(5).iterrows():
        report.append(f"{idx+1}. {row['Model']}")
        report.append(f"   RMSE: {row['RMSE']:.4f} | MAE: {row['MAE']:.4f} | R²: {row['R2']:.4f} | MAPE: {row['MAPE']:.2f}%")
        report.append("")
    
    report.append("="*70)
    report.append(f"BEST MODEL: {best_model_name}")
    report.append("="*70)
    
    best_metrics = results_df[results_df['Model'] == best_model_name].iloc[0]
    report.append(f"RMSE:         {best_metrics['RMSE']:.4f}")
    report.append(f"MAE:          {best_metrics['MAE']:.4f}")
    report.append(f"R² Score:     {best_metrics['R2']:.4f}")
    report.append(f"Adjusted R²:  {best_metrics['Adjusted R2']:.4f}")
    report.append(f"MAPE:         {best_metrics['MAPE']:.2f}%")
    report.append("")
    
    # Prediction statistics
    errors = np.abs(y_test - y_pred)
    report.append("PREDICTION STATISTICS:")
    report.append(f"Mean Error:   ${errors.mean() * 100000:,.0f}")
    report.append(f"Median Error: ${np.median(errors) * 100000:,.0f}")
    report.append(f"Max Error:    ${errors.max() * 100000:,.0f}")
    report.append(f"Min Error:    ${errors.min() * 100000:,.0f}")
    
    report.append("="*70)
    
    return "\n".join(report)


if __name__ == "__main__":
    print("Evaluation module loaded successfully")
