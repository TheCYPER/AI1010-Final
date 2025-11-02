"""
Metrics and evaluation utilities.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = 'weighted'
) -> Dict[str, float]:
    """
    Compute classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        average: Averaging strategy for multi-class metrics
    
    Returns:
        Dictionary of metrics
    """
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
        'f1': f1_score(y_true, y_pred, average=average, zero_division=0)
    }


def evaluate_model(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Evaluate model on train and validation sets.
    
    Args:
        model: Trained model
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        verbose: Whether to print results
    
    Returns:
        Dictionary containing all metrics
    """
    # Predictions
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    
    # Metrics
    train_metrics = compute_classification_metrics(y_train, y_train_pred)
    val_metrics = compute_classification_metrics(y_val, y_val_pred)
    
    # Classification report
    val_report = classification_report(y_val, y_val_pred, output_dict=True)
    
    # Confusion matrix
    cm = confusion_matrix(y_val, y_val_pred)
    
    results = {
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'val_report': val_report,
        'confusion_matrix': cm.tolist()
    }
    
    if verbose:
        print("\n" + "="*70)
        print("MODEL EVALUATION")
        print("="*70)
        print(f"\nTrain Accuracy: {train_metrics['accuracy']:.4f}")
        print(f"Val Accuracy:   {val_metrics['accuracy']:.4f}")
        print(f"\nVal Precision:  {val_metrics['precision']:.4f}")
        print(f"Val Recall:     {val_metrics['recall']:.4f}")
        print(f"Val F1:         {val_metrics['f1']:.4f}")
        print("\nClassification Report (Validation):")
        print(classification_report(y_val, y_val_pred))
        print("="*70)
    
    return results


def aggregate_cv_metrics(metrics_list: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """
    Aggregate metrics from cross-validation folds.
    
    Args:
        metrics_list: List of metric dictionaries from each fold
    
    Returns:
        Dictionary with mean and std for each metric
    """
    aggregated = {}
    
    # Get all metric names
    metric_names = list(metrics_list[0].keys())
    
    for metric_name in metric_names:
        values = [m[metric_name] for m in metrics_list]
        aggregated[metric_name] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values))
        }
    
    return aggregated

