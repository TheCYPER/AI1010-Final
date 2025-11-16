"""
MLP (Multi-Layer Perceptron) model implementation using sklearn.

MLPClassifier is a fully sklearn-compatible neural network model.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, Optional
from sklearn.neural_network import MLPClassifier

from .base_model import BaseModel


class MLPModel(BaseModel):
    """
    Multi-Layer Perceptron classifier wrapper.
    
    Uses sklearn's MLPClassifier, which is a fully sklearn-compatible
    neural network implementation.
    
    Advantages:
    - Fully sklearn-compatible (works with pipelines, voting, stacking)
    - No GPU required (CPU-based)
    - Good for tabular data
    - Supports early stopping
    - Can handle multi-class classification
    
    Notes:
    - Requires scaled/normalized input (already handled by preprocessor)
    - Training can be slower than tree models for large datasets
    - May need tuning of hidden layers and learning rate
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize MLP model.
        
        Args:
            config: Model configuration
        """
        super().__init__(config)
        self.num_classes_ = None
    
    def build_model(self, num_classes: int, **kwargs) -> MLPClassifier:
        """
        Build MLP classifier.
        
        Args:
            num_classes: Number of classes
            **kwargs: Override config parameters
        
        Returns:
            MLPClassifier instance
        """
        self.num_classes_ = num_classes
        
        # Merge config with kwargs
        params = {**self.config, **kwargs}
        
        # Build model
        self.model_ = MLPClassifier(**params)
        
        return self.model_
    
    def fit(
        self,
        X,
        y,
        eval_set=None,
        sample_weight=None,
        early_stopping_rounds=None,
        verbose=False,
        **kwargs
    ):
        """
        Fit MLP model.
        
        Args:
            X: Training features
            y: Training labels
            eval_set: Evaluation set for early stopping (list of tuples)
            sample_weight: Sample weights
            early_stopping_rounds: Early stopping rounds (validation_fraction + n_iter_no_change)
            verbose: Verbosity
            **kwargs: Additional fit parameters
        
        Returns:
            self
        """
        # Infer number of classes if not set
        if self.model_ is None:
            num_classes = len(np.unique(y))
            self.build_model(num_classes)
        
        # Convert to numpy if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        
        # Prepare fit kwargs
        fit_kwargs = {}
        
        # MLPClassifier supports sample_weight
        if sample_weight is not None:
            fit_kwargs['sample_weight'] = sample_weight
        
        # MLPClassifier has built-in early stopping via validation_fraction
        # If eval_set is provided, we can use it for validation
        if eval_set is not None and early_stopping_rounds is not None:
            # MLPClassifier uses validation_fraction and n_iter_no_change for early stopping
            # We need to set these in the model initialization, not in fit
            # But we can still use eval_set to monitor performance
            pass
        
        # Merge additional kwargs
        fit_kwargs.update(kwargs)
        
        # Fit
        self.model_.fit(X, y, **fit_kwargs)
        self.is_fitted_ = True
        
        return self
    
    def predict(self, X):
        """
        Make predictions.
        
        Args:
            X: Features
        
        Returns:
            Predicted class labels
        """
        if not self.is_fitted_:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Convert to numpy if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        return self.model_.predict(X)
    
    def predict_proba(self, X):
        """
        Predict class probabilities.
        
        Args:
            X: Features
        
        Returns:
            Class probabilities
        """
        if not self.is_fitted_:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Convert to numpy if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        return self.model_.predict_proba(X)
    
    def get_feature_importance(self):
        """
        Get feature importance if available.
        
        Note: MLPClassifier doesn't provide feature importance directly.
        This is a limitation of neural networks.
        
        Returns:
            None (MLP doesn't have feature importance)
        """
        return None


def create_mlp_model(config_dict: Dict[str, Any]) -> MLPModel:
    """
    Factory function to create MLP model from config.
    
    Args:
        config_dict: Configuration dictionary
    
    Returns:
        MLPModel instance
    """
    return MLPModel(config=config_dict)

