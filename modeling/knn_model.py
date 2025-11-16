"""
KNN (K-Nearest Neighbors) model implementation using sklearn.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, Optional
from sklearn.neighbors import KNeighborsClassifier

from .base_model import BaseModel


class KNNModel(BaseModel):
    """
    K-Nearest Neighbors classifier wrapper.
    
    KNN is a simple, instance-based learning algorithm that can provide
    diversity in ensemble models.
    
    Advantages:
    - Fully sklearn-compatible
    - Fast training (lazy learning)
    - Good for capturing local patterns
    - Complements tree models well
    
    Notes:
    - Requires scaled/normalized input (already handled by preprocessor)
    - Prediction can be slow for large datasets
    - Sensitive to irrelevant features
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize KNN model.
        
        Args:
            config: Model configuration
        """
        super().__init__(config)
        self.num_classes_ = None
    
    def build_model(self, num_classes: int, **kwargs) -> KNeighborsClassifier:
        """
        Build KNN classifier.
        
        Args:
            num_classes: Number of classes (unused, KNN doesn't need this)
            **kwargs: Override config parameters
        
        Returns:
            KNeighborsClassifier instance
        """
        self.num_classes_ = num_classes
        
        # Merge config with kwargs
        params = {**self.config, **kwargs}
        
        # Build model
        self.model_ = KNeighborsClassifier(**params)
        
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
        Fit KNN model.
        
        Args:
            X: Training features
            y: Training labels
            eval_set: Not used (KNN doesn't support early stopping)
            sample_weight: Sample weights (KNN supports this)
            early_stopping_rounds: Not used
            verbose: Verbosity
            **kwargs: Additional fit parameters
        
        Returns:
            self
        """
        # Convert to numpy if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        
        # Build model if not already built
        if self.model_ is None:
            num_classes = len(np.unique(y))
            self.build_model(num_classes)
        
        # Prepare fit kwargs
        fit_kwargs = {}
        
        # Note: KNN doesn't support sample_weight in fit() method
        # Sample weights can be handled through the 'weights' parameter in KNeighborsClassifier
        # which weights neighbors by distance, not training samples
        
        # Merge additional kwargs (but remove sample_weight as KNN doesn't support it)
        kwargs_clean = {k: v for k, v in kwargs.items() if k != 'sample_weight'}
        fit_kwargs.update(kwargs_clean)
        
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
        
        Note: KNN doesn't provide feature importance.
        
        Returns:
            None
        """
        return None


def create_knn_model(config_dict: Dict[str, Any]) -> KNNModel:
    """
    Factory function to create KNN model from config.
    
    Args:
        config_dict: Configuration dictionary
    
    Returns:
        KNNModel instance
    """
    return KNNModel(config=config_dict)

