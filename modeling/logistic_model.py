"""
Logistic Regression model implementation using sklearn.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, Optional
from sklearn.linear_model import LogisticRegression

from .base_model import BaseModel


class LogisticRegressionModel(BaseModel):
    """
    Logistic Regression classifier wrapper.
    
    Logistic Regression is a linear model that can provide diversity
    in ensemble models by capturing linear relationships.
    
    Advantages:
    - Fully sklearn-compatible
    - Fast training and prediction
    - Good baseline model
    - Captures linear patterns (complements nonlinear models)
    - Interpretable
    
    Notes:
    - Requires scaled/normalized input (already handled by preprocessor)
    - Assumes linear relationships
    - Good regularization support (L1/L2)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Logistic Regression model.
        
        Args:
            config: Model configuration
        """
        super().__init__(config)
        self.num_classes_ = None
    
    def build_model(self, num_classes: int, **kwargs) -> LogisticRegression:
        """
        Build Logistic Regression classifier.
        
        Args:
            num_classes: Number of classes
            **kwargs: Override config parameters
        
        Returns:
            LogisticRegression instance
        """
        self.num_classes_ = num_classes
        
        # Merge config with kwargs
        params = {**self.config, **kwargs}
        
        # Build model
        self.model_ = LogisticRegression(**params)
        
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
        Fit Logistic Regression model.
        
        Args:
            X: Training features
            y: Training labels
            eval_set: Not used (Logistic Regression doesn't support early stopping)
            sample_weight: Sample weights (Logistic Regression supports this)
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
        
        # Logistic Regression supports sample_weight
        if sample_weight is not None:
            fit_kwargs['sample_weight'] = sample_weight
        
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
        Get feature importance (coefficients).
        
        Returns:
            Feature importance array (absolute values of coefficients)
        """
        if not self.is_fitted_:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Return absolute values of coefficients as importance
        # Average across all classes for multi-class
        if hasattr(self.model_, 'coef_'):
            return np.abs(self.model_.coef_).mean(axis=0)
        
        return None


def create_logistic_model(config_dict: Dict[str, Any]) -> LogisticRegressionModel:
    """
    Factory function to create Logistic Regression model from config.
    
    Args:
        config_dict: Configuration dictionary
    
    Returns:
        LogisticRegressionModel instance
    """
    return LogisticRegressionModel(config=config_dict)

