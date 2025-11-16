"""
Naive Bayes model implementation using sklearn.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, Optional
from sklearn.naive_bayes import GaussianNB

from .base_model import BaseModel


class NaiveBayesModel(BaseModel):
    """
    Naive Bayes classifier wrapper.
    
    Naive Bayes is a simple probabilistic classifier that can provide
    diversity in ensemble models.
    
    Advantages:
    - Fully sklearn-compatible
    - Very fast training and prediction
    - Good baseline model
    - Probabilistic predictions
    - Complements other models well
    
    Notes:
    - Assumes features are independent (naive assumption)
    - Works well with scaled/normalized features
    - Good for high-dimensional data
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Naive Bayes model.
        
        Args:
            config: Model configuration
        """
        super().__init__(config)
        self.num_classes_ = None
    
    def build_model(self, num_classes: int, **kwargs) -> GaussianNB:
        """
        Build Naive Bayes classifier.
        
        Args:
            num_classes: Number of classes
            **kwargs: Override config parameters
        
        Returns:
            GaussianNB instance
        """
        self.num_classes_ = num_classes
        
        # Merge config with kwargs
        params = {**self.config, **kwargs}
        
        # Build model
        self.model_ = GaussianNB(**params)
        
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
        Fit Naive Bayes model.
        
        Args:
            X: Training features
            y: Training labels
            eval_set: Not used
            sample_weight: Sample weights (GaussianNB supports this)
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
        
        # GaussianNB supports sample_weight
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
        Get feature importance if available.
        
        Note: Naive Bayes doesn't provide feature importance.
        
        Returns:
            None
        """
        return None

