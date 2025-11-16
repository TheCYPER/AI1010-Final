"""
SVM (Support Vector Machine) model implementation using sklearn.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, Optional
from sklearn.svm import SVC

from .base_model import BaseModel


class SVMModel(BaseModel):
    """
    Support Vector Machine classifier wrapper.
    
    SVM is a powerful linear/non-linear classifier that can provide
    diversity in ensemble models.
    
    Advantages:
    - Fully sklearn-compatible
    - Can handle non-linear relationships (with RBF kernel)
    - Good generalization
    - Complements tree models well
    
    Notes:
    - Requires scaled/normalized input (already handled by preprocessor)
    - Training can be slow for large datasets
    - RBF kernel can capture complex patterns
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize SVM model.
        
        Args:
            config: Model configuration
        """
        super().__init__(config)
        self.num_classes_ = None
    
    def build_model(self, num_classes: int, **kwargs) -> SVC:
        """
        Build SVM classifier.
        
        Args:
            num_classes: Number of classes
            **kwargs: Override config parameters
        
        Returns:
            SVC instance
        """
        self.num_classes_ = num_classes
        
        # Merge config with kwargs
        params = {**self.config, **kwargs}
        
        # Ensure probability=True for ensemble compatibility (predict_proba)
        # Config already has probability=True, but we ensure it's set
        params['probability'] = True
        
        # Build model
        self.model_ = SVC(**params)
        
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
        Fit SVM model.
        
        Args:
            X: Training features
            y: Training labels
            eval_set: Not used (SVM doesn't support early stopping)
            sample_weight: Sample weights (SVM supports this)
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
        
        # SVM supports sample_weight
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
        
        Note: Linear SVM (kernel='linear') provides coefficients.
        
        Returns:
            Feature importance array (for linear kernel) or None
        """
        if not self.is_fitted_:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Only linear SVM has coefficients
        if hasattr(self.model_, 'coef_') and self.model_.kernel == 'linear':
            return np.abs(self.model_.coef_).mean(axis=0)
        
        return None

