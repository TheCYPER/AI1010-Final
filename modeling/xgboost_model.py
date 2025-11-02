"""
XGBoost model implementation.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, Optional
from xgboost import XGBClassifier

from .base_model import BaseModel


class XGBoostModel(BaseModel):
    """
    XGBoost classifier wrapper.
    
    Provides a clean interface for training XGBoost with our pipeline.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize XGBoost model.
        
        Args:
            config: Model configuration
        """
        super().__init__(config)
        self.num_classes_ = None
    
    def build_model(self, num_classes: int, **kwargs) -> XGBClassifier:
        """
        Build XGBoost classifier.
        
        Args:
            num_classes: Number of classes
            **kwargs: Override config parameters
        
        Returns:
            XGBClassifier instance
        """
        self.num_classes_ = num_classes
        
        # Merge config with kwargs
        params = {**self.config, **kwargs}
        
        # Ensure num_class is set correctly
        params['num_class'] = num_classes
        
        # Build model
        self.model_ = XGBClassifier(**params)
        
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
        Fit XGBoost model.
        
        Args:
            X: Training features
            y: Training labels
            eval_set: Evaluation set for early stopping
            sample_weight: Sample weights
            early_stopping_rounds: Early stopping rounds
            verbose: Verbosity
            **kwargs: Additional fit parameters
        
        Returns:
            self
        """
        # Infer number of classes if not set
        if self.model_ is None:
            num_classes = len(np.unique(y))
            self.build_model(num_classes)
        
        # Prepare fit kwargs
        fit_kwargs = {
            'X': X,
            'y': y,
            'verbose': verbose
        }
        
        if eval_set is not None:
            fit_kwargs['eval_set'] = eval_set
        
        if sample_weight is not None:
            fit_kwargs['sample_weight'] = sample_weight
        
        if early_stopping_rounds is not None:
            fit_kwargs['early_stopping_rounds'] = early_stopping_rounds
        
        # Merge additional kwargs
        fit_kwargs.update(kwargs)
        
        # Fit
        self.model_.fit(**fit_kwargs)
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
        
        return self.model_.predict_proba(X)
    
    def get_feature_importance(self, importance_type='gain'):
        """
        Get feature importance.
        
        Args:
            importance_type: Type of importance ('gain', 'weight', 'cover')
        
        Returns:
            Feature importance array
        """
        if not self.is_fitted_:
            raise ValueError("Model not fitted. Call fit() first.")
        
        return self.model_.feature_importances_


def create_xgboost_model(config_dict: Dict[str, Any]) -> XGBoostModel:
    """
    Factory function to create XGBoost model from config.
    
    Args:
        config_dict: Configuration dictionary
    
    Returns:
        XGBoostModel instance
    """
    return XGBoostModel(config=config_dict)

