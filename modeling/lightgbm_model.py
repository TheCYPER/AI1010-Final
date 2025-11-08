"""
LightGBM model implementation.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, Optional
import lightgbm as lgb

from .base_model import BaseModel


class LightGBMModel(BaseModel):
    """
    LightGBM classifier wrapper.
    
    Provides a clean interface for training LightGBM with our pipeline.
    LightGBM 的优势：
    - 训练速度最快
    - 内存占用小
    - 准确率高
    - 支持类别特征
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize LightGBM model.
        
        Args:
            config: Model configuration
        """
        super().__init__(config)
        self.num_classes_ = None
    
    def build_model(self, num_classes: int, **kwargs) -> lgb.LGBMClassifier:
        """
        Build LightGBM classifier.
        
        Args:
            num_classes: Number of classes
            **kwargs: Override config parameters
        
        Returns:
            LGBMClassifier instance
        """
        self.num_classes_ = num_classes
        
        # Merge config with kwargs
        params = {**self.config, **kwargs}
        
        # Ensure num_class is set correctly
        params['num_class'] = num_classes
        
        # Build model
        self.model_ = lgb.LGBMClassifier(**params)
        
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
        Fit LightGBM model.
        
        Args:
            X: Training features
            y: Training labels
            eval_set: Evaluation set for early stopping (list of tuples)
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
        }
        
        # LightGBM uses eval_set parameter
        if eval_set is not None:
            fit_kwargs['eval_set'] = eval_set
        
        if sample_weight is not None:
            fit_kwargs['sample_weight'] = sample_weight
        
        # LightGBM uses callbacks for early stopping
        if early_stopping_rounds is not None:
            fit_kwargs['callbacks'] = [
                lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=verbose)
            ]
        
        # Control verbosity
        if not verbose:
            fit_kwargs['callbacks'] = fit_kwargs.get('callbacks', []) + [
                lgb.log_evaluation(period=0)  # Suppress output
            ]
        
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
            importance_type: Type of importance
                - 'gain': Total gain of splits using the feature
                - 'split': Number of times feature is used in a split
        
        Returns:
            Feature importance array
        """
        if not self.is_fitted_:
            raise ValueError("Model not fitted. Call fit() first.")
        
        return self.model_.feature_importances_


def create_lightgbm_model(config_dict: Dict[str, Any]) -> LightGBMModel:
    """
    Factory function to create LightGBM model from config.
    
    Args:
        config_dict: Configuration dictionary
    
    Returns:
        LightGBMModel instance
    """
    return LightGBMModel(config=config_dict)

