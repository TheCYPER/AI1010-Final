"""
CatBoost model implementation.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, Optional
from catboost import CatBoostClassifier

from .base_model import BaseModel


class CatBoostModel(BaseModel):
    """
    CatBoost classifier wrapper.
    
    Provides a clean interface for training CatBoost with our pipeline.
    CatBoost 的优势：
    - 自动处理类别特征（不需要手动编码）
    - 训练速度快
    - 对参数不敏感
    - 内置处理缺失值
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize CatBoost model.
        
        Args:
            config: Model configuration
        """
        super().__init__(config)
        self.num_classes_ = None
    
    def build_model(self, num_classes: int, **kwargs) -> CatBoostClassifier:
        """
        Build CatBoost classifier.
        
        Args:
            num_classes: Number of classes
            **kwargs: Override config parameters
        
        Returns:
            CatBoostClassifier instance
        """
        self.num_classes_ = num_classes
        
        # Merge config with kwargs
        params = {**self.config, **kwargs}
        
        # Ensure classes parameter is set
        params['classes_count'] = num_classes
        
        # Build model
        self.model_ = CatBoostClassifier(**params)
        
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
        Fit CatBoost model.
        
        Args:
            X: Training features
            y: Training labels
            eval_set: Evaluation set for early stopping (tuple of (X_val, y_val))
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
        
        # CatBoost uses different parameter names
        if eval_set is not None:
            # eval_set is list of tuples [(X_val, y_val)]
            fit_kwargs['eval_set'] = eval_set
        
        if sample_weight is not None:
            fit_kwargs['sample_weight'] = sample_weight
        
        # CatBoost uses early_stopping_rounds parameter
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
    
    def get_feature_importance(self, importance_type='FeatureImportance'):
        """
        Get feature importance.
        
        Args:
            importance_type: Type of importance 
                - 'FeatureImportance': Standard importance
                - 'PredictionValuesChange': How predictions change
                - 'LossFunctionChange': How loss changes
        
        Returns:
            Feature importance array
        """
        if not self.is_fitted_:
            raise ValueError("Model not fitted. Call fit() first.")
        
        return self.model_.feature_importances_


def create_catboost_model(config_dict: Dict[str, Any]) -> CatBoostModel:
    """
    Factory function to create CatBoost model from config.
    
    Args:
        config_dict: Configuration dictionary
    
    Returns:
        CatBoostModel instance
    """
    return CatBoostModel(config=config_dict)

