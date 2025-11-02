"""
Base model class for all models.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import numpy as np


class BaseModel(ABC):
    """
    Abstract base class for all models.
    
    Provides a consistent interface for training and prediction.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize base model.
        
        Args:
            config: Model configuration dictionary
        """
        self.config = config or {}
        self.model_ = None
        self.is_fitted_ = False
    
    @abstractmethod
    def build_model(self, **kwargs):
        """
        Build the underlying model.
        
        Returns:
            Model instance
        """
        pass
    
    @abstractmethod
    def fit(self, X, y, **kwargs):
        """
        Fit the model.
        
        Args:
            X: Training features
            y: Training labels
            **kwargs: Additional arguments
        
        Returns:
            self
        """
        pass
    
    @abstractmethod
    def predict(self, X):
        """
        Make predictions.
        
        Args:
            X: Features
        
        Returns:
            Predictions
        """
        pass
    
    def predict_proba(self, X):
        """
        Predict class probabilities.
        
        Args:
            X: Features
        
        Returns:
            Class probabilities
        """
        if not hasattr(self.model_, 'predict_proba'):
            raise NotImplementedError(
                "Model does not support probability prediction"
            )
        return self.model_.predict_proba(X)
    
    def get_feature_importance(self):
        """
        Get feature importance if available.
        
        Returns:
            Feature importance array or None
        """
        if hasattr(self.model_, 'feature_importances_'):
            return self.model_.feature_importances_
        return None
    
    def save(self, path: str):
        """
        Save model to disk.
        
        Args:
            path: Save path
        """
        import joblib
        joblib.dump(self.model_, path)
    
    def load(self, path: str):
        """
        Load model from disk.
        
        Args:
            path: Load path
        
        Returns:
            self
        """
        import joblib
        self.model_ = joblib.load(path)
        self.is_fitted_ = True
        return self

