"""
Extra Trees (Extremely Randomized Trees) model implementation using sklearn.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, Optional
from sklearn.ensemble import ExtraTreesClassifier

from .base_model import BaseModel


class ExtraTreesModel(BaseModel):
    """
    Extra Trees Classifier wrapper.
    
    Extra Trees is similar to Random Forest but uses more randomization,
    which can provide diversity in ensemble models.
    
    Advantages:
    - Fully sklearn-compatible
    - Fast training (more randomization = faster)
    - Good generalization
    - Provides feature importance
    - Complements other tree models well
    
    Notes:
    - More randomization than Random Forest
    - Can handle missing values (though we preprocess them)
    - Good for high-dimensional data
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Extra Trees model.
        
        Args:
            config: Model configuration
        """
        super().__init__(config)
        self.num_classes_ = None
    
    def build_model(self, num_classes: int, **kwargs) -> ExtraTreesClassifier:
        """
        Build Extra Trees classifier.
        
        Args:
            num_classes: Number of classes
            **kwargs: Override config parameters
        
        Returns:
            ExtraTreesClassifier instance
        """
        self.num_classes_ = num_classes
        
        # Merge config with kwargs
        params = {**self.config, **kwargs}
        
        # Build model
        self.model_ = ExtraTreesClassifier(**params)
        
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
        Fit Extra Trees model.
        
        Args:
            X: Training features
            y: Training labels
            eval_set: Not used (Extra Trees doesn't support early stopping)
            sample_weight: Sample weights (Extra Trees supports this)
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
        
        # Extra Trees supports sample_weight
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
        Get feature importance.
        
        Returns:
            Feature importance array
        """
        if not self.is_fitted_:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if hasattr(self.model_, 'feature_importances_'):
            return self.model_.feature_importances_
        
        return None

