"""
Ridge Classifier model implementation using sklearn.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, Optional
from sklearn.linear_model import RidgeClassifier
from sklearn.base import BaseEstimator, ClassifierMixin

from .base_model import BaseModel


class RidgeSklearnWrapper(BaseEstimator, ClassifierMixin):
    """
    Wrapper to add predict_proba to RidgeClassifier for ensemble compatibility.
    
    RidgeClassifier doesn't have predict_proba by default, but VotingClassifier
    requires it for soft voting. This wrapper adds predict_proba using
    decision_function and softmax.
    """
    
    # Explicitly mark as classifier for sklearn validation
    _estimator_type = "classifier"
    
    def __init__(self, ridge_classifier):
        """
        Initialize wrapper.
        
        Args:
            ridge_classifier: RidgeClassifier instance
        """
        self.ridge_classifier = ridge_classifier
        # Initialize classes_ for sklearn validation
        # If RidgeClassifier already has classes_, use it; otherwise use empty array
        import numpy as np
        if hasattr(ridge_classifier, 'classes_') and ridge_classifier.classes_ is not None:
            self.classes_ = ridge_classifier.classes_
        else:
            self.classes_ = np.array([])
    
    def fit(self, X, y, sample_weight=None):
        """
        Fit RidgeClassifier.
        
        Args:
            X: Training features
            y: Training labels
            sample_weight: Sample weights
        
        Returns:
            self
        """
        self.ridge_classifier.fit(X, y, sample_weight=sample_weight)
        # Set classes_ attribute required by sklearn classifiers
        if hasattr(self.ridge_classifier, 'classes_'):
            self.classes_ = self.ridge_classifier.classes_
        else:
            self.classes_ = np.unique(y)
        return self
    
    def predict(self, X):
        """
        Make predictions.
        
        Args:
            X: Features
        
        Returns:
            Predicted class labels
        """
        return self.ridge_classifier.predict(X)
    
    def predict_proba(self, X):
        """
        Predict class probabilities using decision_function and softmax.
        
        Args:
            X: Features
        
        Returns:
            Class probabilities
        """
        # Get decision function scores
        decision_scores = self.ridge_classifier.decision_function(X)
        
        # Apply softmax to convert to probabilities
        exp_scores = np.exp(decision_scores - np.max(decision_scores, axis=1, keepdims=True))
        proba = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        
        return proba
    
    def _more_tags(self):
        """Return estimator tags for sklearn compatibility."""
        return {
            'requires_y': True,
            'binary_only': False,
            'multilabel': False,
            'multioutput': False,
            'no_validation': False,
            'poor_score': False,
        }


class RidgeModel(BaseModel):
    """
    Ridge Classifier wrapper.
    
    Ridge Classifier is a linear model with L2 regularization that can provide
    diversity in ensemble models.
    
    Advantages:
    - Fully sklearn-compatible
    - Fast training and prediction
    - L2 regularization prevents overfitting
    - Good baseline model
    - Captures linear relationships
    
    Notes:
    - Requires scaled/normalized input (already handled by preprocessor)
    - Linear model (assumes linear relationships)
    - Good regularization support
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Ridge Classifier model.
        
        Args:
            config: Model configuration
        """
        super().__init__(config)
        self.num_classes_ = None
    
    def build_model(self, num_classes: int, **kwargs) -> RidgeClassifier:
        """
        Build Ridge Classifier.
        
        Args:
            num_classes: Number of classes
            **kwargs: Override config parameters
        
        Returns:
            RidgeClassifier instance
        """
        self.num_classes_ = num_classes
        
        # Merge config with kwargs
        params = {**self.config, **kwargs}
        
        # Build RidgeClassifier
        ridge_classifier = RidgeClassifier(**params)
        
        # Add predict_proba method directly to the instance for ensemble compatibility
        # This avoids wrapper issues with sklearn validation
        # We need to create a closure that captures the ridge_classifier instance
        def predict_proba_method(self, X):
            """Predict class probabilities using decision_function and softmax."""
            # Get decision function scores
            decision_scores = self.decision_function(X)
            
            # Apply softmax to convert to probabilities
            exp_scores = np.exp(decision_scores - np.max(decision_scores, axis=1, keepdims=True))
            proba = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
            
            return proba
        
        # Bind the method to the instance using types.MethodType
        import types
        ridge_classifier.predict_proba = types.MethodType(predict_proba_method, ridge_classifier)
        
        self.model_ = ridge_classifier
        
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
        Fit Ridge Classifier model.
        
        Args:
            X: Training features
            y: Training labels
            eval_set: Not used
            sample_weight: Sample weights (RidgeClassifier supports this)
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
        
        # RidgeClassifier supports sample_weight
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
        
        Note: RidgeClassifier now has predict_proba method added directly to the instance.
        
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
        
        # Use the added predict_proba method
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

