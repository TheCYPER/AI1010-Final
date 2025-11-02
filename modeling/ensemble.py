"""
Ensemble model implementations.
"""

import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.ensemble import VotingClassifier, StackingClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

from .base_model import BaseModel


class EnsembleModel(BaseModel):
    """
    Ensemble model combining multiple base models.
    
    Supports both voting and stacking strategies.
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        ensemble_type: str = 'voting',
        voting: str = 'soft'
    ):
        """
        Initialize ensemble model.
        
        Args:
            config: Model configuration
            ensemble_type: 'voting' or 'stacking'
            voting: 'soft' (average probabilities) or 'hard' (majority vote)
        """
        super().__init__(config)
        self.ensemble_type = ensemble_type
        self.voting = voting
        self.base_models_: List[tuple] = []
    
    def add_model(self, name: str, model):
        """
        Add a base model to the ensemble.
        
        Args:
            name: Model name
            model: Model instance
        """
        self.base_models_.append((name, model))
    
    def build_model(self, num_classes: int = None, **kwargs):
        """
        Build ensemble model.
        
        Args:
            num_classes: Number of classes (unused, inferred from base models)
            **kwargs: Additional parameters
        
        Returns:
            Ensemble model instance
        """
        if not self.base_models_:
            raise ValueError("No base models added. Call add_model() first.")
        
        if self.ensemble_type == 'voting':
            self.model_ = VotingClassifier(
                estimators=self.base_models_,
                voting=self.voting,
                n_jobs=-1
            )
        elif self.ensemble_type == 'stacking':
            # Use logistic regression as meta-classifier
            from sklearn.linear_model import LogisticRegression
            self.model_ = StackingClassifier(
                estimators=self.base_models_,
                final_estimator=LogisticRegression(max_iter=1000),
                cv=5,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown ensemble type: {self.ensemble_type}")
        
        return self.model_
    
    def fit(self, X, y, **kwargs):
        """
        Fit ensemble model.
        
        Args:
            X: Training features
            y: Training labels
            **kwargs: Additional fit parameters
        
        Returns:
            self
        """
        if self.model_ is None:
            self.build_model()
        
        self.model_.fit(X, y, **kwargs)
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


def create_xgb_rf_ensemble(xgb_params: Dict, rf_params: Dict) -> EnsembleModel:
    """
    Create an ensemble of XGBoost and Random Forest.
    
    Args:
        xgb_params: XGBoost parameters
        rf_params: Random Forest parameters
    
    Returns:
        EnsembleModel instance
    """
    ensemble = EnsembleModel(ensemble_type='voting', voting='soft')
    
    # Add XGBoost
    xgb = XGBClassifier(**xgb_params)
    ensemble.add_model('xgb', xgb)
    
    # Add Random Forest
    rf = RandomForestClassifier(**rf_params)
    ensemble.add_model('rf', rf)
    
    return ensemble

