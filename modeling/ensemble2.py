"""
Ensemble2: Stacking ensemble with 100 diverse tree models.

This implementation creates 100 different tree models with varying hyperparameters
and combines them using StackingClassifier for improved performance.
"""

import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.ensemble import StackingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from .base_model import BaseModel
from configs.ensemble2_config import Ensemble2Config, sample_hyperparameters


class Ensemble2Model(BaseModel):
    """
    Ensemble2: Stacking ensemble with 100 diverse tree models.
    
    Creates multiple tree models with different hyperparameters and combines
    them using stacking for improved generalization.
    """
    
    def __init__(self, config: Optional[Ensemble2Config] = None):
        """
        Initialize Ensemble2 model.
        
        Args:
            config: Ensemble2Config object. If None, uses default config.
        """
        super().__init__(config or {})
        self.config = config or Ensemble2Config()
        self.base_models_: List[tuple] = []
        self.model_ = None
        self.num_classes_ = None
    
    def _create_catboost_model(self, params: Dict[str, Any], model_idx: int, num_classes: int) -> CatBoostClassifier:
        """Create a CatBoost model with given parameters."""
        # CatBoost parameter compatibility:
        # - Bayesian bootstrap: supports bagging_temperature, but NOT subsample
        # - Other bootstrap types: support subsample, but NOT bagging_temperature
        params = params.copy()  # Don't modify the original dict
        bootstrap_type = params.get('bootstrap_type', 'Bayesian')
        
        if bootstrap_type == 'Bayesian':
            # Remove subsample for Bayesian bootstrap
            params.pop('subsample', None)
            # Keep bagging_temperature (already in params if sampled)
        else:
            # Remove bagging_temperature for non-Bayesian bootstrap
            params.pop('bagging_temperature', None)
            # Keep subsample (already in params if sampled)
        
        catboost_params = {
            'loss_function': 'MultiClass',
            'eval_metric': 'MultiClass',
            'classes_count': num_classes,
            'random_seed': self.config.random_state + model_idx,
            'verbose': False,  # Disable verbose output for ensemble
            'allow_writing_files': False,  # Don't write intermediate files
            'thread_count': 1,  # Use 1 thread per model to avoid oversubscription
            **params
        }
        return CatBoostClassifier(**catboost_params)
    
    def _create_xgb_model(self, params: Dict[str, Any], model_idx: int) -> XGBClassifier:
        """Create an XGBoost model with given parameters."""
        xgb_params = {
            'objective': 'multi:softprob',
            'eval_metric': 'mlogloss',
            'tree_method': 'hist',
            'random_state': self.config.random_state + model_idx,
            'n_jobs': 1,  # Use 1 job per model to avoid oversubscription
            **params
        }
        return XGBClassifier(**xgb_params)
    
    def _create_rf_model(self, params: Dict[str, Any], model_idx: int) -> RandomForestClassifier:
        """Create a RandomForest model with given parameters."""
        rf_params = {
            'random_state': self.config.random_state + model_idx,
            'n_jobs': 1,  # Use 1 job per model to avoid oversubscription
            **params
        }
        return RandomForestClassifier(**rf_params)
    
    def _create_extra_trees_model(self, params: Dict[str, Any], model_idx: int) -> ExtraTreesClassifier:
        """Create an ExtraTrees model with given parameters."""
        et_params = {
            'random_state': self.config.random_state + model_idx,
            'n_jobs': 1,  # Use 1 job per model to avoid oversubscription
            **params
        }
        return ExtraTreesClassifier(**et_params)
    
    def _create_final_estimator(self) -> Any:
        """Create the meta-learner (final estimator) for stacking."""
        final_type = self.config.stacking_config['final_estimator']
        final_params = self.config.stacking_config['final_estimator_params'].get(
            final_type, {}
        )
        
        if final_type == 'logistic':
            return LogisticRegression(**final_params)
        elif final_type == 'rf':
            return RandomForestClassifier(**final_params)
        elif final_type == 'xgb':
            return XGBClassifier(**final_params)
        elif final_type == 'svm':
            return SVC(**final_params)
        else:
            raise ValueError(f"Unknown final estimator type: {final_type}")
    
    def build_model(self, num_classes: int, **kwargs):
        """
        Build the ensemble model with 100 diverse tree models.
        
        Args:
            num_classes: Number of classes
            **kwargs: Additional parameters (unused)
        
        Returns:
            StackingClassifier instance
        """
        self.num_classes_ = num_classes
        
        # Determine model type distribution
        model_type_dist = self.config.model_type_distribution
        total_models = sum(model_type_dist.values())
        
        if total_models != self.config.n_models:
            # Adjust distribution to match n_models
            scale = self.config.n_models / total_models
            model_type_dist = {
                k: int(v * scale) for k, v in model_type_dist.items()
            }
            # Adjust for rounding errors
            current_total = sum(model_type_dist.values())
            if current_total < self.config.n_models:
                # Add remaining to the first model type
                first_type = list(model_type_dist.keys())[0]
                model_type_dist[first_type] += (self.config.n_models - current_total)
        
        # Create base models
        model_idx = 0
        for model_type, count in model_type_dist.items():
            for i in range(count):
                # Sample hyperparameters
                if model_type == 'catboost':
                    params = sample_hyperparameters(
                        self.config.catboost_search_space,
                        random_state=self.config.random_state + model_idx
                    )
                    model = self._create_catboost_model(params, model_idx, num_classes)
                elif model_type == 'xgb':
                    params = sample_hyperparameters(
                        self.config.xgb_search_space,
                        random_state=self.config.random_state + model_idx
                    )
                    model = self._create_xgb_model(params, model_idx)
                elif model_type == 'rf':
                    params = sample_hyperparameters(
                        self.config.rf_search_space,
                        random_state=self.config.random_state + model_idx
                    )
                    model = self._create_rf_model(params, model_idx)
                elif model_type == 'extra_trees':
                    params = sample_hyperparameters(
                        self.config.extra_trees_search_space,
                        random_state=self.config.random_state + model_idx
                    )
                    model = self._create_extra_trees_model(params, model_idx)
                else:
                    raise ValueError(f"Unknown model type: {model_type}")
                
                # Add model to ensemble
                model_name = f"{model_type}_{model_idx}"
                self.base_models_.append((model_name, model))
                model_idx += 1
        
        # Create final estimator (meta-learner)
        final_estimator = self._create_final_estimator()
        
        # Create StackingClassifier
        self.model_ = StackingClassifier(
            estimators=self.base_models_,
            final_estimator=final_estimator,
            cv=self.config.stacking_config['cv'],
            n_jobs=self.config.n_jobs,
            verbose=1 if self.config.verbose else 0
        )
        
        return self.model_
    
    def fit(self, X, y, **kwargs):
        """
        Fit the ensemble model.
        
        Args:
            X: Training features
            y: Training labels
            **kwargs: Additional fit parameters (ignored for stacking)
        
        Returns:
            self
        """
        import logging
        logger = logging.getLogger(__name__)
        
        if self.model_ is None:
            num_classes = len(np.unique(y))
            self.build_model(num_classes)
        
        # Log training information
        n_models = len(self.base_models_)
        cv_folds = self.config.stacking_config['cv']
        total_tasks = n_models * cv_folds
        
        # Count models by type
        model_counts = {}
        for name, _ in self.base_models_:
            model_type = name.split('_')[0]
            model_counts[model_type] = model_counts.get(model_type, 0) + 1
        
        logger.info(f"Starting Ensemble2 training:")
        logger.info(f"  - Number of base models: {n_models}")
        logger.info(f"  - Model distribution: {model_counts}")
        logger.info(f"  - Cross-validation folds: {cv_folds}")
        logger.info(f"  - Total training tasks: {total_tasks} (this will take a while...)")
        logger.info(f"  - Estimated time: ~{total_tasks * 0.5 / 60:.1f} minutes (rough estimate)")
        
        # StackingClassifier doesn't support eval_set, early_stopping_rounds, verbose
        # Filter out unsupported parameters
        fit_kwargs = {k: v for k, v in kwargs.items() 
                     if k not in ['eval_set', 'early_stopping_rounds', 'verbose']}
        
        logger.info(f"Training started. Each 'Done {cv_folds} out of {cv_folds}' message means one model completed CV.")
        self.model_.fit(X, y, **fit_kwargs)
        logger.info("Ensemble2 training completed!")
        self.is_fitted_ = True
        return self
    
    def predict(self, X):
        """Make predictions."""
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before prediction")
        return self.model_.predict(X)
    
    def predict_proba(self, X):
        """Predict class probabilities."""
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before prediction")
        return self.model_.predict_proba(X)
    
    def get_feature_importance(self):
        """
        Get feature importance from the ensemble.
        
        For stacking, we return the feature importance from the final estimator
        (meta-learner), which represents how important each base model is.
        
        Returns:
            Dictionary with model names and their importance scores
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before getting feature importance")
        
        # Get feature importance from final estimator
        # For stacking, "features" are the base model predictions
        if hasattr(self.model_.final_estimator_, 'feature_importances_'):
            importances = self.model_.final_estimator_.feature_importances_
        elif hasattr(self.model_.final_estimator_, 'coef_'):
            # For linear models, use absolute coefficients
            coef = self.model_.final_estimator_.coef_
            importances = np.abs(coef).mean(axis=0)  # Average across classes
        else:
            # Fallback: equal importance
            importances = np.ones(len(self.base_models_)) / len(self.base_models_)
        
        # Create dictionary mapping model names to importance
        importance_dict = {
            name: importance 
            for (name, _), importance in zip(self.base_models_, importances)
        }
        
        return importance_dict


def create_ensemble2(
    config: Optional[Ensemble2Config] = None,
    num_classes: int = None
) -> Ensemble2Model:
    """
    Factory function to create an Ensemble2 model.
    
    Args:
        config: Ensemble2Config object. If None, uses default config.
        num_classes: Number of classes. If None, will be inferred during fit.
    
    Returns:
        Ensemble2Model instance
    
    Example:
        >>> from configs.ensemble2_config import Ensemble2Config
        >>> from modeling.ensemble2 import create_ensemble2
        >>> 
        >>> config = Ensemble2Config(n_models=100)
        >>> ensemble = create_ensemble2(config, num_classes=5)
        >>> ensemble.fit(X_train, y_train)
        >>> predictions = ensemble.predict(X_test)
    """
    ensemble = Ensemble2Model(config=config)
    if num_classes is not None:
        ensemble.build_model(num_classes)
    return ensemble

