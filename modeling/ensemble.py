"""
Ensemble model implementations.
"""

import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

from .base_model import BaseModel


class TabNetSklearnWrapper(BaseEstimator, ClassifierMixin):
    """
    Wrapper to make TabNet compatible with sklearn's VotingClassifier/StackingClassifier.
    
    TabNet uses a different API than sklearn estimators, so we need this wrapper
    to make it work with sklearn ensemble methods.
    """
    
    # Explicitly mark as classifier for sklearn validation
    _estimator_type = "classifier"
    
    def __init__(self, tabnet_model):
        """
        Initialize wrapper.
        
        Args:
            tabnet_model: TabNetModel instance
        """
        self.tabnet_model = tabnet_model
        # Initialize classes_ as None, will be set during fit
        self.classes_ = None
    
    def fit(self, X, y):
        """
        Fit TabNet model.
        
        Args:
            X: Training features
            y: Training labels
        
        Returns:
            self
        """
        self.tabnet_model.fit(X, y)
        # Set classes_ attribute required by sklearn classifiers
        if hasattr(self.tabnet_model.model_, 'classes_'):
            self.classes_ = self.tabnet_model.model_.classes_
        else:
            # Infer from y
            import numpy as np
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
        return self.tabnet_model.predict(X)
    
    def predict_proba(self, X):
        """
        Predict class probabilities.
        
        Args:
            X: Features
        
        Returns:
            Class probabilities
        """
        return self.tabnet_model.predict_proba(X)
    
    def _more_tags(self):
        """Return estimator tags for sklearn compatibility."""
        return {
            'requires_y': True,
            'binary_only': False,
            'multilabel': False,
            'multioutput': False,
            'no_validation': False,
            'poor_score': False,
            'non_deterministic': True,  # TabNet may have some randomness
        }


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
            **kwargs: Additional fit parameters (will be filtered for sklearn compatibility)
        
        Returns:
            self
        """
        if self.model_ is None:
            self.build_model()
        
        # VotingClassifier/StackingClassifier do not support eval_set, early_stopping_rounds, etc.
        # Only keep parameters supported by sklearn ensemble
        sklearn_supported_params = {}
        
        # Check if all base models support sample_weight
        # KNN doesn't support sample_weight, so we need to check
        models_that_dont_support_sample_weight = ['knn']
        has_unsupported_model = any(
            name in models_that_dont_support_sample_weight 
            for name, _ in self.base_models_
        )
        
        # sample_weight is supported by VotingClassifier and StackingClassifier
        # But only if all base models support it
        if 'sample_weight' in kwargs and not has_unsupported_model:
            sklearn_supported_params['sample_weight'] = kwargs['sample_weight']
        elif 'sample_weight' in kwargs and has_unsupported_model:
            import warnings
            warnings.warn(
                "sample_weight is provided but some models (e.g., KNN) don't support it. "
                "sample_weight will be ignored for ensemble training.",
                UserWarning
            )
        
        # Filter out unsupported parameters: eval_set, early_stopping_rounds, verbose, etc.
        # These parameters are specific to certain base models and cannot be passed to ensemble
        # Note: In ensemble mode, base models cannot use early stopping
        # If early stopping is needed, set reasonable n_estimators/iterations when creating base models
        
        unsupported_params = set(kwargs.keys()) - set(sklearn_supported_params.keys())
        if unsupported_params:
            import warnings
            warnings.warn(
                f"Ensemble model does not support these parameters: {unsupported_params}. "
                "They will be ignored. Note: Early stopping is not available in ensemble mode. "
                "Consider adjusting n_estimators/iterations in model configs.",
                UserWarning
            )
        
        self.model_.fit(X, y, **sklearn_supported_params)
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


def create_full_ensemble(
    model_config: Any,
    num_classes: int,
    ensemble_type: str = 'voting',
    voting: str = 'soft',
    include_models: Optional[List[str]] = None
) -> EnsembleModel:
    """
    Create an ensemble model from config containing all available models.
    
    This function reads model parameters from ModelConfig, creates model instances,
    and combines them into an ensemble.
    
    Args:
        model_config: ModelConfig object containing parameters for all models
        num_classes: Number of classes
        ensemble_type: 'voting' or 'stacking'
        voting: 'soft' (average probabilities) or 'hard' (majority vote)
        include_models: List of models to include, e.g., ['xgb', 'catboost', 'lightgbm', 'rf', 'tabnet']
                        If None, includes all tree-based models (TabNet excluded by default)
    
    Returns:
        EnsembleModel instance
    
    Example:
        >>> from configs.config import Config
        >>> config = Config()
        >>> ensemble = create_full_ensemble(
        ...     model_config=config.models,
        ...     num_classes=5,
        ...     ensemble_type='voting',
        ...     voting='soft'
        ... )
    """
    from .xgboost_model import XGBoostModel
    from .catboost_model import CatBoostModel
    from .lightgbm_model import LightGBMModel
    from .mlp_model import MLPModel
    from .knn_model import KNNModel
    from .logistic_model import LogisticRegressionModel
    from .svm_model import SVMModel
    from .naive_bayes_model import NaiveBayesModel
    from .ridge_model import RidgeModel
    from .extra_trees_model import ExtraTreesModel
    from .tabnet_model import TabNetModel
    
    # Default: include tree-based models, MLP, and diverse non-tree models
    # TabNet excluded by default due to long training time
    # Can customize: include_models=['xgb', 'catboost', 'lightgbm', 'rf', 'mlp', 'svm', 'naive_bayes', 'ridge', 'extra_trees', 'tabnet']
    if include_models is None:
        # Include strong models: tree-based + MLP + diverse non-tree models
        include_models = ['xgb', 'catboost', 'lightgbm', 'rf', 'mlp']
    
    ensemble = EnsembleModel(ensemble_type=ensemble_type, voting=voting)
    
    # Add XGBoost
    if 'xgb' in include_models:
        xgb_model = XGBoostModel(config=model_config.xgb_params)
        xgb_model.build_model(num_classes=num_classes)
        ensemble.add_model('xgb', xgb_model.model_)
    
    # Add CatBoost
    if 'catboost' in include_models:
        catboost_model = CatBoostModel(config=model_config.catboost_params)
        catboost_model.build_model(num_classes=num_classes)
        ensemble.add_model('catboost', catboost_model.model_)
    
    # Add LightGBM
    if 'lightgbm' in include_models:
        lightgbm_model = LightGBMModel(config=model_config.lightgbm_params)
        lightgbm_model.build_model(num_classes=num_classes)
        ensemble.add_model('lightgbm', lightgbm_model.model_)
    
    # Add RandomForest
    if 'rf' in include_models:
        rf = RandomForestClassifier(**model_config.rf_params)
        # RandomForest will automatically infer n_classes, but we set it explicitly for clarity
        ensemble.add_model('rf', rf)
    
    # Add MLP (fully sklearn-compatible neural network)
    if 'mlp' in include_models:
        mlp_model = MLPModel(config=model_config.mlp_params)
        mlp_model.build_model(num_classes=num_classes)
        ensemble.add_model('mlp', mlp_model.model_)
    
    # Add KNN (instance-based learning, provides diversity)
    if 'knn' in include_models:
        knn_model = KNNModel(config=model_config.knn_params)
        knn_model.build_model(num_classes=num_classes)
        ensemble.add_model('knn', knn_model.model_)
    
    # Add Logistic Regression (linear model, provides diversity)
    if 'logistic' in include_models:
        logistic_model = LogisticRegressionModel(config=model_config.logistic_params)
        logistic_model.build_model(num_classes=num_classes)
        ensemble.add_model('logistic', logistic_model.model_)
    
    # Add SVM (non-linear classifier, provides diversity)
    if 'svm' in include_models:
        svm_model = SVMModel(config=model_config.svm_params)
        svm_model.build_model(num_classes=num_classes)
        ensemble.add_model('svm', svm_model.model_)
    
    # Add Naive Bayes (probabilistic classifier, provides diversity)
    if 'naive_bayes' in include_models:
        nb_model = NaiveBayesModel(config=model_config.naive_bayes_params)
        nb_model.build_model(num_classes=num_classes)
        ensemble.add_model('naive_bayes', nb_model.model_)
    
    # Add Ridge Classifier (linear model with L2 regularization)
    if 'ridge' in include_models:
        ridge_model = RidgeModel(config=model_config.ridge_params)
        ridge_model.build_model(num_classes=num_classes)
        ensemble.add_model('ridge', ridge_model.model_)
    
    # Add Extra Trees (more randomized than RF, provides diversity)
    if 'extra_trees' in include_models:
        et_model = ExtraTreesModel(config=model_config.extra_trees_params)
        et_model.build_model(num_classes=num_classes)
        ensemble.add_model('extra_trees', et_model.model_)
    
    # Add TabNet (requires special handling as it's not fully sklearn-compatible)
    if 'tabnet' in include_models:
        tabnet_model = TabNetModel(config=model_config.tabnet_params)
        tabnet_model.build_model(num_classes=num_classes)
        # Use wrapper to make it sklearn-compatible
        tabnet_wrapper = TabNetSklearnWrapper(tabnet_model)
        ensemble.add_model('tabnet', tabnet_wrapper)
    
    if not ensemble.base_models_:
        raise ValueError(
            "No models were added to ensemble. "
            "Please check include_models parameter."
        )
    
    return ensemble

