"""
Ensemble2 GPU: Stacking ensemble with GPU-accelerated tree models.

This implementation creates diverse tree models optimized for GPU training
and combines them using StackingClassifier for improved performance.

GPU Support:
- CatBoost: task_type='GPU'
- XGBoost: tree_method='gpu_hist'
- RandomForest/ExtraTrees: CPU only (sklearn doesn't support GPU)
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


class Ensemble2GPUModel(BaseModel):
    """
    Ensemble2 GPU: Stacking ensemble with GPU-accelerated tree models.
    
    Creates multiple tree models optimized for GPU training with different
    hyperparameters and combines them using stacking for improved generalization.
    
    GPU Models:
    - CatBoost: Uses GPU acceleration
    - XGBoost: Uses GPU acceleration (gpu_hist)
    - RandomForest/ExtraTrees: CPU only (sklearn limitation)
    """
    
    def __init__(self, config: Optional[Ensemble2Config] = None):
        """
        Initialize Ensemble2 GPU model.
        
        Args:
            config: Ensemble2Config object. If None, uses default config.
        """
        super().__init__(config or {})
        self.config = config or Ensemble2Config()
        self.base_models_: List[tuple] = []
        self.model_ = None
        self.num_classes_ = None
        self._xgb_gpu_available = None  # Cache XGBoost GPU availability
        self._catboost_gpu_available = None  # Cache CatBoost GPU availability
        self._catboost_uses_gpu = False
    
    def _create_catboost_model(self, params: Dict[str, Any], model_idx: int, num_classes: int) -> CatBoostClassifier:
        """Create a CatBoost model with GPU acceleration when available."""
        # CatBoost parameter compatibility:
        # - Bayesian bootstrap: supports bagging_temperature, but NOT subsample
        # - Other bootstrap types: support subsample, but NOT bagging_temperature
        # - GPU mode: colsample_bylevel (RSM) is NOT supported for MultiClass (only for pairwise modes)
        # - GPU mode: MVS bootstrap is NOT supported for MultiClass on GPU
        params = params.copy()  # Don't modify the original dict
        bootstrap_type = params.get('bootstrap_type', 'Bayesian')
        
        # GPU mode: MVS is not supported for multiclass, replace with Bernoulli
        if bootstrap_type == 'MVS':
            bootstrap_type = 'Bernoulli'
            params['bootstrap_type'] = 'Bernoulli'
        
        if bootstrap_type == 'Bayesian':
            # Remove subsample for Bayesian bootstrap
            params.pop('subsample', None)
            # Keep bagging_temperature (already in params if sampled)
        else:
            # Remove bagging_temperature for non-Bayesian bootstrap (Bernoulli or MVS->Bernoulli)
            params.pop('bagging_temperature', None)
            # Keep subsample (already in params if sampled)
        
        # GPU mode: Remove colsample_bylevel for MultiClass (not supported on GPU)
        params.pop('colsample_bylevel', None)
        
        use_gpu = self._check_catboost_gpu_available()
        if not use_gpu and model_idx == 0:
            import logging
            logging.getLogger('main').warning("CatBoost GPU not available, falling back to CPU.")
        self._catboost_uses_gpu = self._catboost_uses_gpu or use_gpu
        
        task_type = 'GPU' if use_gpu else 'CPU'
        thread_count = 1 if use_gpu else (self.config.n_jobs if self.config.n_jobs and self.config.n_jobs > 0 else 1)
        
        catboost_params = {
            'loss_function': 'MultiClass',
            'eval_metric': 'MultiClass',
            'classes_count': num_classes,
            'random_seed': self.config.random_state + model_idx,
            'task_type': task_type,
            **({'devices': '0'} if use_gpu else {}),
            'verbose': False,  # Disable verbose output for ensemble
            'allow_writing_files': False,  # Don't write intermediate files
            'thread_count': thread_count,
            **params
        }
        return CatBoostClassifier(**catboost_params)
    
    def _check_catboost_gpu_available(self) -> bool:
        """Check if CatBoost can access a GPU."""
        if self._catboost_gpu_available is not None:
            return self._catboost_gpu_available
        
        try:
            from catboost.utils import get_gpu_device_count
            self._catboost_gpu_available = get_gpu_device_count() > 0
        except Exception:
            # If CatBoost isn't built with GPU support or any other error occurs,
            # treat it as unavailable.
            self._catboost_gpu_available = False
        
        return self._catboost_gpu_available
    
    def _check_xgb_gpu_available(self) -> bool:
        """Check if XGBoost GPU is available."""
        if self._xgb_gpu_available is not None:
            return self._xgb_gpu_available
        
        # Check if 'device' parameter is supported (XGBoost 1.6+)
        import inspect
        xgb_init = XGBClassifier.__init__
        sig = inspect.signature(xgb_init)
        has_device_param = 'device' in sig.parameters
        
        if has_device_param:
            # New API: use device='cuda' with tree_method='hist'
            # We'll try it when creating the model, for now assume available if param exists
            # and CUDA is available
            try:
                import subprocess
                result = subprocess.run(['nvidia-smi', '--query-gpu=count', '--format=csv,noheader'], 
                                      capture_output=True, text=True, timeout=2)
                if result.returncode == 0:
                    self._xgb_gpu_available = True
                else:
                    self._xgb_gpu_available = False
            except Exception:
                self._xgb_gpu_available = False
        else:
            # Old XGBoost version - gpu_hist is not supported in this version
            self._xgb_gpu_available = False
        
        return self._xgb_gpu_available
    
    def _create_xgb_model(self, params: Dict[str, Any], model_idx: int) -> XGBClassifier:
        """Create an XGBoost model with GPU acceleration (if available)."""
        xgb_params = {
            'objective': 'multi:softprob',
            'eval_metric': 'mlogloss',
            'random_state': self.config.random_state + model_idx,
            'n_jobs': 1,  # Use 1 job per model
            **params
        }
        
        # Check GPU availability (cached)
        if self._check_xgb_gpu_available():
            # Try new API first (XGBoost 1.6+)
            xgb_params['tree_method'] = 'hist'
            xgb_params['device'] = 'cuda'
        else:
            # Use CPU - XGBoost GPU not available or not installed
            xgb_params['tree_method'] = 'hist'
            import logging
            logger = logging.getLogger('main')
            if model_idx == 0:  # Only log once
                logger.warning("XGBoost GPU not available, using CPU. Install xgboost with GPU support for acceleration.")
        
        return XGBClassifier(**xgb_params)
    
    def _create_rf_model(self, params: Dict[str, Any], model_idx: int) -> RandomForestClassifier:
        """Create a RandomForest model (CPU only, sklearn limitation)."""
        rf_params = {
            'random_state': self.config.random_state + model_idx,
            'n_jobs': 1,  # Use 1 job per model to avoid oversubscription
            **params
        }
        return RandomForestClassifier(**rf_params)
    
    def _create_extra_trees_model(self, params: Dict[str, Any], model_idx: int) -> ExtraTreesClassifier:
        """Create an ExtraTrees model (CPU only, sklearn limitation)."""
        et_params = {
            'random_state': self.config.random_state + model_idx,
            'n_jobs': 1,  # Use 1 job per model to avoid oversubscription
            **params
        }
        return ExtraTreesClassifier(**et_params)
    
    def _create_final_estimator(self, num_classes: int) -> Any:
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
            # Use GPU for final estimator if XGBoost
            xgb_final_params = final_params.copy()
            xgb_final_params['tree_method'] = 'gpu_hist'
            xgb_final_params['objective'] = 'multi:softprob'
            if 'num_class' not in xgb_final_params:
                xgb_final_params['num_class'] = num_classes
            return XGBClassifier(**xgb_final_params)
        elif final_type == 'svm':
            return SVC(**final_params)
        else:
            raise ValueError(f"Unknown final estimator type: {final_type}")
    
    def build_model(self, num_classes: int, **kwargs):
        """
        Build the ensemble model with GPU-accelerated tree models.
        
        Args:
            num_classes: Number of classes
            **kwargs: Additional parameters (unused)
        
        Returns:
            StackingClassifier instance
        """
        import logging
        import time
        # Use main logger to ensure output
        logger = logging.getLogger('main')
        
        start_time = time.time()
        self.num_classes_ = num_classes
        
        # Determine model type distribution
        model_type_dist = self.config.model_type_distribution
        total_models = sum(model_type_dist.values())
        
        if total_models != self.config.n_models:
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
        
        logger.info(f"Building Ensemble2 GPU: {self.config.n_models} models ({model_type_dist})")
        
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
        final_type = self.config.stacking_config['final_estimator']
        final_estimator = self._create_final_estimator(num_classes)
        
        # Create StackingClassifier
        # CatBoost GPU models cannot run in parallel (device conflict)
        # Check if we have CatBoost GPU models
        has_catboost_gpu = self._catboost_uses_gpu
        
        if has_catboost_gpu:
            # CatBoost GPU doesn't support parallel training on same device
            # Must use n_jobs=1 to avoid "device already requested" error
            gpu_n_jobs = 1
            import logging
            logger = logging.getLogger('main')
            logger.warning("CatBoost GPU detected: using n_jobs=1 (serial training) to avoid GPU device conflicts")
        else:
            # No CatBoost GPU, can use limited parallelism
            gpu_n_jobs = min(self.config.n_jobs, 2) if self.config.n_jobs > 0 else 2
        
        cv_folds = self.config.stacking_config['cv']
        
        self.model_ = StackingClassifier(
            estimators=self.base_models_,
            final_estimator=final_estimator,
            cv=cv_folds,
            n_jobs=gpu_n_jobs,  # Serial training for CatBoost GPU, limited parallel for others
            verbose=1 if self.config.verbose else 0
        )
        
        build_time = time.time() - start_time
        logger.info(f"Model building completed in {build_time:.1f}s")
        
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
        import time
        # Use main logger to ensure output
        logger = logging.getLogger('main')
        
        if self.model_ is None:
            num_classes = len(np.unique(y))
            self.build_model(num_classes)
        
        # Log training information
        n_models = len(self.base_models_)
        cv_folds = self.config.stacking_config['cv']
        total_tasks = n_models * cv_folds
        
        # Count models by type
        model_counts = {}
        gpu_models = 0
        xgb_gpu_available = self._check_xgb_gpu_available()
        for name, _ in self.base_models_:
            model_type = name.split('_')[0]
            model_counts[model_type] = model_counts.get(model_type, 0) + 1
            is_gpu = (
                (model_type == 'catboost' and self._catboost_uses_gpu) or
                (model_type == 'xgb' and xgb_gpu_available)
            )
            if is_gpu:
                gpu_models += 1
        
        logger.info(f"Starting Ensemble2 GPU training: {n_models} models ({gpu_models} GPU, {n_models - gpu_models} CPU), {cv_folds}-fold CV, {total_tasks} tasks")
        logger.info(f"Data: {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(y))} classes")
        
        # StackingClassifier doesn't support eval_set, early_stopping_rounds, verbose
        # Filter out unsupported parameters
        fit_kwargs = {k: v for k, v in kwargs.items() 
                     if k not in ['eval_set', 'early_stopping_rounds', 'verbose']}
        
        # Start training with timing
        training_start = time.time()
        
        try:
            from joblib import parallel_backend
            backend_n_jobs = self.model_.n_jobs if hasattr(self.model_, 'n_jobs') else None
            # Use threading backend to avoid OS restrictions on multiprocessing/semaphores
            with parallel_backend('threading', n_jobs=backend_n_jobs):
                self.model_.fit(X, y, **fit_kwargs)
            training_time = time.time() - training_start
            logger.info(f"Training completed in {training_time / 60:.1f} min ({training_time / total_tasks:.1f}s per task)")
        except Exception as e:
            training_time = time.time() - training_start
            logger.error(f"Training failed after {training_time / 60:.1f} min: {str(e)}")
            raise
        
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


def create_ensemble2_gpu(
    config: Optional[Ensemble2Config] = None,
    num_classes: int = None
) -> Ensemble2GPUModel:
    """
    Factory function to create an Ensemble2 GPU model.
    
    Args:
        config: Ensemble2Config object. If None, uses default config.
        num_classes: Number of classes. If None, will be inferred during fit.
    
    Returns:
        Ensemble2GPUModel instance
    
    Example:
        >>> from configs.ensemble2_config import Ensemble2Config
        >>> from modeling.ensemble2_gpu import create_ensemble2_gpu
        >>> 
        >>> config = Ensemble2Config(n_models=100)
        >>> ensemble = create_ensemble2_gpu(config, num_classes=5)
        >>> ensemble.fit(X_train, y_train)
        >>> predictions = ensemble.predict(X_test)
    """
    ensemble = Ensemble2GPUModel(config=config)
    if num_classes is not None:
        ensemble.build_model(num_classes)
    return ensemble
