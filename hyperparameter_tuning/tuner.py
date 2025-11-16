"""
Hyperparameter tuning utilities.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, Callable
from sklearn.model_selection import cross_val_score, StratifiedKFold

from utils.logger import get_logger

logger = get_logger(__name__)


class HyperparameterTuner:
    """
    Base class for hyperparameter tuning.
    
    Provides a consistent interface for different tuning strategies.
    """
    
    def __init__(self, config):
        """
        Initialize tuner.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.best_params_ = None
        self.best_score_ = None
        self.tuning_results_ = []
    
    def tune(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        preprocessor,
        model_builder: Callable
    ) -> Dict[str, Any]:
        """
        Run hyperparameter tuning.
        
        Args:
            X: Features
            y: Target
            preprocessor: Fitted preprocessor
            model_builder: Function that builds model given params
        
        Returns:
            Dictionary with best parameters and score
        """
        raise NotImplementedError("Subclasses must implement tune()")
    
    def save_results(self, output_path: Optional[str] = None):
        """
        Save tuning results.
        
        Args:
            output_path: Path to save results
        """
        if output_path is None:
            output_path = Path(self.config.paths.output_dir) / "tuning_results.json"
        
        results = {
            'best_params': self.best_params_,
            'best_score': float(self.best_score_),
            'all_results': self.tuning_results_
        }
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Saved tuning results to {output_path}")


class OptunaTuner(HyperparameterTuner):
    """
    Hyperparameter tuning using Optuna.
    
    Optuna provides efficient Bayesian optimization for hyperparameter search.
    """
    
    def __init__(self, config):
        """
        Initialize Optuna tuner.
        
        Args:
            config: Configuration object
        """
        super().__init__(config)
        self.study_ = None
    
    def _get_search_space_for_model(self, model_type: str):
        """
        Get search space configuration for specific model type.
        
        Args:
            model_type: Model type ('xgboost', 'catboost', 'lightgbm', 'mlp')
        
        Returns:
            Search space dictionary
        """
        # Default XGBoost search space (backward compatible)
        default_space = self.config.tuning.search_space
        
        # Model-specific search spaces
        search_spaces = {
            'xgboost': default_space,
            'catboost': {
                'iterations': (500, 2000),
                'learning_rate': (0.01, 0.1),
                'depth': (4, 10),
                'l2_leaf_reg': (1.0, 20.0),
            },
            'lightgbm': {
                'n_estimators': (500, 2000),
                'learning_rate': (0.01, 0.1),
                'max_depth': (3, 10),
                'num_leaves': (15, 127),
                'min_child_samples': (10, 50),
                'subsample': (0.5, 1.0),
                'colsample_bytree': (0.5, 1.0),
                'reg_alpha': (0, 10),
                'reg_lambda': (0, 20),
            },
            'mlp': {
                'hidden_layer_sizes': [(64,), (128,), (64, 32), (128, 64), (128, 64, 32)],
                'alpha': (0.0001, 0.1),
                'learning_rate_init': (0.0001, 0.01),
                'max_iter': (100, 500),
            }
        }
        
        return search_spaces.get(model_type, default_space)
    
    def _sample_params(self, trial, model_type: str, search_space: Dict):
        """
        Sample hyperparameters from search space.
        
        Args:
            trial: Optuna trial object
            model_type: Model type
            search_space: Search space dictionary
        
        Returns:
            Sampled parameters dictionary
        """
        params = {}
        
        for param_name, param_range in search_space.items():
            if param_name == 'hidden_layer_sizes' and isinstance(param_range, list):
                # Special handling for hidden_layer_sizes (categorical)
                params[param_name] = trial.suggest_categorical(param_name, param_range)
            elif isinstance(param_range, tuple) and len(param_range) == 2:
                # Float or int range
                if isinstance(param_range[0], int) and isinstance(param_range[1], int):
                    params[param_name] = trial.suggest_int(param_name, *param_range)
                else:
                    params[param_name] = trial.suggest_float(param_name, *param_range)
        
        return params
    
    def _objective(
        self,
        trial,
        X,
        y,
        preprocessor,
        model_builder,
        model_type: str = 'xgboost'
    ):
        """
        Objective function for Optuna.
        
        Args:
            trial: Optuna trial object
            X: Features
            y: Target
            preprocessor: Preprocessor
            model_builder: Model builder function
            model_type: Model type to tune
        
        Returns:
            Score to maximize
        """
        # Get search space for model type
        search_space = self._get_search_space_for_model(model_type)
        
        # Sample hyperparameters
        params = self._sample_params(trial, model_type, search_space)
        
        # Get base parameters for model type
        if model_type == 'xgboost':
            base_params = self.config.models.xgb_params.copy()
        elif model_type == 'catboost':
            base_params = self.config.models.catboost_params.copy()
        elif model_type == 'lightgbm':
            base_params = self.config.models.lightgbm_params.copy()
        elif model_type == 'mlp':
            base_params = self.config.models.mlp_params.copy()
        else:
            base_params = {}
        
        # Merge sampled params with base params (sampled params override base)
        full_params = {**base_params, **params}
        
        # Build model
        model = model_builder(full_params)
        num_classes = len(np.unique(y))
        model.build_model(num_classes=num_classes)
        
        # Transform data
        X_transformed = preprocessor.transform(X)
        
        # Cross-validation
        cv = StratifiedKFold(
            n_splits=self.config.tuning.cv_folds,
            shuffle=True,
            random_state=self.config.training.random_state
        )
        
        scores = cross_val_score(
            model.model_,
            X_transformed,
            y,
            cv=cv,
            scoring=self.config.tuning.scoring,
            n_jobs=-1
        )
        
        return scores.mean()
    
    def tune(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        preprocessor,
        model_builder: Optional[Callable] = None,
        model_type: str = 'xgboost'
    ) -> Dict[str, Any]:
        """
        Run Optuna hyperparameter tuning.
        
        Args:
            X: Features
            y: Target
            preprocessor: Fitted preprocessor
            model_builder: Function to build model (optional)
            model_type: Model type to tune ('xgboost', 'catboost', 'lightgbm', 'mlp')
        
        Returns:
            Best parameters and score
        """
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            logger.error("Optuna not installed. Install with: pip install optuna")
            raise
        
        logger.info("="*70)
        logger.info(f"STARTING HYPERPARAMETER TUNING (OPTUNA) - {model_type.upper()}")
        logger.info("="*70)
        
        # Default model builder based on model type
        if model_builder is None:
            if model_type == 'xgboost':
                from modeling import XGBoostModel
                model_builder = lambda params: XGBoostModel(config=params)
            elif model_type == 'catboost':
                from modeling.catboost_model import CatBoostModel
                model_builder = lambda params: CatBoostModel(config=params)
            elif model_type == 'lightgbm':
                from modeling.lightgbm_model import LightGBMModel
                model_builder = lambda params: LightGBMModel(config=params)
            elif model_type == 'mlp':
                from modeling.mlp_model import MLPModel
                model_builder = lambda params: MLPModel(config=params)
            else:
                from modeling import XGBoostModel
                model_builder = lambda params: XGBoostModel(config=params)
                logger.warning(f"Unknown model type {model_type}, defaulting to XGBoost")
        
        # Create study with TPE sampler for better optimization
        try:
            import optuna
            self.study_ = optuna.create_study(
                direction='maximize',
                study_name=f'{model_type}_tuning',
                sampler=optuna.samplers.TPESampler(seed=self.config.training.random_state)
            )
        except ImportError:
            logger.error("Optuna not installed. Install with: pip install optuna")
            raise
        
        # Run optimization
        logger.info(f"Running {self.config.tuning.n_trials} trials for {model_type}...")
        
        self.study_.optimize(
            lambda trial: self._objective(
                trial, X, y, preprocessor, model_builder, model_type
            ),
            n_trials=self.config.tuning.n_trials,
            timeout=self.config.tuning.timeout,
            show_progress_bar=True
        )
        
        # Extract results
        self.best_params_ = self.study_.best_params
        self.best_score_ = self.study_.best_value
        
        # Store all trials
        self.tuning_results_ = [
            {
                'params': trial.params,
                'score': trial.value,
                'trial_number': trial.number
            }
            for trial in self.study_.trials
        ]
        
        logger.info("="*70)
        logger.info("TUNING COMPLETE")
        logger.info("="*70)
        logger.info(f"\nBest Score: {self.best_score_:.4f}")
        logger.info(f"Best Parameters:")
        for k, v in self.best_params_.items():
            logger.info(f"  {k}: {v}")
        
        return {
            'best_params': self.best_params_,
            'best_score': self.best_score_
        }
    
    def plot_optimization_history(self, save_path: Optional[str] = None):
        """
        Plot optimization history.
        
        Args:
            save_path: Path to save plot
        """
        try:
            from optuna.visualization import plot_optimization_history
            import plotly
            
            if self.study_ is None:
                logger.warning("No study available. Run tune() first.")
                return
            
            fig = plot_optimization_history(self.study_)
            
            if save_path:
                plotly.io.write_html(fig, save_path)
                logger.info(f"Saved optimization history to {save_path}")
            else:
                fig.show()
        
        except ImportError:
            logger.warning("Plotly not installed. Cannot plot optimization history.")

