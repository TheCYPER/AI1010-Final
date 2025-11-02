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
    
    def _objective(
        self,
        trial,
        X,
        y,
        preprocessor,
        model_builder
    ):
        """
        Objective function for Optuna.
        
        Args:
            trial: Optuna trial object
            X: Features
            y: Target
            preprocessor: Preprocessor
            model_builder: Model builder function
        
        Returns:
            Score to maximize
        """
        from modeling import XGBoostModel
        
        # Sample hyperparameters
        search_space = self.config.tuning.search_space
        
        params = {
            'learning_rate': trial.suggest_float(
                'learning_rate',
                *search_space['learning_rate']
            ),
            'max_depth': trial.suggest_int(
                'max_depth',
                *search_space['max_depth']
            ),
            'min_child_weight': trial.suggest_int(
                'min_child_weight',
                *search_space['min_child_weight']
            ),
            'subsample': trial.suggest_float(
                'subsample',
                *search_space['subsample']
            ),
            'colsample_bytree': trial.suggest_float(
                'colsample_bytree',
                *search_space['colsample_bytree']
            ),
            'gamma': trial.suggest_float(
                'gamma',
                *search_space['gamma']
            ),
            'reg_alpha': trial.suggest_float(
                'reg_alpha',
                *search_space['reg_alpha']
            ),
            'reg_lambda': trial.suggest_float(
                'reg_lambda',
                *search_space['reg_lambda']
            )
        }
        
        # Add base parameters
        full_params = {**self.config.models.xgb_params, **params}
        
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
        model_builder: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Run Optuna hyperparameter tuning.
        
        Args:
            X: Features
            y: Target
            preprocessor: Fitted preprocessor
            model_builder: Function to build model (optional)
        
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
        logger.info("STARTING HYPERPARAMETER TUNING (OPTUNA)")
        logger.info("="*70)
        
        # Default model builder
        if model_builder is None:
            from modeling import XGBoostModel
            model_builder = lambda params: XGBoostModel(config=params)
        
        # Create study
        self.study_ = optuna.create_study(
            direction='maximize',
            study_name='xgboost_tuning'
        )
        
        # Run optimization
        logger.info(f"Running {self.config.tuning.n_trials} trials...")
        
        self.study_.optimize(
            lambda trial: self._objective(
                trial, X, y, preprocessor, model_builder
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

