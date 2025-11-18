"""
Configuration for Ensemble2: 100 diverse tree models with Stacking.

This configuration generates 100 different tree models with varying hyperparameters
to maximize diversity and improve ensemble performance through stacking.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any
import numpy as np


@dataclass
class Ensemble2Config:
    """Configuration for Ensemble2 (100 diverse tree models)."""
    
    # Number of base models
    # WARNING: 100 models with 5-fold CV = 500 training tasks, very slow!
    # Start with 20-30 models for testing
    n_models: int = 15  # Reduced from 100 for faster training
    
    # Model types to use (can mix different tree algorithms)
    model_types: List[str] = field(default_factory=lambda: [
        'catboost',  # CatBoost - gradient boosting with categorical handling
        'xgb',       # XGBoost - gradient boosting
        'rf',        # RandomForest - bagging
        'extra_trees'  # ExtraTrees - more randomized
    ])
    
    # Distribution of model types (should sum to n_models)
    # CatBoost is set to have more models than XGBoost
    model_type_distribution: Dict[str, int] = field(default_factory=lambda: {
        'catboost': 6,    # 40 CatBoost models (more than XGBoost)
        'xgb': 5,         # 40 XGBoost models
        'rf': 3,           # 6 RandomForest models
        'extra_trees': 1   # 4 ExtraTrees models
    })
    
    # Hyperparameter search spaces for each model type
    catboost_search_space: Dict[str, Any] = field(default_factory=lambda: {
        'iterations': (100, 2000),      # Number of boosting iterations
        'learning_rate': (0.005, 0.1),  # Learning rate range
        'depth': (3, 10),               # Tree depth range
        'l2_leaf_reg': (1, 20),         # L2 regularization
        'bootstrap_type': ['Bayesian', 'Bernoulli', 'MVS'],  # Bootstrap type
        'subsample': (0.5, 1.0),        # Row sampling
        'colsample_bylevel': (0.3, 1.0), # Column sampling per level
        'random_strength': (0, 1),      # Random strength
        'bagging_temperature': (0, 1),  # Bagging temperature
    })
    
    xgb_search_space: Dict[str, Any] = field(default_factory=lambda: {
        'n_estimators': (100, 2000),      # Range for number of trees
        'learning_rate': (0.005, 0.1),   # Learning rate range
        'max_depth': (3, 10),            # Tree depth range
        'min_child_weight': (1, 20),     # Minimum child weight
        'subsample': (0.5, 1.0),        # Row sampling
        'colsample_bytree': (0.3, 1.0), # Column sampling
        'gamma': (0, 5),                 # Minimum split loss
        'reg_alpha': (0, 10),            # L1 regularization
        'reg_lambda': (0, 20),           # L2 regularization
    })
    
    rf_search_space: Dict[str, Any] = field(default_factory=lambda: {
        'n_estimators': (50, 500),       # Number of trees
        'max_depth': (3, 20),            # Tree depth
        'min_samples_split': (2, 20),    # Minimum samples to split
        'min_samples_leaf': (1, 10),     # Minimum samples in leaf
        'max_features': ['sqrt', 'log2', None, 0.3, 0.5, 0.7],  # Feature sampling
        'bootstrap': [True, False],       # Bootstrap sampling
    })
    
    extra_trees_search_space: Dict[str, Any] = field(default_factory=lambda: {
        'n_estimators': (50, 500),       # Number of trees
        'max_depth': (3, 20),            # Tree depth
        'min_samples_split': (2, 20),    # Minimum samples to split
        'min_samples_leaf': (1, 10),     # Minimum samples in leaf
        'max_features': ['sqrt', 'log2', None, 0.3, 0.5, 0.7],  # Feature sampling
        'bootstrap': [True, False],     # Bootstrap sampling
    })
    
    # Stacking configuration
    stacking_config: Dict[str, Any] = field(default_factory=lambda: {
        'cv': 3,                         # Cross-validation folds for stacking (reduced from 5 for speed)
        'final_estimator': 'logistic',   # Meta-learner: 'logistic', 'rf', 'xgb', 'svm'
        'final_estimator_params': {      # Parameters for meta-learner
            'logistic': {
                'max_iter': 1000,
                'C': 1.0,
                'solver': 'lbfgs',
                'random_state': 42,
                'n_jobs': -1
            },
            'rf': {
                'n_estimators': 100,
                'max_depth': 5,
                'random_state': 42,
                'n_jobs': -1
            },
            'xgb': {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 3,
                'random_state': 42,
                'n_jobs': -1
            },
            'svm': {
                'C': 1.0,
                'kernel': 'rbf',
                'probability': True,
                'random_state': 42
            }
        }
    })
    
    # Random seed for reproducibility
    random_state: int = 42
    
    # Parallel processing
    n_jobs: int = -1  # Use all available cores
    
    # Verbosity
    verbose: bool = True
    
    # Early stopping for XGBoost models (if applicable)
    use_early_stopping: bool = False  # Disabled for ensemble diversity
    early_stopping_rounds: int = 50


def sample_hyperparameters(
    search_space: Dict[str, Any],
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Sample hyperparameters from search space.
    
    Args:
        search_space: Dictionary defining parameter ranges
        random_state: Random seed
    
    Returns:
        Dictionary of sampled hyperparameters
    """
    rng = np.random.RandomState(random_state)
    params = {}
    
    for param_name, param_range in search_space.items():
        if isinstance(param_range, tuple) and len(param_range) == 2:
            # Continuous range (tuple of two numbers)
            if isinstance(param_range[0], int):
                # Integer range
                params[param_name] = rng.randint(
                    param_range[0], param_range[1] + 1
                )
            else:
                # Float range
                params[param_name] = rng.uniform(
                    param_range[0], param_range[1]
                )
        elif isinstance(param_range, list):
            # Categorical (list of options)
            params[param_name] = rng.choice(param_range)
        else:
            # Single value
            params[param_name] = param_range
    
    return params

