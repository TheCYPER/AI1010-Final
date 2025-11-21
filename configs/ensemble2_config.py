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
    n_models: int = 20  # Expanded ensemble with mixed tree models
    
    # Model types to use (can mix different algorithms)
    model_types: List[str] = field(default_factory=lambda: [
        'catboost',    # CatBoost - gradient boosting with categorical handling
        'xgb',         # XGBoost - gradient boosting
        'rf',          # RandomForest - bagging
        'extra_trees', # ExtraTrees - randomized trees
        'hist_gbm',    # sklearn HistGradientBoosting
    ])
    
    # Distribution of model types (should sum to n_models)
    model_type_distribution: Dict[str, int] = field(default_factory=lambda: {
        'catboost': 6,
        'xgb': 6,
        'rf': 4,
        'extra_trees': 2,
        'hist_gbm': 2
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
    
    mlp_search_space: Dict[str, Any] = field(default_factory=lambda: {
        'hidden_layer_sizes': [(64, 32), (128, 64), (64, 64)],
        'alpha': (1e-4, 1e-2),
        'learning_rate_init': (0.001, 0.01),
        'max_iter': (120, 200),
        'batch_size': [64, 128, 256],
    })
    
    hgb_search_space: Dict[str, Any] = field(default_factory=lambda: {
        'learning_rate': (0.03, 0.15),
        'max_depth': (4, 10),
        'max_leaf_nodes': (16, 64),
        'min_samples_leaf': (5, 30),
        'l2_regularization': (0.0, 0.2)
    })
    
    svm_search_space: Dict[str, Any] = field(default_factory=lambda: {
        'C': (0.5, 5.0),
        'gamma': ['scale', 'auto']
    })
    
    knn_search_space: Dict[str, Any] = field(default_factory=lambda: {
        'n_neighbors': (3, 25),
        'weights': ['uniform', 'distance'],
        'p': [1, 2]
    })
    
    logistic_search_space: Dict[str, Any] = field(default_factory=lambda: {
        'C': (0.3, 2.5),
        'penalty': ['l2'],
        'solver': ['lbfgs'],
        'max_iter': (400, 800),  # higher cap to avoid convergence warnings
        'tol': (1e-4, 1e-3)
    })
    
    # Stacking configuration
    stacking_config: Dict[str, Any] = field(default_factory=lambda: {
        'cv': 2,                         # Cross-validation folds for stacking (reduced for speed)
        'final_estimator': 'logistic',   # Meta-learner: 'logistic', 'rf', 'xgb', 'svm'
        'final_estimator_params': {      # Parameters for meta-learner
            'logistic': {
                'max_iter': 1000,
                'C': 0.5,
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
    n_jobs: int = 2  # Threaded parallelism without oversubscription
    
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
            # Categorical (list of options) - sample by index to handle tuples safely
            idx = rng.randint(0, len(param_range))
            params[param_name] = param_range[idx]
        else:
            # Single value
            params[param_name] = param_range
    
    return params
