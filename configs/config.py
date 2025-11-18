"""
Centralized configuration management for the ML pipeline.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pathlib import Path


@dataclass
class PathConfig:
    """File paths configuration."""
    # Data paths
    train_csv: str = "datasets/office_train.csv"
    test_csv: str = "datasets/office_test.csv"
    
    # Output paths
    output_dir: str = "outputs"
    models_dir: str = "outputs/models"
    metrics_dir: str = "outputs/metrics"
    predictions_dir: str = "outputs/predictions"
    logs_dir: str = "outputs/logs"
    
    def __post_init__(self):
        """Create output directories if they don't exist."""
        for attr in ['output_dir', 'models_dir', 'metrics_dir', 'predictions_dir', 'logs_dir']:
            path = Path(getattr(self, attr))
            path.mkdir(parents=True, exist_ok=True)


@dataclass
class ColumnConfig:
    """Column configuration for data."""
    target: str = "OfficeCategory"
    numeric: Optional[List[str]] = None
    categorical: Optional[List[str]] = None
    
    # Columns to drop (low quality/redundant/severe imbalance)
    drop_columns: List[str] = field(default_factory=lambda: [
        "AlleyAccess",
        "ExteriorFinishType", 
        "RecreationQuality",
        "MiscellaneousFeature",
        # Severely imbalanced categorical features
        "ZoningClassification",
        "StreetType",
        "LandFlatness",
        "UtilitiesAvailable",
        "PlotConfiguration",
        "PlotSlope",
        "Proximity1",
        "Proximity2",
        "BuildingType",
        "FunctionalityRating",
        "ParkingType",
        "ParkingQuality",
        "ParkingCondition",
        "PavedAccess",
        "ListingType",
        "ListingCondition"
    ])
    
    # Business missing column
    business_missing_col: str = "ConferenceRoomQuality"


@dataclass
class ModelConfig:
    """Model-specific configurations."""
    # Model type selection
    model_type: str = "ensemble"  
    # Options: "xgboost", "catboost", "lightgbm", "tabnet", "mlp", "knn", "logistic", "svm", "naive_bayes", "ridge", "extra_trees", "ensemble"

    # CatBoost: 0.8550
    # xgboost: 0.8400
    # lightgbm: 0.8403
    # mlp: 0.8383

    
    # XGBoost parameters
    xgb_params: Dict[str, Any] = field(default_factory=lambda: {
        'objective': 'multi:softprob',
        'n_estimators': 1500,
        'learning_rate': 0.01,
        'max_depth': 7,
        'gamma': 1.0,
        'min_child_weight': 10,
        'subsample': 0.75,
        'colsample_bytree': 0.55,
        'reg_lambda': 10.0,
        'reg_alpha': 3.0,
        'random_state': 42,
        'eval_metric': 'mlogloss',
        'tree_method': 'hist',
        'n_jobs': -1
    })
    
    # CatBoost parameters
    catboost_params: Dict[str, Any] = field(default_factory=lambda: {
        'iterations': 1700,
        'learning_rate': 0.01,
        'depth': 7,
        'l2_leaf_reg': 15.0,
        'random_seed': 42,
        'loss_function': 'MultiClass',
        'eval_metric': 'MultiClass',
        'task_type': 'GPU',  # 'GPU' if you have GPU
        'verbose': False,
        'allow_writing_files': False,  # 不生成中间文件
    })
    
    # LightGBM parameters
    lightgbm_params: Dict[str, Any] = field(default_factory=lambda: {
        'n_estimators': 1500,
        'learning_rate': 0.01,
        'max_depth': 8,
        'num_leaves': 64,  # 通常设为 2^max_depth - 1
        'min_child_samples': 20,    
        'subsample': 0.8,
        'subsample_freq': 1,
        'colsample_bytree': 0.8,
        'reg_alpha': 3.0,
        'reg_lambda': 10.0,
        'random_state': 42,
        'objective': 'multiclass',
        'metric': 'multi_logloss',
        'verbosity': -1,  # 不输出日志
        'n_jobs': -1,
        'force_col_wise': True,  # 避免警告
    })

    # TabNet parameters (深度学习模型)
    # TabNet 优化建议：
    # 1. 使用标准化输入（已在 preprocessor 中添加）
    # 2. 简化特征工程（TabNet 能自动学习特征交互）
    # 3. 适当的学习率和训练轮数
    tabnet_params: Dict[str, Any] = field(default_factory=lambda: {
        # 模型架构参数（传给 TabNetClassifier.__init__）
        'n_d': 16,                  # 决策层维度（降低以减少过拟合）
        'n_a': 16,                  # 注意力层维度（降低以减少过拟合）
        'n_steps': 5,               # 决策步数（降低以减少过拟合，5-7 通常较好）
        'gamma': 1.3,               # 特征选择的松弛因子（降低以减少过拟合）
        'n_independent': 2,         # 独立的 GLU 层数
        'n_shared': 2,              # 共享的 GLU 层数
        'lambda_sparse': 1e-3,      # 稀疏正则化（降低以允许更多特征）
        'momentum': 0.3,            # Batch normalization 动量
        'clip_value': 2.0,          # 梯度裁剪
        'mask_type': 'entmax',      # 掩码类型: 'sparsemax' 或 'entmax'
        'seed': 42,
        'verbose': 0,
        # 训练参数（传给 fit() 方法，需要单独处理）
        '_max_epochs': 250,         # 增加训练轮数（TabNet 需要更多训练）
        '_batch_size': 512,         # 增加批大小（提高稳定性）
        '_patience': 30,            # 增加早停轮数（给模型更多机会）
        '_lr': 0.01,               # 学习率（0.01-0.02 通常较好）
    })
    
    # MLP (Multi-Layer Perceptron) parameters - sklearn neural network
    # Optimized to reduce overfitting, similar to tree models' regularization approach
    mlp_params: Dict[str, Any] = field(default_factory=lambda: {
        'hidden_layer_sizes': (256, 128, 64),         # Reduced from (256, 128, 64) to prevent overfitting
        'activation': 'relu',                    # Activation function: 'relu', 'tanh', 'logistic'
        'solver': 'adam',                        # Optimizer: 'adam', 'lbfgs', 'sgd'
        'alpha': 0.001,                           # Increased L2 regularization (from 0.001) to match tree models' strong regularization
        'batch_size': 'auto',                    # Batch size for 'adam' and 'sgd' solvers
        'learning_rate': 'adaptive',             # Learning rate schedule: 'constant', 'invscaling', 'adaptive'
        'learning_rate_init': 0.001,             # Initial learning rate
        'max_iter': 500,                         # Reduced iterations (from 500) to prevent overfitting
        'shuffle': True,                         # Shuffle samples in each iteration
        'random_state': 42,
        'tol': 1e-4,                            # Tolerance for optimization
        'verbose': False,                        # Print progress
        'warm_start': False,                     # Reuse previous solution
        'momentum': 0.9,                         # Momentum for 'sgd' solver
        'nesterovs_momentum': True,              # Use Nesterov's momentum
        'early_stopping': True,                  # Use early stopping
        'validation_fraction': 0.1,              # Fraction of data for validation
        'beta_1': 0.9,                           # Exponential decay rate for first moment (adam)
        'beta_2': 0.999,                         # Exponential decay rate for second moment (adam)
        'epsilon': 1e-8,                         # Epsilon for numerical stability (adam)
        'n_iter_no_change': 20,                 # Reduced patience (from 20) for earlier stopping
    })
    
    # RandomForest parameters (for ensemble)
    rf_params: Dict[str, Any] = field(default_factory=lambda: {
        'n_estimators': 100,
        'max_depth': 7,
        'min_samples_split': 5,
        'random_state': 42,
        'n_jobs': -1
    })
    
    # KNN (K-Nearest Neighbors) parameters - simple, fast, provides diversity
    knn_params: Dict[str, Any] = field(default_factory=lambda: {
        'n_neighbors': 15,                    # Number of neighbors (balanced between local and global)
        'weights': 'distance',                # Weight by distance (better than uniform)
        'algorithm': 'auto',                  # Auto-select best algorithm
        'leaf_size': 10,                      # Leaf size for tree-based algorithms
        'p': 2,                               # Power parameter for Minkowski metric (2 = Euclidean)
        'metric': 'minkowski',                # Distance metric
        'n_jobs': -1,                         # Use all cores
    })
    
    # Logistic Regression parameters - linear model, fast, provides diversity
    logistic_params: Dict[str, Any] = field(default_factory=lambda: {
        'penalty': 'l2',                      # Regularization: 'l1', 'l2', 'elasticnet', None
        'C': 1.0,                             # Inverse regularization strength (smaller = stronger)
        'solver': 'lbfgs',                    # Solver: 'lbfgs', 'liblinear', 'newton-cg', 'sag', 'saga'
        'max_iter': 1000,                     # Maximum iterations
        # 'multi_class' is deprecated in sklearn 1.5+, removed to avoid warning
        # Default behavior is 'multinomial' for multi-class problems
        'random_state': 42,
        'n_jobs': -1,                         # Use all cores
    })
    
    # SVM (Support Vector Machine) parameters - powerful non-linear classifier
    svm_params: Dict[str, Any] = field(default_factory=lambda: {
        'C': 1.0,                             # Regularization parameter
        'kernel': 'rbf',                      # Kernel: 'linear', 'poly', 'rbf', 'sigmoid'
        'gamma': 'scale',                    # Kernel coefficient: 'scale', 'auto', or float
        'degree': 3,                          # Degree for polynomial kernel
        'coef0': 0.0,                         # Independent term in kernel function
        'probability': True,                 # Enable probability estimates (required for ensemble)
        'random_state': 42,
        'max_iter': -1,                      # -1 means no limit
    })
    
    # Naive Bayes parameters - simple probabilistic classifier
    naive_bayes_params: Dict[str, Any] = field(default_factory=lambda: {
        'var_smoothing': 1e-9,               # Portion of the largest variance to add for stability
    })
    
    # Ridge Classifier parameters - linear model with L2 regularization
    ridge_params: Dict[str, Any] = field(default_factory=lambda: {
        'alpha': 1.0,                        # Regularization strength
        'solver': 'auto',                    # Solver: 'auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'
        'max_iter': None,                    # Maximum iterations
        'random_state': 42,
    })
    
    # Extra Trees (Extremely Randomized Trees) parameters - more randomized than RF
    extra_trees_params: Dict[str, Any] = field(default_factory=lambda: {
        'n_estimators': 100,                 # Number of trees
        'max_depth': 7,                      # Maximum depth
        'min_samples_split': 5,              # Minimum samples to split
        'min_samples_leaf': 2,               # Minimum samples in leaf
        'max_features': 'sqrt',              # Features to consider: 'sqrt', 'log2', None, int, float
        'bootstrap': True,                   # Bootstrap sampling
        'random_state': 42,
        'n_jobs': -1,                        # Use all cores
    })


@dataclass
class FeatureEngineeringConfig:
    """Feature engineering configurations."""
    # Encoding strategies
    freq_encoding_cols: List[str] = field(default_factory=lambda: [
        'RoofType', 
        'ExteriorCovering1', 
        'FoundationType'
    ])
    
    target_encoding_cols: List[str] = field(default_factory=lambda: [
        'ZoningClassification',
        'BuildingType',
        'BuildingStyle',
        'HeatingType'
    ])
    
    target_encoding_alpha: float = 10.0  # Smoothing parameter
    
    # Statistical aggregation
    groupby_cols: List[str] = field(default_factory=lambda: [
        'ZoningClassification',
        'BuildingType'
    ])
    
    agg_cols: List[str] = field(default_factory=lambda: [
        'TotalLivingArea',
        'BuildingAge',
        'OverallQuality'
    ])
    
    # Log transform columns
    log_transform_cols: List[str] = field(default_factory=lambda: ['PlotSize'])


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Train/validation split
    test_size: float = 0.2
    random_state: int = 42
    
    # Cross-validation
    n_splits: int = 5
    shuffle: bool = True
    
    # Class weighting
    use_class_weights: bool = True
    class_weight_power: float = 1.0
    
    # Early stopping
    use_early_stopping: bool = True
    early_stopping_rounds: int = 80
    
    # Verbosity
    verbose: bool = True


@dataclass
class HyperparameterTuningConfig:
    """Hyperparameter tuning configuration."""
    method: str = "optuna"  # "grid", "random", "optuna"
    n_trials: int = 50
    cv_folds: int = 5
    scoring: str = "accuracy"
    timeout: Optional[int] = 3600  # 1 hour timeout
    
    # Search space for XGBoost
    search_space: Dict[str, Any] = field(default_factory=lambda: {
        'learning_rate': (0.01, 0.3),
        'max_depth': (3, 10),
        'min_child_weight': (1, 20),
        'subsample': (0.5, 1.0),
        'colsample_bytree': (0.3, 1.0),
        'gamma': (0, 5),
        'reg_alpha': (0, 10),
        'reg_lambda': (0, 20)
    })


@dataclass
class Config:
    """Main configuration class aggregating all configs."""
    paths: PathConfig = field(default_factory=PathConfig)
    columns: ColumnConfig = field(default_factory=ColumnConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    features: FeatureEngineeringConfig = field(default_factory=FeatureEngineeringConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    tuning: HyperparameterTuningConfig = field(default_factory=HyperparameterTuningConfig)
    
    def __repr__(self):
        return (
            f"Config(\n"
            f"  paths={self.paths},\n"
            f"  columns={self.columns},\n"
            f"  models={self.models},\n"
            f"  features={self.features},\n"
            f"  training={self.training},\n"
            f"  tuning={self.tuning}\n"
            f")"
        )


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration from file or return default config.
    
    Args:
        config_path: Path to configuration file (JSON or YAML). If None, uses defaults.
    
    Returns:
        Config object
    """
    if config_path is None:
        return Config()
    
    # TODO: Implement loading from file if needed
    raise NotImplementedError("Loading config from file not yet implemented")

