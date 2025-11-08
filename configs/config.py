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
    
    # Columns to drop (low quality/redundant)
    drop_columns: List[str] = field(default_factory=lambda: [
        "AlleyAccess",
        "ExteriorFinishType", 
        "RecreationQuality",
        "MiscellaneousFeature"
    ])
    
    # Business missing column
    business_missing_col: str = "ConferenceRoomQuality"


@dataclass
class ModelConfig:
    """Model-specific configurations."""
    # Model type selection
    model_type: str = "tabnet"  # Options: "xgboost", "catboost", "lightgbm", "tabnet", "ensemble"
    
    # XGBoost parameters
    xgb_params: Dict[str, Any] = field(default_factory=lambda: {
        'objective': 'multi:softprob',
        'n_estimators': 1500,
        'learning_rate': 0.06,
        'max_depth': 4,
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
        'learning_rate': 0.03,
        'depth': 7,
        'l2_leaf_reg': 15.0,
        'random_seed': 42,
        'loss_function': 'MultiClass',
        'eval_metric': 'MultiClass',
        'task_type': 'CPU',  # 'GPU' if you have GPU
        'verbose': False,
        'allow_writing_files': False,  # 不生成中间文件
    })
    
    # LightGBM parameters
    lightgbm_params: Dict[str, Any] = field(default_factory=lambda: {
        'n_estimators': 1500,
        'learning_rate': 0.05,
        'max_depth': 7,
        'num_leaves': 63,  # 通常设为 2^max_depth - 1
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
    tabnet_params: Dict[str, Any] = field(default_factory=lambda: {
        # 模型架构参数（传给 TabNetClassifier.__init__）
        'n_d': 64,                  # 决策层维度
        'n_a': 64,                  # 注意力层维度
        'n_steps': 5,               # 决策步数（类似树的深度）
        'gamma': 1.5,               # 特征选择的松弛因子
        'n_independent': 2,         # 独立的 GLU 层数
        'n_shared': 2,              # 共享的 GLU 层数
        'lambda_sparse': 1e-4,      # 稀疏正则化
        'momentum': 0.3,            # Batch normalization 动量
        'clip_value': 2.0,          # 梯度裁剪
        'mask_type': 'entmax',      # 掩码类型: 'sparsemax' 或 'entmax'
        'seed': 42,
        'verbose': 0,
        # 训练参数（传给 fit() 方法，需要单独处理）
        '_max_epochs': 100,         # 最大训练轮数
        '_batch_size': 256,         # 批大小
        '_patience': 20,            # 早停轮数
        '_lr': 2e-2,               # 学习率
    })
    
    # RandomForest parameters (for ensemble)
    rf_params: Dict[str, Any] = field(default_factory=lambda: {
        'n_estimators': 200,
        'max_depth': 20,
        'min_samples_split': 5,
        'random_state': 42,
        'n_jobs': -1
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

