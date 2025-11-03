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

