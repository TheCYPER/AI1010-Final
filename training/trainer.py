"""
Model trainer for single train/validation split.
"""

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from utils.logger import get_logger
from utils.metrics import evaluate_model

logger = get_logger(__name__)


class Trainer:
    """
    Trainer for single train/validation split.
    
    Handles data loading, preprocessing, model training, and evaluation.
    """
    
    def __init__(self, config):
        """
        Initialize trainer.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.preprocessor_ = None
        self.model_ = None
        self.pipeline_ = None
        self.train_results_ = None
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load training data.
        
        Returns:
            Tuple of (features, target)
        """
        logger.info(f"Loading data from {self.config.paths.train_csv}")
        
        df = pd.read_csv(self.config.paths.train_csv)
        target = self.config.columns.target
        
        X = df.drop(columns=[target])
        y = df[target]
        
        logger.info(f"Data loaded: {X.shape}")
        logger.info(f"Target distribution:\n{y.value_counts().sort_index()}")
        
        return X, y
    
    def split_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: Optional[float] = None,
        random_state: Optional[int] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into train and validation sets.
        
        Args:
            X: Features
            y: Target
            test_size: Validation set size
            random_state: Random seed
        
        Returns:
            Tuple of (X_train, X_val, y_train, y_val)
        """
        test_size = test_size or self.config.training.test_size
        random_state = random_state or self.config.training.random_state
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )
        
        logger.info(f"Train set: {X_train.shape}")
        logger.info(f"Validation set: {X_val.shape}")
        
        return X_train, X_val, y_train, y_val
    
    def build_preprocessor(self, X: pd.DataFrame, num_cols=None, cat_cols=None):
        """
        Build preprocessing pipeline.
        
        Args:
            X: Sample data for column type inference
            num_cols: Numeric columns (auto-inferred if None)
            cat_cols: Categorical columns (auto-inferred if None)
        
        Returns:
            Preprocessor
        """
        from data_cleaning import infer_column_types
        from feature_engineering import build_preprocessor
        
        # Infer column types if not provided
        if num_cols is None or cat_cols is None:
            # Create a temporary DataFrame with target for inference
            temp_df = pd.concat(
                [X, pd.Series([0]*len(X), name=self.config.columns.target)],
                axis=1
            )
            num_cols, cat_cols = infer_column_types(
                temp_df,
                self.config.columns.target
            )
        
        logger.info(f"Numeric features: {len(num_cols)}")
        logger.info(f"Categorical features: {len(cat_cols)}")
        
        # Build preprocessor
        self.preprocessor_ = build_preprocessor(
            num_cols=num_cols,
            cat_cols=cat_cols,
            drop_cols=self.config.columns.drop_columns,
            freq_encoding_cols=self.config.features.freq_encoding_cols,
            target_encoding_cols=self.config.features.target_encoding_cols,
            target_encoding_alpha=self.config.features.target_encoding_alpha,
            business_missing_col=self.config.columns.business_missing_col,
            log_transform_cols=self.config.features.log_transform_cols
        )
        
        return self.preprocessor_
    
    def compute_sample_weights(self, y: pd.Series) -> np.ndarray:
        """
        Compute sample weights for class balancing.
        
        Args:
            y: Target labels
        
        Returns:
            Sample weights
        """
        if not self.config.training.use_class_weights:
            return None
        
        cls, cnt = np.unique(y, return_counts=True)
        total = len(y)
        power = self.config.training.class_weight_power
        
        wmap = {c: (total / cnt[i]) ** power for i, c in enumerate(cls)}
        w = y.map(wmap).astype(float).values
        w = w / np.mean(w)  # Normalize to mean=1
        
        logger.info("Using class weights for training")
        return w
    
    def train(
        self,
        model,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        num_cols=None,
        cat_cols=None
    ) -> Dict[str, Any]:
        """
        Train model with preprocessing.
        
        Args:
            model: Model instance
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            num_cols: Numeric columns
            cat_cols: Categorical columns
        
        Returns:
            Training results dictionary
        """
        logger.info("="*70)
        logger.info("STARTING TRAINING")
        logger.info("="*70)
        
        # Build preprocessor
        self.build_preprocessor(X_train, num_cols, cat_cols)
        
        # Fit preprocessor on training data
        logger.info("Fitting preprocessor...")
        X_train_transformed = self.preprocessor_.fit_transform(X_train, y_train)
        X_val_transformed = self.preprocessor_.transform(X_val)
        
        logger.info(f"Transformed shapes: train={X_train_transformed.shape}, val={X_val_transformed.shape}")
        
        # Compute sample weights
        sample_weight = self.compute_sample_weights(y_train)
        
        # Train model
        logger.info("Training model...")
        
        fit_kwargs = {
            'X': X_train_transformed,
            'y': y_train,
            'eval_set': [(X_val_transformed, y_val)],
            'verbose': False
        }
        
        if sample_weight is not None:
            fit_kwargs['sample_weight'] = sample_weight
        
        if self.config.training.use_early_stopping:
            fit_kwargs['early_stopping_rounds'] = self.config.training.early_stopping_rounds
        
        model.fit(**fit_kwargs)
        self.model_ = model
        
        # Evaluate
        logger.info("Evaluating model...")
        results = evaluate_model(
            model,
            X_train_transformed,
            y_train,
            X_val_transformed,
            y_val,
            verbose=self.config.training.verbose
        )
        
        # Build pipeline
        self.pipeline_ = Pipeline(steps=[
            ("preprocessor", self.preprocessor_),
            ("model", model.model_)  # Extract underlying sklearn/xgb model
        ])
        
        self.train_results_ = results
        
        return results
    
    def save_artifacts(
        self,
        results: Dict[str, Any],
        pipeline_path: Optional[str] = None,
        metrics_path: Optional[str] = None
    ):
        """
        Save training artifacts.
        
        Args:
            results: Training results
            pipeline_path: Path to save pipeline
            metrics_path: Path to save metrics
        """
        # Default paths
        if pipeline_path is None:
            pipeline_path = Path(self.config.paths.models_dir) / "pipeline.joblib"
        if metrics_path is None:
            metrics_path = Path(self.config.paths.metrics_dir) / "metrics.json"
        
        # Save pipeline
        logger.info(f"Saving pipeline to {pipeline_path}")
        joblib.dump(self.pipeline_, pipeline_path)
        
        # Save metrics
        logger.info(f"Saving metrics to {metrics_path}")
        
        # Convert numpy types to Python types for JSON serialization
        metrics_to_save = {
            'train_accuracy': float(results['train_metrics']['accuracy']),
            'val_accuracy': float(results['val_metrics']['accuracy']),
            'train_metrics': {k: float(v) for k, v in results['train_metrics'].items()},
            'val_metrics': {k: float(v) for k, v in results['val_metrics'].items()},
            'val_report': results['val_report'],
            'confusion_matrix': results['confusion_matrix']
        }
        
        with open(metrics_path, 'w') as f:
            json.dump(metrics_to_save, f, indent=2)
        
        logger.info("Artifacts saved successfully")
    
    def create_model(self, num_classes: int):
        """
        Create model based on config.
        
        Args:
            num_classes: Number of classes
        
        Returns:
            Model instance
        """
        model_type = self.config.models.model_type.lower()
        
        if model_type == "xgboost":
            from modeling import XGBoostModel
            model = XGBoostModel(config=self.config.models.xgb_params)
            logger.info("Using XGBoost model")
        elif model_type == "catboost":
            from modeling.catboost_model import CatBoostModel
            model = CatBoostModel(config=self.config.models.catboost_params)
            logger.info("Using CatBoost model")
        elif model_type == "lightgbm":
            from modeling.lightgbm_model import LightGBMModel
            model = LightGBMModel(config=self.config.models.lightgbm_params)
            logger.info("Using LightGBM model")
        elif model_type == "tabnet":
            from modeling.tabnet_model import TabNetModel
            model = TabNetModel(config=self.config.models.tabnet_params)
            logger.info("Using TabNet model (Deep Learning)")
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model.build_model(num_classes=num_classes)
        return model
    
    def run(self) -> Dict[str, Any]:
        """
        Run complete training pipeline.
        
        Returns:
            Training results
        """
        # Load data
        X, y = self.load_data()
        
        # Split data
        X_train, X_val, y_train, y_val = self.split_data(X, y)
        
        # Create model based on config
        num_classes = len(np.unique(y))
        model = self.create_model(num_classes)
        
        # Train
        results = self.train(
            model,
            X_train, y_train,
            X_val, y_val
        )
        
        # Save artifacts
        self.save_artifacts(results)
        
        logger.info("="*70)
        logger.info("TRAINING COMPLETE")
        logger.info("="*70)
        
        return results

