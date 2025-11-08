"""
Cross-validation trainer.
"""

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Tuple
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from utils.logger import get_logger
from utils.metrics import evaluate_model, aggregate_cv_metrics

logger = get_logger(__name__)


class CrossValidator:
    """
    Cross-validation trainer.
    
    Performs k-fold cross-validation and aggregates results.
    """
    
    def __init__(self, config):
        """
        Initialize cross-validator.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.fold_results_: List[Dict[str, Any]] = []
        self.aggregated_results_: Dict[str, Any] = {}
    
    def _create_model(self, num_classes: int):
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
        elif model_type == "catboost":
            from modeling.catboost_model import CatBoostModel
            model = CatBoostModel(config=self.config.models.catboost_params)
        elif model_type == "lightgbm":
            from modeling.lightgbm_model import LightGBMModel
            model = LightGBMModel(config=self.config.models.lightgbm_params)
        elif model_type == "tabnet":
            from modeling.tabnet_model import TabNetModel
            model = TabNetModel(config=self.config.models.tabnet_params)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model.build_model(num_classes=num_classes)
        return model
    
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
    
    def compute_sample_weights(self, y: pd.Series) -> np.ndarray:
        """
        Compute sample weights for class balancing.
        
        Args:
            y: Target labels
        
        Returns:
            Sample weights or None
        """
        if not self.config.training.use_class_weights:
            return None
        
        cls, cnt = np.unique(y, return_counts=True)
        total = len(y)
        power = self.config.training.class_weight_power
        
        wmap = {c: (total / cnt[i]) ** power for i, c in enumerate(cls)}
        w = y.map(wmap).astype(float).values
        w = w / np.mean(w)
        
        return w
    
    def train_fold(
        self,
        fold_idx: int,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        num_cols: List[str],
        cat_cols: List[str]
    ) -> Dict[str, Any]:
        """
        Train a single fold.
        
        Args:
            fold_idx: Fold index
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            num_cols: Numeric columns
            cat_cols: Categorical columns
        
        Returns:
            Fold results
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"TRAINING FOLD {fold_idx}")
        logger.info(f"{'='*70}")
        
        from feature_engineering import build_preprocessor
        
        # Build preprocessor
        preprocessor = build_preprocessor(
            num_cols=num_cols,
            cat_cols=cat_cols,
            drop_cols=self.config.columns.drop_columns,
            freq_encoding_cols=self.config.features.freq_encoding_cols,
            target_encoding_cols=self.config.features.target_encoding_cols,
            target_encoding_alpha=self.config.features.target_encoding_alpha,
            business_missing_col=self.config.columns.business_missing_col,
            log_transform_cols=self.config.features.log_transform_cols
        )
        
        # Fit preprocessor
        X_train_transformed = preprocessor.fit_transform(X_train, y_train)
        X_val_transformed = preprocessor.transform(X_val)
        
        logger.info(f"Transformed: train={X_train_transformed.shape}, val={X_val_transformed.shape}")
        
        # Compute sample weights
        sample_weight = self.compute_sample_weights(y_train)
        
        # Build and train model based on config
        num_classes = len(np.unique(y_train))
        model = self._create_model(num_classes)
        
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
        
        # Evaluate
        results = evaluate_model(
            model,
            X_train_transformed,
            y_train,
            X_val_transformed,
            y_val,
            verbose=True
        )
        
        # Add fold info
        results['fold'] = fold_idx
        results['n_train'] = len(y_train)
        results['n_val'] = len(y_val)
        
        # Build pipeline
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", model.model_)
        ])
        
        results['pipeline'] = pipeline
        
        return results
    
    def run(self, save_per_fold: bool = False) -> Dict[str, Any]:
        """
        Run cross-validation.
        
        Args:
            save_per_fold: Whether to save artifacts for each fold
        
        Returns:
            Aggregated results
        """
        logger.info("="*70)
        logger.info("STARTING CROSS-VALIDATION")
        logger.info("="*70)
        
        # Load data
        X, y = self.load_data()
        
        # Infer column types
        from data_cleaning import infer_column_types
        temp_df = pd.concat(
            [X, pd.Series(y, name=self.config.columns.target)],
            axis=1
        )
        num_cols, cat_cols = infer_column_types(
            temp_df,
            self.config.columns.target
        )
        
        logger.info(f"Numeric features: {len(num_cols)}")
        logger.info(f"Categorical features: {len(cat_cols)}")
        
        # Create stratified k-fold
        n_splits = self.config.training.n_splits
        shuffle = self.config.training.shuffle
        random_state = self.config.training.random_state
        
        skf = StratifiedKFold(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_state
        )
        
        logger.info(f"Using {n_splits}-fold cross-validation")
        
        # Train each fold
        self.fold_results_ = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
            X_train = X.iloc[train_idx].copy()
            X_val = X.iloc[val_idx].copy()
            y_train = y.iloc[train_idx].copy()
            y_val = y.iloc[val_idx].copy()
            
            # Train fold
            fold_results = self.train_fold(
                fold_idx,
                X_train, y_train,
                X_val, y_val,
                num_cols, cat_cols
            )
            
            # Save fold results
            if save_per_fold:
                self._save_fold_artifacts(fold_idx, fold_results)
            
            # Remove pipeline from results for aggregation
            pipeline = fold_results.pop('pipeline')
            
            self.fold_results_.append(fold_results)
        
        # Aggregate results
        self._aggregate_results()
        
        # Save aggregated results
        self._save_aggregated_results()
        
        logger.info("="*70)
        logger.info("CROSS-VALIDATION COMPLETE")
        logger.info("="*70)
        logger.info(f"\nFinal Results:")
        logger.info(f"  Accuracy: {self.aggregated_results_['accuracy_mean']:.4f} ± {self.aggregated_results_['accuracy_std']:.4f}")
        
        return self.aggregated_results_
    
    def _aggregate_results(self):
        """Aggregate results across folds."""
        # Extract metrics from each fold
        train_accs = [r['train_metrics']['accuracy'] for r in self.fold_results_]
        val_accs = [r['val_metrics']['accuracy'] for r in self.fold_results_]
        
        # Aggregate
        self.aggregated_results_ = {
            'n_splits': self.config.training.n_splits,
            'accuracy_mean': float(np.mean(val_accs)),
            'accuracy_std': float(np.std(val_accs)),
            'accuracy_min': float(np.min(val_accs)),
            'accuracy_max': float(np.max(val_accs)),
            'train_accuracy_mean': float(np.mean(train_accs)),
            'train_accuracy_std': float(np.std(train_accs)),
            'per_fold_val_accuracy': [float(a) for a in val_accs],
            'per_fold_train_accuracy': [float(a) for a in train_accs],
            'fold_details': self.fold_results_
        }
    
    def _save_fold_artifacts(self, fold_idx: int, results: Dict[str, Any]):
        """Save artifacts for a single fold."""
        fold_dir = Path(self.config.paths.models_dir) / "cv" / f"fold_{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        
        # Save pipeline
        pipeline_path = fold_dir / "pipeline.joblib"
        joblib.dump(results['pipeline'], pipeline_path)
        
        # Save metrics
        metrics_path = fold_dir / "metrics.json"
        metrics = {
            'fold': fold_idx,
            'train_accuracy': float(results['train_metrics']['accuracy']),
            'val_accuracy': float(results['val_metrics']['accuracy']),
            'train_metrics': {k: float(v) for k, v in results['train_metrics'].items()},
            'val_metrics': {k: float(v) for k, v in results['val_metrics'].items()}
        }
        
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
    
    def _save_aggregated_results(self):
        """Save aggregated cross-validation results."""
        cv_dir = Path(self.config.paths.models_dir) / "cv"
        cv_dir.mkdir(parents=True, exist_ok=True)
        
        summary_path = cv_dir / "cv_summary.json"
        
        # Prepare summary (without full fold details)
        summary = {
            'n_splits': self.aggregated_results_['n_splits'],
            'accuracy_mean': self.aggregated_results_['accuracy_mean'],
            'accuracy_std': self.aggregated_results_['accuracy_std'],
            'accuracy_min': self.aggregated_results_['accuracy_min'],
            'accuracy_max': self.aggregated_results_['accuracy_max'],
            'train_accuracy_mean': self.aggregated_results_['train_accuracy_mean'],
            'train_accuracy_std': self.aggregated_results_['train_accuracy_std'],
            'per_fold_val_accuracy': self.aggregated_results_['per_fold_val_accuracy'],
            'per_fold_train_accuracy': self.aggregated_results_['per_fold_train_accuracy']
        }
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Saved CV summary to {summary_path}")

