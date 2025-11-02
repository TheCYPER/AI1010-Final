"""
Feature auditing and importance analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance

from utils.logger import get_logger

logger = get_logger(__name__)


class FeatureAuditor:
    """
    Feature auditing toolkit.
    
    Provides methods for:
    - Feature importance analysis
    - Drift detection (adversarial validation)
    - Leakage detection
    - Correlation analysis
    """
    
    def __init__(self, config=None):
        """
        Initialize feature auditor.
        
        Args:
            config: Configuration object (optional)
        """
        self.config = config
        self.audit_results_ = {}
    
    def compute_feature_importance(
        self,
        model,
        feature_names: Optional[list] = None
    ) -> pd.Series:
        """
        Get feature importance from trained model.
        
        Args:
            model: Trained model (can be our wrapper or sklearn/xgb model)
            feature_names: Feature names (optional)
        
        Returns:
            Series of feature importances
        """
        logger.info("Computing feature importance...")
        
        # Try different methods to get feature importance
        importance = None
        
        # Method 1: Try our wrapper's method
        if hasattr(model, 'get_feature_importance'):
            importance = model.get_feature_importance()
        
        # Method 2: Try sklearn/xgb attribute
        if importance is None and hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        
        # Method 3: Try coef_ (for linear models)
        if importance is None and hasattr(model, 'coef_'):
            importance = np.abs(model.coef_).ravel()
        
        if importance is None:
            logger.warning("Model does not support feature importance")
            return pd.Series()
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importance))]
        
        importance_series = pd.Series(
            importance,
            index=feature_names
        ).sort_values(ascending=False)
        
        logger.info(f"Top 10 important features:")
        for feat, imp in importance_series.head(10).items():
            logger.info(f"  {feat}: {imp:.4f}")
        
        self.audit_results_['feature_importance'] = importance_series.to_dict()
        
        return importance_series
    
    def compute_permutation_importance(
        self,
        model,
        X_val,
        y_val,
        feature_names: Optional[list] = None,
        n_repeats: int = 5
    ) -> pd.DataFrame:
        """
        Compute permutation importance on validation set.
        
        Args:
            model: Trained model
            X_val: Validation features
            y_val: Validation target
            feature_names: Feature names (optional)
            n_repeats: Number of permutation repeats
        
        Returns:
            DataFrame with importance statistics
        """
        logger.info(f"Computing permutation importance ({n_repeats} repeats)...")
        
        # Get underlying model
        if hasattr(model, 'model_'):
            sklearn_model = model.model_
        else:
            sklearn_model = model
        
        result = permutation_importance(
            sklearn_model,
            X_val,
            y_val,
            n_repeats=n_repeats,
            scoring='accuracy',
            random_state=42,
            n_jobs=-1
        )
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(result.importances_mean))]
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance_mean': result.importances_mean,
            'importance_std': result.importances_std
        }).sort_values('importance_mean', ascending=False)
        
        logger.info("Top 10 features by permutation importance:")
        for _, row in importance_df.head(10).iterrows():
            logger.info(f"  {row['feature']}: {row['importance_mean']:.4f} ± {row['importance_std']:.4f}")
        
        self.audit_results_['permutation_importance'] = importance_df.to_dict('records')
        
        return importance_df
    
    def adversarial_validation(
        self,
        X_train,
        X_val
    ) -> Dict[str, Any]:
        """
        Perform adversarial validation to detect train/val drift.
        
        Args:
            X_train: Training features
            X_val: Validation features
        
        Returns:
            Dictionary with AUC score and interpretation
        """
        logger.info("Running adversarial validation...")
        
        # Combine data
        Z = np.vstack([np.asarray(X_train), np.asarray(X_val)])
        d = np.r_[np.zeros(len(X_train)), np.ones(len(X_val))]
        
        # Train logistic regression to distinguish train from val
        from sklearn.model_selection import train_test_split
        Z_train, Z_test, d_train, d_test = train_test_split(
            Z, d, test_size=0.3, stratify=d, random_state=42
        )
        
        adv_model = LogisticRegression(max_iter=2000, random_state=42)
        adv_model.fit(Z_train, d_train)
        
        proba = adv_model.predict_proba(Z_test)[:, 1]
        auc = roc_auc_score(d_test, proba)
        
        # Interpret AUC
        if auc < 0.55:
            interpretation = "No significant drift (excellent)"
        elif auc < 0.65:
            interpretation = "Mild drift (acceptable)"
        elif auc < 0.75:
            interpretation = "Moderate drift (caution)"
        else:
            interpretation = "High drift (warning!)"
        
        logger.info(f"Adversarial AUC: {auc:.4f} - {interpretation}")
        
        result = {
            'auc': float(auc),
            'interpretation': interpretation
        }
        
        self.audit_results_['adversarial_validation'] = result
        
        return result
    
    def check_correlation(
        self,
        X: pd.DataFrame,
        threshold: float = 0.95
    ) -> Tuple[pd.DataFrame, list]:
        """
        Check for highly correlated features.
        
        Args:
            X: Feature matrix (numeric only)
            threshold: Correlation threshold
        
        Returns:
            Tuple of (correlation matrix, list of high correlation pairs)
        """
        logger.info(f"Checking correlations (threshold={threshold})...")
        
        # Select numeric columns if DataFrame
        if isinstance(X, pd.DataFrame):
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            X_numeric = X[numeric_cols]
        else:
            X_numeric = pd.DataFrame(X)
        
        # Compute correlation
        corr = X_numeric.corr()
        
        # Find high correlations
        high_corr_pairs = []
        upper_tri = corr.where(
            np.triu(np.ones(corr.shape), k=1).astype(bool)
        )
        
        for col in upper_tri.columns:
            high_corr_features = upper_tri.index[
                np.abs(upper_tri[col]) > threshold
            ].tolist()
            
            for feat in high_corr_features:
                high_corr_pairs.append({
                    'feature1': col,
                    'feature2': feat,
                    'correlation': float(corr.loc[feat, col])
                })
        
        logger.info(f"Found {len(high_corr_pairs)} pairs with |correlation| > {threshold}")
        
        if high_corr_pairs:
            logger.info("High correlation pairs:")
            for pair in high_corr_pairs[:10]:  # Show top 10
                logger.info(f"  {pair['feature1']} <-> {pair['feature2']}: {pair['correlation']:.3f}")
        
        self.audit_results_['high_correlations'] = high_corr_pairs
        
        return corr, high_corr_pairs
    
    def save_audit_report(self, output_path: Optional[str] = None):
        """
        Save audit results to file.
        
        Args:
            output_path: Path to save report
        """
        if output_path is None and self.config:
            output_path = f"{self.config.paths.output_dir}/feature_audit.json"
        
        import json
        with open(output_path, 'w') as f:
            json.dump(self.audit_results_, f, indent=2)
        
        logger.info(f"Saved audit report to {output_path}")

