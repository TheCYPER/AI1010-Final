"""
Outlier detection and handling utilities.
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Dict
from sklearn.base import BaseEstimator, TransformerMixin


class OutlierHandler(BaseEstimator, TransformerMixin):
    """
    Handle outliers using IQR or Z-score methods.
    
    Note: In many cases, tree-based models are robust to outliers,
    so this is optional and should be used with caution.
    """
    
    def __init__(
        self,
        method: str = 'iqr',
        threshold: float = 3.0,
        action: str = 'clip',
        columns: Optional[List[str]] = None
    ):
        """
        Initialize outlier handler.
        
        Args:
            method: Detection method ('iqr' or 'zscore')
            threshold: Threshold for outlier detection
                       - For IQR: multiplier for IQR (e.g., 1.5, 3.0)
                       - For zscore: number of standard deviations (e.g., 3.0)
            action: Action to take ('clip', 'remove', or 'nan')
            columns: Columns to check. If None, checks all numeric columns.
        """
        self.method = method
        self.threshold = threshold
        self.action = action
        self.columns = columns
        
        # Fitted attributes
        self.bounds_: Dict[str, tuple] = {}
        self.feature_names_in_ = None
    
    def fit(self, X: pd.DataFrame, y=None):
        """
        Fit the outlier handler by computing bounds.
        
        Args:
            X: Input DataFrame
            y: Target (unused, for compatibility)
        
        Returns:
            self
        """
        self.feature_names_in_ = X.columns.tolist()
        
        # Determine columns to process
        if self.columns is None:
            cols_to_process = X.select_dtypes(include=[np.number]).columns.tolist()
        else:
            cols_to_process = [c for c in self.columns if c in X.columns]
        
        # Compute bounds for each column
        for col in cols_to_process:
            if self.method == 'iqr':
                Q1 = X[col].quantile(0.25)
                Q3 = X[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - self.threshold * IQR
                upper = Q3 + self.threshold * IQR
            elif self.method == 'zscore':
                mean = X[col].mean()
                std = X[col].std()
                lower = mean - self.threshold * std
                upper = mean + self.threshold * std
            else:
                raise ValueError(f"Unknown method: {self.method}")
            
            self.bounds_[col] = (lower, upper)
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data by handling outliers.
        
        Args:
            X: Input DataFrame
        
        Returns:
            Transformed DataFrame
        """
        X = X.copy()
        
        for col, (lower, upper) in self.bounds_.items():
            if col not in X.columns:
                continue
            
            if self.action == 'clip':
                X[col] = X[col].clip(lower=lower, upper=upper)
            elif self.action == 'nan':
                mask = (X[col] < lower) | (X[col] > upper)
                X.loc[mask, col] = np.nan
            elif self.action == 'remove':
                # Note: 'remove' should be used carefully as it changes row count
                mask = (X[col] >= lower) & (X[col] <= upper)
                X = X[mask]
            else:
                raise ValueError(f"Unknown action: {self.action}")
        
        return X
    
    def get_outlier_info(self, X: pd.DataFrame) -> Dict[str, Dict]:
        """
        Get information about outliers in the data.
        
        Args:
            X: Input DataFrame
        
        Returns:
            Dictionary with outlier statistics for each column
        """
        info = {}
        
        for col, (lower, upper) in self.bounds_.items():
            if col not in X.columns:
                continue
            
            below = (X[col] < lower).sum()
            above = (X[col] > upper).sum()
            total = len(X)
            
            info[col] = {
                'lower_bound': lower,
                'upper_bound': upper,
                'n_outliers_below': int(below),
                'n_outliers_above': int(above),
                'n_outliers_total': int(below + above),
                'pct_outliers': float((below + above) / total * 100)
            }
        
        return info

