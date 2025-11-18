"""
Miscellaneous transformers.
"""

import numpy as np
import pandas as pd
from typing import Sequence, Optional, List
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer


class Log1pTransformer(BaseEstimator, TransformerMixin):
    """
    Apply log1p transformation to specific columns.
    
    Useful for skewed numeric features like area or price.
    """
    
    def __init__(
        self,
        col_names: Sequence[str],
        target_col: str = "PlotSize",
        clip_min: float = 0.0
    ):
        """
        Initialize log1p transformer.
        
        Args:
            col_names: All column names in the input
            target_col: Column to transform
            clip_min: Minimum value to clip before log1p
        """
        self.col_names = tuple(col_names)  # Immutable for sklearn clone
        self.target_col = target_col
        self.clip_min = clip_min
        self._idx: Optional[int] = None
    
    def fit(self, X, y=None):
        """
        Fit by finding the index of target column.
        
        Args:
            X: Input data
            y: Target (unused)
        
        Returns:
            self
        """
        if self.target_col in self.col_names:
            self._idx = self.col_names.index(self.target_col)
        else:
            self._idx = None
        
        return self
    
    def transform(self, X) -> np.ndarray:
        """
        Transform by applying log1p to target column.
        
        Args:
            X: Input data
        
        Returns:
            Transformed array
        """
        A = np.asarray(X, dtype=float)
        
        if self._idx is not None and self._idx < A.shape[1]:
            A = A.copy()
            v = A[:, self._idx]
            v = np.log1p(np.clip(v, self.clip_min, None))
            A[:, self._idx] = v
        
        return A


class BusinessMissingIndicator(BaseEstimator, TransformerMixin):
    """
    Transformer for creating business missing indicator.
    
    Creates a binary indicator for missing values that have business meaning.
    This class is picklable, unlike a FunctionTransformer with a closure.
    """
    
    def __init__(self):
        """Initialize business missing indicator."""
        pass
    
    def fit(self, X, y=None):
        """
        Fit is a no-op for this transformer.
        
        Args:
            X: Input data
            y: Target (unused)
        
        Returns:
            self
        """
        return self
    
    def transform(self, X):
        """
        Transform by creating missing indicator.
        
        Args:
            X: Input data (DataFrame, Series, or array)
        
        Returns:
            Binary array indicating missingness
        """
        if isinstance(X, pd.DataFrame):
            s = X.iloc[:, 0]
        elif isinstance(X, pd.Series):
            s = X
        else:
            s = pd.Series(np.asarray(X).ravel())
        
        s2 = s.copy()
        
        # Handle string columns
        mask_str = s2.apply(lambda v: isinstance(v, str))
        s2.loc[mask_str] = s2.loc[mask_str].str.strip()
        
        # Identify missing
        missing = s2.eq("").fillna(False) | s2.isna()
        
        return missing.astype(int).to_numpy().reshape(-1, 1)
    
    def get_feature_names_out(self, input_features=None):
        """Get output feature names."""
        return np.array(["business_missing_indicator"])


class KNNImputerWithIndicators(BaseEstimator, TransformerMixin):
    """
    Combined transformer that performs KNN imputation and adds missing indicators.
    
    This wraps KNNImputer and adds missing value indicators, since KNNImputer
    doesn't support add_indicator directly.
    """
    
    def __init__(self, n_neighbors: int = 5, weights: str = 'uniform', col_names: List[str] = None):
        """
        Initialize KNN imputer with indicators.
        
        Args:
            n_neighbors: Number of neighbors for KNN imputation
            weights: Weight function for KNN ('uniform' or 'distance')
            col_names: Column names for creating indicator feature names
        """
        from sklearn.impute import KNNImputer
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.col_names = col_names or []
        self.imputer_ = KNNImputer(n_neighbors=n_neighbors, weights=weights)
        self.missing_mask_ = None
        self.n_features_in_ = None
    
    def fit(self, X, y=None):
        """
        Fit by storing missing pattern and fitting KNN imputer.
        
        Args:
            X: Input data (numpy array)
            y: Target (unused)
        
        Returns:
            self
        """
        X = np.asarray(X)
        self.n_features_in_ = X.shape[1]
        
        # Store missing pattern (before imputation)
        self.missing_mask_ = np.isnan(X)
        
        # Fit KNN imputer
        self.imputer_.fit(X)
        
        return self
    
    def transform(self, X):
        """
        Transform by imputing and appending missing indicators.
        
        Args:
            X: Input data (numpy array)
        
        Returns:
            Array with imputed features + missing indicators
        """
        X = np.asarray(X)
        
        # Get missing indicators (from fit time for training, current for test)
        if self.missing_mask_ is not None and X.shape[0] == self.missing_mask_.shape[0]:
            # Training data: use stored missing pattern
            missing_indicators = self.missing_mask_.astype(int)
        else:
            # Test data: use current missing pattern
            missing_indicators = np.isnan(X).astype(int)
        
        # Impute missing values
        X_imputed = self.imputer_.transform(X)
        
        # Combine imputed values and indicators
        X_with_indicators = np.hstack([X_imputed, missing_indicators])
        
        return X_with_indicators
    
    def get_feature_names_out(self, input_features=None):
        """Get output feature names."""
        if input_features is None:
            input_features = [f"feature_{i}" for i in range(self.n_features_in_)]
        
        output_names = list(input_features) + [f"{col}_missing" for col in self.col_names]
        return np.array(output_names)


class MissingIndicatorTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer to add missing value indicators for numeric columns.
    
    This is used with KNNImputer, which doesn't support add_indicator directly.
    The transformer stores the missing pattern during fit and adds indicators during transform.
    """
    
    def __init__(self, col_names: List[str]):
        """
        Initialize missing indicator transformer.
        
        Args:
            col_names: List of column names to create indicators for
        """
        self.col_names = col_names
        self.n_features_in_ = None
        self.missing_mask_ = None
    
    def fit(self, X, y=None):
        """
        Fit by storing the missing pattern.
        
        Args:
            X: Input data (numpy array)
            y: Target (unused)
        
        Returns:
            self
        """
        X = np.asarray(X)
        self.n_features_in_ = X.shape[1]
        
        # Store missing pattern (before imputation)
        self.missing_mask_ = np.isnan(X)
        
        return self
    
    def transform(self, X):
        """
        Transform by appending missing indicators.
        
        Args:
            X: Input data (numpy array, already imputed)
        
        Returns:
            Array with original features + missing indicators
        """
        X = np.asarray(X)
        
        # Get missing indicators (from fit time)
        if self.missing_mask_ is not None:
            missing_indicators = self.missing_mask_.astype(int)
            # Combine imputed values and indicators
            X_with_indicators = np.hstack([X, missing_indicators])
        else:
            # Fallback: check current missing values
            missing_indicators = np.isnan(X).astype(int)
            X_with_indicators = np.hstack([X, missing_indicators])
        
        return X_with_indicators
    
    def get_feature_names_out(self, input_features=None):
        """Get output feature names."""
        if input_features is None:
            input_features = [f"feature_{i}" for i in range(self.n_features_in_)]
        
        output_names = list(input_features) + [f"{col}_missing" for col in self.col_names]
        return np.array(output_names)

