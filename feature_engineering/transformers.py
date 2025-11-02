"""
Miscellaneous transformers.
"""

import numpy as np
import pandas as pd
from typing import Sequence, Optional
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

