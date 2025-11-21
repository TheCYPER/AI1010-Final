"""
Missing value handling utilities.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer, KNNImputer


class MissingValueHandler(BaseEstimator, TransformerMixin):
    """
    Handle missing values with different strategies for numeric and categorical columns.
    
    Features:
    - Separate strategies for numeric and categorical columns
    - Optional missing indicators
    - Business-aware pre-fills and rare-category collapsing
    """
    
    def __init__(
        self,
        numeric_strategy: str = 'median',
        categorical_strategy: str = 'constant',
        categorical_fill_value: str = '__MISSING__',
        add_indicator: bool = True,
        business_missing_col: Optional[str] = None,
        use_knn: bool = False,
        knn_neighbors: int = 5,
        numeric_fill_map: Optional[Dict[str, float]] = None,
        categorical_fill_map: Optional[Dict[str, str]] = None,
        rare_category_threshold: float = 0.01,
        rare_category_name: str = '__RARE__'
    ):
        """
        Initialize missing value handler.
        """
        self.numeric_strategy = numeric_strategy
        self.categorical_strategy = categorical_strategy
        self.categorical_fill_value = categorical_fill_value
        self.add_indicator = add_indicator
        self.business_missing_col = business_missing_col
        self.use_knn = use_knn
        self.knn_neighbors = knn_neighbors
        self.numeric_fill_map = numeric_fill_map or {}
        self.categorical_fill_map = categorical_fill_map or {}
        self.rare_category_threshold = rare_category_threshold
        self.rare_category_name = rare_category_name
        
        # Fitted attributes
        self.numeric_imputer_ = None
        self.categorical_imputer_ = None
        self.numeric_cols_: List[str] = []
        self.categorical_cols_: List[str] = []
        self.feature_names_in_: Optional[List[str]] = None
        self.numeric_all_nan_cols_: List[str] = []
        self.categorical_all_nan_cols_: List[str] = []
        self.rare_categories_: Dict[str, List[Any]] = {}
    
    def fit(self, X: pd.DataFrame, y=None):
        """Fit the missing value handlers."""
        self.feature_names_in_ = X.columns.tolist()
        
        # Separate numeric and categorical columns
        self.numeric_cols_ = X.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols_ = X.select_dtypes(exclude=[np.number]).columns.tolist()
        
        # Track entirely missing columns
        self.numeric_all_nan_cols_ = [c for c in self.numeric_cols_ if X[c].isna().all()]
        self.categorical_all_nan_cols_ = [c for c in self.categorical_cols_ if X[c].isna().all()]
        
        # Detect rare categories
        self.rare_categories_ = {}
        for col in self.categorical_cols_:
            counts = X[col].value_counts(normalize=True, dropna=True)
            rare_vals = counts[counts < self.rare_category_threshold].index.tolist()
            if rare_vals:
                self.rare_categories_[col] = rare_vals
        
        # Fit numeric imputer
        if self.numeric_cols_:
            if self.use_knn or self.numeric_strategy == 'knn':
                self.numeric_imputer_ = KNNImputer(
                    n_neighbors=self.knn_neighbors,
                    weights='uniform'
                )
                self.numeric_imputer_.fit(X[self.numeric_cols_])
                
                if self.add_indicator:
                    self.missing_mask_ = X[self.numeric_cols_].isna()
            else:
                self.numeric_imputer_ = SimpleImputer(
                    strategy=self.numeric_strategy,
                    add_indicator=self.add_indicator
                )
                self.numeric_imputer_.fit(X[self.numeric_cols_])
        
        # Fit categorical imputer
        if self.categorical_cols_:
            self.categorical_imputer_ = SimpleImputer(
                strategy=self.categorical_strategy,
                fill_value=self.categorical_fill_value,
                add_indicator=False
            )
            self.categorical_imputer_.fit(X[self.categorical_cols_])
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the data by imputing missing values."""
        X = X.copy()
        
        # Apply business-driven fills before model-driven imputation
        for col, val in self.numeric_fill_map.items():
            if col in X.columns:
                X[col] = X[col].fillna(val)
        for col, val in self.categorical_fill_map.items():
            if col in X.columns:
                X[col] = X[col].fillna(val)
        
        # Prefill all-NaN columns
        for col in self.numeric_all_nan_cols_:
            if col in X.columns:
                X[col] = X[col].fillna(0.0)
        for col in self.categorical_all_nan_cols_:
            if col in X.columns:
                X[col] = X[col].fillna("__ALLNAN__")
        
        # Collapse rare categories
        for col, rare_vals in self.rare_categories_.items():
            if col in X.columns and rare_vals:
                X[col] = X[col].where(~X[col].isin(rare_vals), other=self.rare_category_name)
        
        # Transform numeric columns
        if self.numeric_cols_ and self.numeric_imputer_:
            numeric_transformed = self.numeric_imputer_.transform(X[self.numeric_cols_])
            
            if self.use_knn or self.numeric_strategy == 'knn':
                if self.add_indicator:
                    missing_indicators = X[self.numeric_cols_].isna().astype(int)
                    numeric_names = (
                        self.numeric_cols_ +
                        [f"{col}_missing" for col in self.numeric_cols_]
                    )
                    numeric_transformed = np.hstack([numeric_transformed, missing_indicators.values])
                else:
                    numeric_names = self.numeric_cols_
            else:
                if self.add_indicator:
                    numeric_names = (
                        self.numeric_cols_ +
                        [f"{col}_missing" for col in self.numeric_cols_]
                    )
                else:
                    numeric_names = self.numeric_cols_
            
            X = X.drop(columns=self.numeric_cols_)
            numeric_df = pd.DataFrame(
                numeric_transformed,
                columns=numeric_names,
                index=X.index
            )
            X = pd.concat([X, numeric_df], axis=1)
        
        # Transform categorical columns
        if self.categorical_cols_ and self.categorical_imputer_:
            categorical_transformed = self.categorical_imputer_.transform(
                X[self.categorical_cols_]
            )
            categorical_df = pd.DataFrame(
                categorical_transformed,
                columns=self.categorical_cols_,
                index=X.index
            )
            X[self.categorical_cols_] = categorical_df
        
        return X
    
    def get_feature_names_out(self, input_features=None):
        """Get output feature names."""
        if input_features is None:
            input_features = self.feature_names_in_
        
        output_names = []
        
        if self.numeric_cols_:
            output_names.extend(self.numeric_cols_)
            if self.add_indicator:
                output_names.extend([f"{col}_missing" for col in self.numeric_cols_])
        
        if self.categorical_cols_:
            output_names.extend(self.categorical_cols_)
        
        return np.array(output_names)
    
    def set_params(self, **params):
        """Set parameters for compatibility with sklearn."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter: {key}")
        return self


def create_business_missing_indicator(X: pd.DataFrame, column: str) -> pd.Series:
    """Create a missing indicator for a column where missing has business meaning."""
    if column not in X.columns:
        return pd.Series(0, index=X.index)
    
    s = X[column].copy()
    
    if s.dtype == 'object':
        s = s.str.strip()
        missing = s.eq("") | s.isna()
    else:
        missing = s.isna()
    
    return missing.astype(int)
