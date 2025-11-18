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
    - Business logic for specific columns (e.g., ConferenceRoomQuality)
    """
    
    def __init__(
        self,
        numeric_strategy: str = 'median',
        categorical_strategy: str = 'constant',
        categorical_fill_value: str = '__MISSING__',
        add_indicator: bool = True,
        business_missing_col: Optional[str] = None,
        use_knn: bool = False,
        knn_neighbors: int = 5
    ):
        """
        Initialize missing value handler.
        
        Args:
            numeric_strategy: Strategy for numeric columns ('mean', 'median', 'constant', 'knn')
            categorical_strategy: Strategy for categorical columns ('most_frequent', 'constant')
            categorical_fill_value: Fill value when using 'constant' strategy for categoricals
            add_indicator: Whether to add missing indicator columns
            business_missing_col: Column where missing has business meaning
            use_knn: Whether to use KNN imputation for numeric columns (overrides numeric_strategy)
            knn_neighbors: Number of neighbors for KNN imputation
        """
        self.numeric_strategy = numeric_strategy
        self.categorical_strategy = categorical_strategy
        self.categorical_fill_value = categorical_fill_value
        self.add_indicator = add_indicator
        self.business_missing_col = business_missing_col
        self.use_knn = use_knn
        self.knn_neighbors = knn_neighbors
        
        # Fitted attributes
        self.numeric_imputer_ = None
        self.categorical_imputer_ = None
        self.numeric_cols_ = None
        self.categorical_cols_ = None
        self.feature_names_in_ = None
    
    def fit(self, X: pd.DataFrame, y=None):
        """
        Fit the missing value handlers.
        
        Args:
            X: Input DataFrame
            y: Target (unused, for compatibility)
        
        Returns:
            self
        """
        self.feature_names_in_ = X.columns.tolist()
        
        # Separate numeric and categorical columns
        self.numeric_cols_ = X.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols_ = X.select_dtypes(exclude=[np.number]).columns.tolist()
        
        # Fit numeric imputer
        if self.numeric_cols_:
            if self.use_knn or self.numeric_strategy == 'knn':
                # Use KNN imputation for better accuracy
                # Note: KNNImputer doesn't support add_indicator directly
                # We'll add indicators manually if needed
                self.numeric_imputer_ = KNNImputer(
                    n_neighbors=self.knn_neighbors,
                    weights='uniform'
                )
                self.numeric_imputer_.fit(X[self.numeric_cols_])
                
                # If add_indicator is True, we need to track which columns had missing values
                if self.add_indicator:
                    self.missing_mask_ = X[self.numeric_cols_].isna()
            else:
                # Use SimpleImputer (median, mean, etc.)
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
                add_indicator=False  # Handled separately for categoricals
            )
            self.categorical_imputer_.fit(X[self.categorical_cols_])
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the data by imputing missing values.
        
        Args:
            X: Input DataFrame
        
        Returns:
            Transformed DataFrame
        """
        X = X.copy()
        
        # Transform numeric columns
        if self.numeric_cols_ and self.numeric_imputer_:
            numeric_transformed = self.numeric_imputer_.transform(X[self.numeric_cols_])
            
            # Handle missing indicators for KNN imputation
            if self.use_knn or self.numeric_strategy == 'knn':
                if self.add_indicator:
                    # Add missing indicators manually for KNN
                    missing_indicators = X[self.numeric_cols_].isna().astype(int)
                    numeric_names = (
                        self.numeric_cols_ +
                        [f"{col}_missing" for col in self.numeric_cols_]
                    )
                    # Combine imputed values and indicators
                    numeric_transformed = np.hstack([numeric_transformed, missing_indicators.values])
                else:
                    numeric_names = self.numeric_cols_
            else:
                # SimpleImputer handles indicators automatically
                if self.add_indicator:
                    numeric_names = (
                        self.numeric_cols_ +
                        [f"{col}_missing" for col in self.numeric_cols_]
                    )
                else:
                    numeric_names = self.numeric_cols_
            
            # Replace numeric columns
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
        
        # Add numeric columns
        if self.numeric_cols_:
            output_names.extend(self.numeric_cols_)
            if self.add_indicator:
                output_names.extend([f"{col}_missing" for col in self.numeric_cols_])
        
        # Add categorical columns
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
    """
    Create a missing indicator for a column where missing has business meaning.
    
    Args:
        X: Input DataFrame
        column: Column name
    
    Returns:
        Binary series indicating missingness
    """
    if column not in X.columns:
        return pd.Series(0, index=X.index)
    
    s = X[column].copy()
    
    # Handle string columns
    if s.dtype == 'object':
        s = s.str.strip()
        missing = s.eq("") | s.isna()
    else:
        missing = s.isna()
    
    return missing.astype(int)

