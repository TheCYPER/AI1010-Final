"""
Utilities for inferring and managing column types.
"""

from typing import Tuple, List
import pandas as pd


def infer_column_types(
    df: pd.DataFrame,
    target: str,
    exclude_cols: List[str] = None
) -> Tuple[List[str], List[str]]:
    """
    Infer numeric and categorical columns from DataFrame.
    
    Args:
        df: Input DataFrame
        target: Target column name to exclude
        exclude_cols: Additional columns to exclude
    
    Returns:
        Tuple of (numeric_columns, categorical_columns)
    """
    if exclude_cols is None:
        exclude_cols = []
    
    # Get feature columns
    features = df.drop(columns=[target] + exclude_cols, errors='ignore')
    
    # Identify categorical columns
    cat_cols = [
        c for c in features.columns 
        if features[c].dtype == "object" or str(features[c].dtype) == "category"
    ]
    
    # Remaining are numeric
    num_cols = [c for c in features.columns if c not in cat_cols]
    
    return num_cols, cat_cols

