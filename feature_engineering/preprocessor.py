"""
Main preprocessor that combines all feature engineering steps.
"""

import numpy as np
import pandas as pd
from typing import List
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .encoders import FrequencyEncoder, MultiClassTargetEncoder
from .wide_features import WideFeatureBuilder
from .statistical_features import StatisticalAggregator
from .transformers import Log1pTransformer, BusinessMissingIndicator


def build_preprocessor(
    num_cols: List[str],
    cat_cols: List[str],
    drop_cols: List[str] = None,
    freq_encoding_cols: List[str] = None,
    target_encoding_cols: List[str] = None,
    target_encoding_alpha: float = 10.0,
    business_missing_col: str = "ConferenceRoomQuality",
    use_wide_features: bool = True,
    use_statistical_aggregation: bool = True,
    log_transform_cols: List[str] = None,
    model_type: str = "tree"  # "tree" or "tabnet" - TabNet 需要简化特征工程
) -> ColumnTransformer:
    """
    Build a comprehensive preprocessing pipeline.
    
    Args:
        num_cols: Numeric column names
        cat_cols: Categorical column names
        drop_cols: Columns to drop
        freq_encoding_cols: Columns for frequency encoding
        target_encoding_cols: Columns for target encoding
        target_encoding_alpha: Smoothing parameter for target encoding
        business_missing_col: Column with business-meaningful missing values
        use_wide_features: Whether to include wide feature builder
        use_statistical_aggregation: Whether to include statistical aggregation
        log_transform_cols: Columns to log-transform
        model_type: Model type ("tree" or "tabnet") - TabNet 需要简化特征工程
    
    Returns:
        ColumnTransformer pipeline
    """
    # Set defaults
    if drop_cols is None:
        drop_cols = [
            "AlleyAccess",
            "ExteriorFinishType",
            "RecreationQuality",
            "MiscellaneousFeature"
        ]
    
    if freq_encoding_cols is None:
        freq_encoding_cols = ['RoofType', 'ExteriorCovering1', 'FoundationType']
    
    if target_encoding_cols is None:
        target_encoding_cols = [
            'ZoningClassification',
            'BuildingType',
            'BuildingStyle',
            'HeatingType'
        ]
    
    if log_transform_cols is None:
        log_transform_cols = ['PlotSize']
    
    # Clean up column lists
    num_cols = [c for c in num_cols if c not in drop_cols]
    cat_cols = [c for c in cat_cols if c not in drop_cols]
    
    # Ensure business missing column is in categorical
    if business_missing_col not in num_cols and business_missing_col not in cat_cols:
        cat_cols.append(business_missing_col)
    
    # Filter encoding columns
    freq_encoding_cols = [c for c in freq_encoding_cols if c in (num_cols + cat_cols)]
    target_encoding_cols = [c for c in target_encoding_cols if c in (num_cols + cat_cols)]
    
    # Categorical columns for one-hot encoding (excluding those already encoded)
    cat_cols_onehot = [
        c for c in cat_cols
        if c not in set(freq_encoding_cols + target_encoding_cols)
    ]
    
    # Build numeric pipeline
    num_pipe_steps = [
        ("imputer", SimpleImputer(strategy="median", add_indicator=True))
    ]
    
    # Add log transform if needed
    if any(col in num_cols for col in log_transform_cols):
        num_pipe_steps.append(
            ("log_transform", Log1pTransformer(
                col_names=num_cols,
                target_col=log_transform_cols[0] if log_transform_cols else "PlotSize"
            ))
        )
    
    # Add StandardScaler for TabNet (深度学习模型需要标准化)
    # 注意：对于树模型（XGBoost等），标准化不是必需的，但不会有害
    num_pipe_steps.append(("scaler", StandardScaler()))
    
    num_pipe = Pipeline(steps=num_pipe_steps)
    
    # Build categorical pipeline
    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    
    # Transformers list
    transformers = [
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols_onehot),
    ]
    
    # Business missing indicator
    if business_missing_col:
        crq_missing_tf = BusinessMissingIndicator()
        transformers.append(
            ("crq_missing", crq_missing_tf, [business_missing_col])
        )
    
    # Wide features
    # TabNet 不需要太多手工特征，它能自动学习特征交互
    # 对于 TabNet，建议关闭 wide features 和 statistical aggregation
    if use_wide_features and model_type != "tabnet":
        wide_needed = [
            "YearListed", "ConstructionYear", "RenovationYear",
            "GroundFloorArea", "UpperFloorArea", "BasementArea",
            "FinishedBasementArea1", "FinishedBasementArea2", "UnfinishedBasementArea",
            "OfficeSpace", "PlotSize", "TotalRooms", "ParkingSpots",
            "BuildingGrade", "BuildingCondition", "ExteriorQuality", "ExteriorCondition",
            "BasementQuality", "BasementCondition", "BasementExposure",
            "Proximity1", "Proximity2", "MonthListed"
        ]
        wide_input_cols = [c for c in wide_needed if c in (num_cols + cat_cols)]
        
        if wide_input_cols:
            wide_pipe = Pipeline(steps=[
                ("builder", WideFeatureBuilder()),
                ("imputer", SimpleImputer(strategy="median"))
            ])
            transformers.append(("wide_feats", wide_pipe, wide_input_cols))
    
    # Frequency encoding
    if freq_encoding_cols:
        freq_pipe = Pipeline([("freq", FrequencyEncoder(freq_encoding_cols))])
        transformers.append(("freq_enc", freq_pipe, freq_encoding_cols))
    
    # Target encoding
    if target_encoding_cols:
        te_pipe = Pipeline([
            ("te", MultiClassTargetEncoder(target_encoding_cols, alpha=target_encoding_alpha))
        ])
        transformers.append(("target_enc", te_pipe, target_encoding_cols))
    
    # Statistical aggregation
    # TabNet 不需要统计聚合特征
    if use_statistical_aggregation and model_type != "tabnet":
        agg_input_cols = list(set([
            'ZoningClassification', 'BuildingType',
            'GroundFloorArea', 'UpperFloorArea',
            'YearListed', 'ConstructionYear',
            'BuildingGrade', 'BuildingCondition',
            'PlotSize'
        ]) & set(num_cols + cat_cols))
        
        if agg_input_cols:
            agg_pipe = Pipeline(steps=[
                ("agg", StatisticalAggregator(
                    groupby_cols=('ZoningClassification', 'BuildingType'),
                    agg_cols=('TotalLivingArea', 'BuildingAge', 'OverallQuality')
                )),
                ("imputer", SimpleImputer(strategy="median"))
            ])
            transformers.append(("agg_feats", agg_pipe, agg_input_cols))
    
    # Build final preprocessor
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )
    
    return preprocessor


class FeaturePreprocessor:
    """
    High-level wrapper for the preprocessing pipeline.
    
    Provides a clean interface for fitting and transforming data.
    """
    
    def __init__(self, config=None):
        """
        Initialize feature preprocessor.
        
        Args:
            config: Configuration object (optional)
        """
        self.config = config
        self.preprocessor_ = None
        self.num_cols_ = None
        self.cat_cols_ = None
    
    def fit(self, X: pd.DataFrame, y=None, num_cols=None, cat_cols=None):
        """
        Fit the preprocessor.
        
        Args:
            X: Input DataFrame
            y: Target (required for target encoding)
            num_cols: Numeric column names (auto-inferred if None)
            cat_cols: Categorical column names (auto-inferred if None)
        
        Returns:
            self
        """
        from data_cleaning import infer_column_types
        
        # Infer column types if not provided
        if num_cols is None or cat_cols is None:
            target_col = self.config.columns.target if self.config else "OfficeCategory"
            num_cols, cat_cols = infer_column_types(
                pd.concat([X, pd.Series(y, name=target_col)], axis=1),
                target_col
            )
        
        self.num_cols_ = num_cols
        self.cat_cols_ = cat_cols
        
        # Build preprocessor
        kwargs = {}
        if self.config:
            kwargs.update({
                'drop_cols': self.config.columns.drop_columns,
                'freq_encoding_cols': self.config.features.freq_encoding_cols,
                'target_encoding_cols': self.config.features.target_encoding_cols,
                'target_encoding_alpha': self.config.features.target_encoding_alpha,
                'business_missing_col': self.config.columns.business_missing_col,
            })
        
        self.preprocessor_ = build_preprocessor(num_cols, cat_cols, **kwargs)
        
        # Fit
        self.preprocessor_.fit(X, y)
        
        return self
    
    def transform(self, X: pd.DataFrame):
        """
        Transform data.
        
        Args:
            X: Input DataFrame
        
        Returns:
            Transformed array
        """
        if self.preprocessor_ is None:
            raise ValueError("Preprocessor not fitted. Call fit() first.")
        
        return self.preprocessor_.transform(X)
    
    def fit_transform(self, X: pd.DataFrame, y=None, num_cols=None, cat_cols=None):
        """
        Fit and transform in one step.
        
        Args:
            X: Input DataFrame
            y: Target
            num_cols: Numeric columns
            cat_cols: Categorical columns
        
        Returns:
            Transformed array
        """
        return self.fit(X, y, num_cols, cat_cols).transform(X)

