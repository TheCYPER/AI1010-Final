"""
Main preprocessor that combines all feature engineering steps.
"""

import numpy as np
import pandas as pd
from typing import List
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler, FunctionTransformer

from .encoders import FrequencyEncoder, MultiClassTargetEncoder
from .wide_features import WideFeatureBuilder
from .statistical_features import StatisticalAggregator
from .transformers import Log1pTransformer, BusinessMissingIndicator, KNNImputerWithIndicators


def _to_str_array(X):
    """Safely cast inputs to string for categorical encoding (pickle friendly)."""
    if isinstance(X, pd.DataFrame):
        return X.astype(str)
    return np.asarray(X).astype(str)


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
    model_type: str = "tree",  # "tree" or "tabnet" - TabNet 需要简化特征工程
    use_knn_imputation: bool = True,  # Use KNN imputation for numeric columns
    knn_neighbors: int = 5  # Number of neighbors for KNN imputation
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
    
    # Validate that we have at least some features
    if not num_cols and not cat_cols:
        raise ValueError(
            f"All features were dropped! Please check drop_columns: {drop_cols}. "
            f"Original num_cols: {len([c for c in num_cols if c not in drop_cols])}, "
            f"cat_cols: {len([c for c in cat_cols if c not in drop_cols])}"
        )
    
    # Ensure business missing column is in categorical
    if business_missing_col and business_missing_col not in num_cols and business_missing_col not in cat_cols:
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
    # Use KNN imputation for better accuracy (considers feature relationships)
    if use_knn_imputation:
        # Create a combined transformer that stores missing pattern, imputes, and adds indicators
        num_pipe_steps = [
            ("knn_imputer_with_indicators", KNNImputerWithIndicators(
                n_neighbors=knn_neighbors,
                col_names=num_cols
            ))
        ]
    else:
        # Use SimpleImputer (faster but less accurate)
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
    
    # Add scaler based on model type
    # - Linear models (SVM, Ridge, Logistic, MLP, KNN, Naive Bayes): RobustScaler (robust to outliers)
    # - Tree models (XGBoost, CatBoost, LightGBM, RF): StandardScaler (not required but harmless)
    # - Deep learning (TabNet): StandardScaler (required for neural networks)
    # - Ensemble: RobustScaler (contains linear models)
    linear_models = ['svm', 'ridge', 'logistic', 'mlp', 'knn', 'naive_bayes']
    if model_type.lower() in linear_models or model_type.lower() == 'ensemble':
        # Use RobustScaler for linear models (more robust to outliers)
        num_pipe_steps.append(("scaler", RobustScaler()))
    else:
        # Use StandardScaler for tree models and deep learning
        num_pipe_steps.append(("scaler", StandardScaler()))
    
    num_pipe = Pipeline(steps=num_pipe_steps)
    
    # Build categorical pipeline (only if there are categorical columns)
    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
        ("to_str", FunctionTransformer(_to_str_array, validate=False)),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    
    # Transformers list
    transformers = []
    
    # Only add numeric pipeline if there are numeric columns to process
    if num_cols:
        transformers.append(("num", num_pipe, num_cols))
    
    # Only add categorical pipeline if there are categorical columns to process
    if cat_cols_onehot:
        transformers.append(("cat", cat_pipe, cat_cols_onehot))
    
    # Business missing indicator (only if column exists in data)
    if business_missing_col and business_missing_col in (num_cols + cat_cols):
        crq_missing_tf = BusinessMissingIndicator()
        transformers.append(
            ("crq_missing", crq_missing_tf, [business_missing_col])
        )
    
    # Wide features
    # TabNet 不需要太多手工特征，它能自动学习特征交互
    # 对于 TabNet，建议关闭 wide features 和 statistical aggregation
    # Note: WideFeatureBuilder should always generate features, but we check input cols exist
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
        
        # Only add if we have at least some input columns
        # WideFeatureBuilder should generate features even if some columns are missing
        if wide_input_cols and len(wide_input_cols) > 0:
            wide_pipe = Pipeline(steps=[
                ("builder", WideFeatureBuilder()),
                ("imputer", SimpleImputer(strategy="median"))
            ])
            transformers.append(("wide_feats", wide_pipe, wide_input_cols))
    
    # Frequency encoding (only if columns exist after filtering)
    if freq_encoding_cols and len(freq_encoding_cols) > 0:
        freq_pipe = Pipeline([("freq", FrequencyEncoder(freq_encoding_cols))])
        transformers.append(("freq_enc", freq_pipe, freq_encoding_cols))
    
    # Target encoding (only if columns exist after filtering)
    if target_encoding_cols and len(target_encoding_cols) > 0:
        te_pipe = Pipeline([
            ("te", MultiClassTargetEncoder(target_encoding_cols, alpha=target_encoding_alpha))
        ])
        transformers.append(("target_enc", te_pipe, target_encoding_cols))
    
    # Statistical aggregation
    # TabNet 不需要统计聚合特征
    # Note: StatisticalAggregator requires groupby_cols to generate features
    # If groupby_cols (ZoningClassification, BuildingType) are dropped, it will return empty array
    # So we only add this pipeline if at least one groupby_col exists
    if use_statistical_aggregation and model_type != "tabnet":
        agg_input_cols = list(set([
            'ZoningClassification', 'BuildingType',
            'GroundFloorArea', 'UpperFloorArea',
            'YearListed', 'ConstructionYear',
            'BuildingGrade', 'BuildingCondition',
            'PlotSize'
        ]) & set(num_cols + cat_cols))
        
        # Check if groupby columns exist (required for StatisticalAggregator to work)
        groupby_cols_required = ['ZoningClassification', 'BuildingType']
        groupby_cols_available = [col for col in groupby_cols_required if col in (num_cols + cat_cols)]
        
        # Only add if we have at least one groupby column AND some input columns
        # StatisticalAggregator needs groupby_cols to generate features
        if groupby_cols_available and agg_input_cols and len(agg_input_cols) > 0:
            agg_pipe = Pipeline(steps=[
                ("agg", StatisticalAggregator(
                    groupby_cols=tuple(groupby_cols_available),  # Use available groupby cols
                    agg_cols=('TotalLivingArea', 'BuildingAge', 'OverallQuality')
                )),
                ("imputer", SimpleImputer(strategy="median"))
            ])
            transformers.append(("agg_feats", agg_pipe, agg_input_cols))
    
    # Validate that we have at least one transformer
    if not transformers:
        raise ValueError(
            "No transformers available! All features may have been dropped. "
            f"num_cols after filtering: {len(num_cols)}, "
            f"cat_cols after filtering: {len(cat_cols)}, "
            f"drop_cols: {drop_cols}"
        )
    
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
