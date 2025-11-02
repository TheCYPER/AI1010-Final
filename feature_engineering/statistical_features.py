"""
Statistical aggregation features based on grouping.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from sklearn.base import BaseEstimator, TransformerMixin


class StatisticalAggregator(BaseEstimator, TransformerMixin):
    """
    Generate statistical aggregation features.
    
    For each groupby column and aggregation column:
    - Compute group statistics (mean, std, median)
    - Generate z-score within group
    - Generate relative shift from group mean
    """
    
    def __init__(
        self,
        groupby_cols: Optional[Tuple[str, ...]] = None,
        agg_cols: Optional[Tuple[str, ...]] = None
    ):
        """
        Initialize statistical aggregator.
        
        Args:
            groupby_cols: Columns to group by
            agg_cols: Columns to aggregate
        """
        # Store as-is for sklearn clone compatibility
        self.groupby_cols = groupby_cols
        self.agg_cols = agg_cols
        
        # Fitted attributes
        self.agg_stats_: Dict[Tuple[str, str], pd.DataFrame] = {}
        self.feature_names_: List[str] = []
    
    @staticmethod
    def _ensure_cols(df: pd.DataFrame) -> pd.DataFrame:
        """Ensure required derived columns exist."""
        out = df.copy()
        
        def to_num(col):
            if col in out.columns:
                return pd.to_numeric(out[col], errors='coerce')
            return pd.Series([np.nan] * len(out), index=out.index)
        
        # Total living area
        if "TotalLivingArea" not in out.columns:
            out["TotalLivingArea"] = to_num("GroundFloorArea") + to_num("UpperFloorArea")
        
        # Building age
        if "BuildingAge" not in out.columns:
            out["BuildingAge"] = to_num("YearListed") - to_num("ConstructionYear")
            out["BuildingAge"] = out["BuildingAge"].mask(out["BuildingAge"] < 0, np.nan)
        
        # Overall quality
        if "OverallQuality" not in out.columns:
            out["OverallQuality"] = to_num("BuildingGrade") * to_num("BuildingCondition")
        
        return out
    
    def fit(self, X: pd.DataFrame, y=None):
        """
        Fit by computing group statistics.
        
        Args:
            X: Input DataFrame
            y: Target (unused)
        
        Returns:
            self
        """
        Xc = self._ensure_cols(X)
        
        # Use defaults if not provided
        groupby_cols = self.groupby_cols or ('ZoningClassification', 'BuildingType')
        agg_cols = self.agg_cols or ('TotalLivingArea', 'BuildingAge', 'OverallQuality')
        
        self.agg_stats_ = {}
        
        for gcol in groupby_cols:
            if gcol not in Xc.columns:
                continue
            
            for acol in agg_cols:
                if acol not in Xc.columns:
                    continue
                
                # Compute statistics
                stats = Xc.groupby(gcol)[acol].agg(['mean', 'std', 'median'])
                self.agg_stats_[(gcol, acol)] = stats
        
        return self
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform by creating statistical features.
        
        Args:
            X: Input DataFrame
        
        Returns:
            Numpy array of statistical features
        """
        Xc = self._ensure_cols(X)
        feats: Dict[str, np.ndarray] = {}
        
        for (gcol, acol), stats in self.agg_stats_.items():
            if gcol not in Xc.columns or acol not in Xc.columns:
                continue
            
            # Map group statistics
            gmean = Xc[gcol].map(stats['mean'])
            gstd = Xc[gcol].map(stats['std'])
            cur = Xc[acol]
            
            # Z-score within group
            feats[f"{acol}_ZScore__{gcol}"] = (
                (cur - gmean) / (gstd + 1e-6)
            ).to_numpy()
            
            # Relative shift from group mean
            feats[f"{acol}_RelShift__{gcol}"] = (
                (cur - gmean) / (np.abs(gmean) + 1e-6)
            ).to_numpy()
        
        if not feats:
            return np.empty((len(X), 0))
        
        out_df = pd.DataFrame(feats, index=X.index)
        self.feature_names_ = list(out_df.columns)
        
        return out_df.to_numpy(dtype=float)
    
    def get_feature_names_out(self, input_features=None):
        """Get output feature names."""
        return np.array(self.feature_names_ if self.feature_names_ else [])

