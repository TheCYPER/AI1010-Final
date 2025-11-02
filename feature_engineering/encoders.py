"""
Encoding transformers for categorical variables.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any
from sklearn.base import BaseEstimator, TransformerMixin


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """
    Frequency encoding: Map categories to their frequency in training set.
    
    Unsupervised encoding that doesn't require target information.
    Unknown categories at transform time get frequency 0.
    """
    
    def __init__(self, cols: List[str]):
        """
        Initialize frequency encoder.
        
        Args:
            cols: Column names to encode
        """
        self.cols = cols
        self.maps_: Dict[str, pd.Series] = {}
        self.feature_names_out_: List[str] = []
    
    def fit(self, X: pd.DataFrame, y=None):
        """
        Fit by computing frequency for each category.
        
        Args:
            X: Input DataFrame
            y: Target (unused)
        
        Returns:
            self
        """
        n = len(X)
        self.maps_ = {}
        
        for c in self.cols:
            if c in X.columns:
                vc = X[c].value_counts(dropna=False)
                self.maps_[c] = (vc / n)
        
        self.feature_names_out_ = [f"FE__{c}" for c in self.cols if c in self.maps_]
        return self
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform by replacing categories with frequencies.
        
        Args:
            X: Input DataFrame
        
        Returns:
            Numpy array of frequencies
        """
        outs = []
        for c in self.cols:
            if c in self.maps_:
                m = self.maps_[c]
                vals = X[c].map(m).fillna(0.0).to_numpy().reshape(-1, 1)
                outs.append(vals)
        
        return np.hstack(outs) if outs else np.empty((len(X), 0))
    
    def get_feature_names_out(self, input_features=None):
        """Get output feature names."""
        return np.array(self.feature_names_out_)


class MultiClassTargetEncoder(BaseEstimator, TransformerMixin):
    """
    Target encoding for multi-class classification with Laplace smoothing.
    
    For each categorical column, outputs K features (one per class) representing
    P(y=k | category=x) with smoothing to handle rare categories.
    """
    
    def __init__(self, cols: List[str], alpha: float = 10.0):
        """
        Initialize target encoder.
        
        Args:
            cols: Column names to encode
            alpha: Smoothing parameter (higher = more smoothing towards global prior)
        """
        self.cols = cols
        self.alpha = alpha
        
        # Fitted attributes
        self.classes_: np.ndarray = None
        self.prior_: np.ndarray = None
        self.maps_: Dict[str, Dict[Any, np.ndarray]] = {}
        self.feature_names_out_: List[str] = []
    
    def fit(self, X: pd.DataFrame, y):
        """
        Fit by computing smoothed conditional probabilities.
        
        Args:
            X: Input DataFrame
            y: Target labels
        
        Returns:
            self
        """
        y = pd.Series(y).astype(int).values
        self.classes_ = np.sort(np.unique(y))
        K = len(self.classes_)
        
        # Compute global prior
        counts = pd.Series(y).value_counts().reindex(self.classes_, fill_value=0).to_numpy()
        self.prior_ = (counts / counts.sum()).astype(float)
        
        # Compute conditional probabilities for each column
        self.maps_ = {}
        for c in self.cols:
            if c not in X.columns:
                continue
            
            dfc = pd.DataFrame({
                c: X[c].astype("category"),
                "__y__": y
            })
            
            # Total count per category
            total = dfc.groupby(c, observed=True)["__y__"].count()
            
            # Count per class per category
            by_cls = {
                k: dfc[dfc["__y__"] == k].groupby(c, observed=True)["__y__"].count()
                for k in self.classes_
            }
            
            # Compute smoothed probabilities
            mapping: Dict[Any, np.ndarray] = {}
            for cat, tot in total.items():
                post = np.zeros(K, dtype=float)
                for i, k in enumerate(self.classes_):
                    cnt_k = float(by_cls[k].get(cat, 0.0))
                    # Laplace smoothing
                    post[i] = (cnt_k + self.alpha * self.prior_[i]) / (tot + self.alpha)
                mapping[cat] = post
            
            self.maps_[c] = mapping
        
        # Feature names
        self.feature_names_out_ = []
        for c in self.cols:
            if c in self.maps_:
                self.feature_names_out_.extend([
                    f"TE__{c}__p{int(k)}" for k in self.classes_
                ])
        
        return self
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform by replacing categories with probability vectors.
        
        Args:
            X: Input DataFrame
        
        Returns:
            Numpy array of probabilities
        """
        n = len(X)
        outs = []
        
        for c in self.cols:
            if c not in self.maps_:
                continue
            
            m = self.maps_[c]
            K = len(self.classes_)
            
            # Default to prior for unknown categories
            rows = []
            col_vals = X[c].astype("category") if c in X.columns else pd.Series([None] * n)
            
            for val in col_vals:
                rows.append(m.get(val, self.prior_))
            
            M = np.vstack(rows).astype(float)
            outs.append(M)
        
        return np.hstack(outs) if outs else np.empty((n, 0))
    
    def get_feature_names_out(self, input_features=None):
        """Get output feature names."""
        return np.array(self.feature_names_out_)

