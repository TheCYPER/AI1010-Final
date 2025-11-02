"""
Exploratory data analysis utilities.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any
from pathlib import Path

from utils.logger import get_logger

logger = get_logger(__name__)


class ExploratoryAnalysis:
    """
    Exploratory Data Analysis toolkit.
    
    Provides methods for understanding data distribution,
    missing values, correlations, etc.
    """
    
    def __init__(self, config=None):
        """
        Initialize EDA tool.
        
        Args:
            config: Configuration object (optional)
        """
        self.config = config
        self.data_ = None
        self.report_ = {}
    
    def load_data(self, path: Optional[str] = None) -> pd.DataFrame:
        """
        Load data for analysis.
        
        Args:
            path: Path to data file
        
        Returns:
            DataFrame
        """
        if path is None and self.config:
            path = self.config.paths.train_csv
        
        logger.info(f"Loading data from {path}")
        self.data_ = pd.read_csv(path)
        logger.info(f"Data shape: {self.data_.shape}")
        
        return self.data_
    
    def analyze_missing_values(self) -> Dict[str, Any]:
        """
        Analyze missing values in the dataset.
        
        Returns:
            Dictionary with missing value statistics
        """
        if self.data_ is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        logger.info("Analyzing missing values...")
        
        missing_count = self.data_.isnull().sum()
        missing_pct = (missing_count / len(self.data_) * 100)
        
        missing_df = pd.DataFrame({
            'column': missing_count.index,
            'missing_count': missing_count.values,
            'missing_pct': missing_pct.values
        })
        
        # Filter columns with missing values
        missing_df = missing_df[missing_df['missing_count'] > 0]
        missing_df = missing_df.sort_values('missing_pct', ascending=False)
        
        self.report_['missing_values'] = missing_df.to_dict('records')
        
        logger.info(f"Found {len(missing_df)} columns with missing values")
        logger.info(f"Top 5 columns by missing percentage:")
        if len(missing_df) > 0:
            for _, row in missing_df.head().iterrows():
                logger.info(f"  {row['column']}: {row['missing_pct']:.2f}%")
        
        return self.report_['missing_values']
    
    def analyze_target_distribution(self, target_col: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze target variable distribution.
        
        Args:
            target_col: Target column name
        
        Returns:
            Dictionary with target distribution
        """
        if self.data_ is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        if target_col is None and self.config:
            target_col = self.config.columns.target
        
        if target_col not in self.data_.columns:
            raise ValueError(f"Column {target_col} not found in data")
        
        logger.info(f"Analyzing target distribution for {target_col}...")
        
        value_counts = self.data_[target_col].value_counts().sort_index()
        distribution = {
            'counts': value_counts.to_dict(),
            'percentages': (value_counts / len(self.data_) * 100).to_dict()
        }
        
        self.report_['target_distribution'] = distribution
        
        logger.info(f"Target distribution:")
        for k, v in distribution['counts'].items():
            pct = distribution['percentages'][k]
            logger.info(f"  Class {k}: {v} ({pct:.2f}%)")
        
        return distribution
    
    def analyze_cardinality(self) -> Dict[str, int]:
        """
        Analyze cardinality of categorical columns.
        
        Returns:
            Dictionary mapping column names to unique value counts
        """
        if self.data_ is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        logger.info("Analyzing cardinality...")
        
        # Select categorical columns
        cat_cols = self.data_.select_dtypes(include=['object', 'category']).columns
        
        cardinality = {}
        for col in cat_cols:
            n_unique = self.data_[col].nunique(dropna=False)
            cardinality[col] = n_unique
        
        # Sort by cardinality
        cardinality = dict(sorted(cardinality.items(), key=lambda x: x[1], reverse=True))
        
        self.report_['cardinality'] = cardinality
        
        logger.info(f"Cardinality analysis (top 10):")
        for i, (col, n_unique) in enumerate(list(cardinality.items())[:10]):
            logger.info(f"  {col}: {n_unique} unique values")
        
        return cardinality
    
    def analyze_numeric_features(self) -> pd.DataFrame:
        """
        Analyze numeric feature distributions.
        
        Returns:
            DataFrame with statistics
        """
        if self.data_ is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        logger.info("Analyzing numeric features...")
        
        num_cols = self.data_.select_dtypes(include=[np.number]).columns
        
        stats = self.data_[num_cols].describe()
        
        self.report_['numeric_stats'] = stats.to_dict()
        
        logger.info(f"Numeric features: {len(num_cols)}")
        
        return stats
    
    def generate_report(self, save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate comprehensive EDA report.
        
        Args:
            save_path: Path to save report (JSON)
        
        Returns:
            Full report dictionary
        """
        logger.info("="*70)
        logger.info("GENERATING EDA REPORT")
        logger.info("="*70)
        
        # Run all analyses
        self.analyze_missing_values()
        self.analyze_target_distribution()
        self.analyze_cardinality()
        self.analyze_numeric_features()
        
        # Add basic info
        self.report_['basic_info'] = {
            'n_rows': int(len(self.data_)),
            'n_cols': int(len(self.data_.columns)),
            'n_numeric': int(len(self.data_.select_dtypes(include=[np.number]).columns)),
            'n_categorical': int(len(self.data_.select_dtypes(include=['object', 'category']).columns))
        }
        
        # Save if requested
        if save_path:
            import json
            with open(save_path, 'w') as f:
                json.dump(self.report_, f, indent=2)
            logger.info(f"Saved report to {save_path}")
        
        logger.info("="*70)
        logger.info("EDA REPORT COMPLETE")
        logger.info("="*70)
        
        return self.report_

