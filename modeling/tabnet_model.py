"""
TabNet model implementation.

TabNet is a deep learning model specifically designed for tabular data.
Reference: https://arxiv.org/abs/1908.07442
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, Optional
from pytorch_tabnet.tab_model import TabNetClassifier

from .base_model import BaseModel


class TabNetModel(BaseModel):
    """
    TabNet classifier wrapper.
    
    Provides a clean interface for training TabNet with our pipeline.
    
    TabNet 的优势：
    - 深度学习模型，专为表格数据设计
    - 可解释性强（内置特征重要性）
    - 基于注意力机制，自动特征选择
    - 不需要太多特征工程
    - 对类别和数值特征都处理得很好
    
    注意：
    - TabNet 需要数值输入，所以仍需要预处理
    - 训练时间比树模型长
    - 需要更多的调参
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize TabNet model.
        
        Args:
            config: Model configuration
        """
        super().__init__(config)
        self.num_classes_ = None
        
        # Separate training params from model params
        self.training_params_ = {}
        if config:
            # Extract training params (those starting with _)
            self.training_params_ = {
                k[1:]: v for k, v in config.items() if k.startswith('_')
            }
    
    def build_model(self, num_classes: int, **kwargs) -> TabNetClassifier:
        """
        Build TabNet classifier.
        
        Args:
            num_classes: Number of classes
            **kwargs: Override config parameters
        
        Returns:
            TabNetClassifier instance
        """
        self.num_classes_ = num_classes
        
        # Only use model architecture params (not training params starting with _)
        model_params = {k: v for k, v in self.config.items() if not k.startswith('_')}
        
        # TabNet 的学习率在初始化时设置，不是在 fit 时
        # 将 _lr 转换为 optimizer_params
        if '_lr' in self.config:
            # TabNet 使用 optimizer_params 字典来设置学习率
            if 'optimizer_params' not in model_params:
                model_params['optimizer_params'] = {}
            model_params['optimizer_params']['lr'] = self.config['_lr']
        elif 'lr' in self.config:
            if 'optimizer_params' not in model_params:
                model_params['optimizer_params'] = {}
            model_params['optimizer_params']['lr'] = self.config['lr']
        
        # Merge with kwargs
        model_params = {**model_params, **kwargs}
        
        # Build model
        self.model_ = TabNetClassifier(**model_params)
        
        return self.model_
    
    def fit(
        self,
        X,
        y,
        eval_set=None,
        sample_weight=None,
        early_stopping_rounds=None,
        verbose=False,
        **kwargs
    ):
        """
        Fit TabNet model.
        
        Args:
            X: Training features (numpy array or pandas DataFrame)
            y: Training labels
            eval_set: Evaluation set for early stopping (list of tuples)
            sample_weight: Sample weights
            early_stopping_rounds: Early stopping rounds (patience)
            verbose: Verbosity
            **kwargs: Additional fit parameters
        
        Returns:
            self
        """
        # Infer number of classes if not set
        if self.model_ is None:
            num_classes = len(np.unique(y))
            self.build_model(num_classes)
        
        # Convert to numpy if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        
        # Prepare fit kwargs with training params
        fit_kwargs = {
            'X_train': X,
            'y_train': y,
        }
        
        # Use training params from config
        if 'max_epochs' in self.training_params_:
            fit_kwargs['max_epochs'] = self.training_params_['max_epochs']
        else:
            fit_kwargs['max_epochs'] = 100
            
        if 'batch_size' in self.training_params_:
            fit_kwargs['batch_size'] = self.training_params_['batch_size']
        
        # TabNet 的学习率在初始化时设置（通过 optimizer_params），不在 fit 时设置
        # 所以这里不需要传递 learning_rate
        
        # TabNet uses eval_set parameter
        if eval_set is not None:
            # eval_set is list of tuples [(X_val, y_val)]
            X_val, y_val = eval_set[0]
            if isinstance(X_val, pd.DataFrame):
                X_val = X_val.values
            if isinstance(y_val, pd.Series):
                y_val = y_val.values
            fit_kwargs['eval_set'] = [(X_val, y_val)]
            fit_kwargs['eval_metric'] = ['accuracy']
        
        # TabNet uses weights parameter
        if sample_weight is not None:
            fit_kwargs['weights'] = 1  # TabNet handles class weights internally
        
        # TabNet uses patience for early stopping
        if early_stopping_rounds is not None:
            fit_kwargs['patience'] = early_stopping_rounds
        elif 'patience' in self.training_params_:
            fit_kwargs['patience'] = self.training_params_['patience']
        
        # Merge additional kwargs
        fit_kwargs.update(kwargs)
        
        # Fit
        self.model_.fit(**fit_kwargs)
        self.is_fitted_ = True
        
        return self
    
    def predict(self, X):
        """
        Make predictions.
        
        Args:
            X: Features (numpy array or pandas DataFrame)
        
        Returns:
            Predicted class labels
        """
        if not self.is_fitted_:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Convert to numpy if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        return self.model_.predict(X)
    
    def predict_proba(self, X):
        """
        Predict class probabilities.
        
        Args:
            X: Features (numpy array or pandas DataFrame)
        
        Returns:
            Class probabilities
        """
        if not self.is_fitted_:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Convert to numpy if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        return self.model_.predict_proba(X)
    
    def get_feature_importance(self):
        """
        Get feature importance from TabNet.
        
        TabNet provides feature importance through its attention mechanism.
        
        Returns:
            Feature importance array (aggregated over all steps)
        """
        if not self.is_fitted_:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # TabNet stores feature importances as attribute
        if hasattr(self.model_, 'feature_importances_'):
            return self.model_.feature_importances_
        
        return None


def create_tabnet_model(config_dict: Dict[str, Any]) -> TabNetModel:
    """
    Factory function to create TabNet model from config.
    
    Args:
        config_dict: Configuration dictionary
    
    Returns:
        TabNetModel instance
    """
    return TabNetModel(config=config_dict)

