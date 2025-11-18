# Ensemble2 使用指南

## 概述

Ensemble2 是一个使用 Stacking 方法组合 100 个不同参数的树模型的集成学习实现。与传统的 Voting 方法不同，Stacking 使用一个元学习器（meta-learner）来学习如何最好地组合基模型的预测结果。

## 特点

1. **100 个多样化的树模型**：
   - 50 个 XGBoost 模型（不同超参数）
   - 30 个 RandomForest 模型（不同超参数）
   - 20 个 ExtraTrees 模型（不同超参数）

2. **Stacking 集成**：
   - 使用 5 折交叉验证生成基模型的预测
   - 使用 Logistic Regression 作为元学习器（可配置）

3. **独立的配置文件**：
   - 配置在 `configs/ensemble2_config.py` 中，不污染主配置文件

## 使用方法

### 方法 1：通过修改 config.py

在 `configs/config.py` 中修改 `model_type`：

```python
@dataclass
class ModelConfig:
    model_type: str = "ensemble2"  # 改为 ensemble2
```

然后运行：

```bash
python main.py --mode train
```

### 方法 2：直接使用代码

```python
from configs.ensemble2_config import Ensemble2Config
from modeling.ensemble2 import create_ensemble2

# 创建配置（可以自定义）
config = Ensemble2Config(
    n_models=100,  # 100 个模型
    model_type_distribution={
        'xgb': 50,
        'rf': 30,
        'extra_trees': 20
    }
)

# 创建 ensemble
ensemble = create_ensemble2(config=config, num_classes=5)

# 训练
ensemble.fit(X_train, y_train)

# 预测
predictions = ensemble.predict(X_test)
probabilities = ensemble.predict_proba(X_test)
```

## 配置选项

### 模型数量

```python
config = Ensemble2Config(n_models=100)  # 可以调整模型数量
```

### 模型类型分布

```python
config = Ensemble2Config(
    model_type_distribution={
        'xgb': 50,        # XGBoost 模型数量
        'rf': 30,         # RandomForest 模型数量
        'extra_trees': 20  # ExtraTrees 模型数量
    }
)
```

### 超参数搜索空间

可以在 `configs/ensemble2_config.py` 中修改：

- `xgb_search_space`: XGBoost 的超参数范围
- `rf_search_space`: RandomForest 的超参数范围
- `extra_trees_search_space`: ExtraTrees 的超参数范围

### Stacking 配置

```python
config = Ensemble2Config(
    stacking_config={
        'cv': 5,  # 交叉验证折数
        'final_estimator': 'logistic',  # 元学习器: 'logistic', 'rf', 'xgb', 'svm'
        'final_estimator_params': {
            'logistic': {
                'max_iter': 1000,
                'C': 1.0,
                'solver': 'lbfgs',
                'random_state': 42,
                'n_jobs': -1
            }
        }
    }
)
```

## 预期效果

### 优势

1. **更高的泛化能力**：Stacking 通过元学习器学习最优组合方式，通常比 Voting 效果更好
2. **模型多样性**：100 个不同参数的模型提供了丰富的多样性
3. **自动特征工程**：元学习器可以学习到基模型之间的复杂交互

### 注意事项

1. **训练时间**：100 个模型 + Stacking 的训练时间会很长（可能需要数小时）
2. **内存占用**：需要存储 100 个模型，内存占用较大
3. **过拟合风险**：如果基模型过拟合，Stacking 也可能过拟合

### 性能预期

- **预期提升**：相比单个模型，通常可以提升 1-3% 的准确率
- **相比 Voting Ensemble**：Stacking 通常比 Voting 好 0.5-1.5%
- **最佳场景**：当基模型有足够的多样性且数据量足够大时效果最好

## 工程实践

### 1. 渐进式训练

如果训练时间太长，可以先尝试较少的模型：

```python
config = Ensemble2Config(n_models=20)  # 先试 20 个模型
```

### 2. 并行化

配置已经设置了 `n_jobs=-1`，会自动使用所有 CPU 核心。

### 3. 早停策略

对于 XGBoost 模型，可以启用早停（但会降低多样性）：

```python
config = Ensemble2Config(use_early_stopping=True)
```

### 4. 模型重要性分析

训练后可以查看哪些基模型最重要：

```python
importance = ensemble.get_feature_importance()
# 返回字典，键是模型名，值是对应的重要性分数
```

## 与 Ensemble (Voting) 的对比

| 特性 | Ensemble (Voting) | Ensemble2 (Stacking) |
|------|-------------------|---------------------|
| 基模型数量 | 5-10 个 | 100 个 |
| 组合方式 | 平均概率/多数投票 | 元学习器学习 |
| 训练时间 | 中等 | 很长 |
| 内存占用 | 低 | 高 |
| 预期性能 | 好 | 更好 |
| 适用场景 | 快速实验 | 追求最佳性能 |

## 常见问题

### Q: 100 个模型会不会太多？

A: 可以调整 `n_models` 参数。通常 20-50 个模型已经能提供很好的效果。

### Q: 训练时间太长怎么办？

A: 
1. 减少模型数量
2. 减少每个模型的 `n_estimators`
3. 使用更少的交叉验证折数（`cv=3`）

### Q: 会不会过拟合？

A: Stacking 使用交叉验证生成预测，有助于减少过拟合。但如果基模型都过拟合，Stacking 也可能过拟合。

### Q: 可以混合不同类型的模型吗？

A: 当前实现主要使用树模型（XGBoost, RF, ExtraTrees）。可以修改代码添加其他类型的模型。

## 参考

- [Stacking Classifier - scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html)
- [Ensemble Methods in Machine Learning](https://en.wikipedia.org/wiki/Ensemble_learning)

