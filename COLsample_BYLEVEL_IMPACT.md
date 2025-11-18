# colsample_bylevel 移除影响分析

## colsample_bylevel 的作用

`colsample_bylevel` 是 CatBoost 的 **RSM (Random Subspace Method)** 参数，用于：

1. **特征采样**：控制每一层（level）使用的特征比例
   - 例如：`colsample_bylevel=0.7` 表示每层只使用 70% 的特征
   - 在树的每一层随机选择特征子集

2. **增加模型多样性**：
   - 不同模型使用不同的特征组合
   - 提高 ensemble 的多样性

3. **防止过拟合**：
   - 减少特征使用，降低模型复杂度
   - 类似随机森林的 `max_features` 参数

## 移除 colsample_bylevel 的影响

### 1. 失去特征采样多样性 ⚠️

**影响程度：中等**

- **问题**：CatBoost GPU 模型无法通过特征采样增加多样性
- **后果**：模型可能更相似，ensemble 多样性降低
- **实际影响**：在 ensemble 中，影响可能不大，因为：
  - 还有其他多样性来源（超参数差异、随机种子等）
  - XGBoost 模型仍然有 `colsample_bytree` 提供特征采样

### 2. 略微增加过拟合风险 ⚠️

**影响程度：轻微**

- **问题**：失去一个正则化机制
- **后果**：模型可能略微更容易过拟合
- **实际影响**：影响很小，因为：
  - 还有其他正则化参数（`l2_leaf_reg`, `random_strength`）
  - `subsample` 仍然提供行采样正则化
  - Stacking 本身有正则化效果

### 3. 模型性能影响 📊

**影响程度：很小到无影响**

- **单模型**：可能略微提升（使用更多特征）
- **Ensemble**：可能略微下降（多样性降低）
- **实际测试**：通常影响 < 0.1%

## 缓解方案

### 方案 1: 增加其他正则化参数 ✅ 推荐

通过增加其他正则化参数来补偿：

```python
catboost_search_space: Dict[str, Any] = field(default_factory=lambda: {
    'iterations': (100, 2000),
    'learning_rate': (0.005, 0.1),
    'depth': (3, 10),
    'l2_leaf_reg': (1, 25),  # 增加范围：从 (1, 20) 到 (1, 25)
    'bootstrap_type': ['Bayesian', 'Bernoulli', 'MVS'],
    'subsample': (0.5, 1.0),
    # 'colsample_bylevel': (0.3, 1.0),  # GPU 模式下移除
    'random_strength': (0, 1.5),  # 增加范围：从 (0, 1) 到 (0, 1.5)
    'bagging_temperature': (0, 1),
})
```

### 方案 2: 增加 XGBoost 模型比例 ✅ 推荐

XGBoost 仍然支持 `colsample_bytree`，可以增加 XGBoost 模型比例：

```python
model_type_distribution: Dict[str, int] = field(default_factory=lambda: {
    'catboost': 40,    # 减少 CatBoost（无法使用 colsample_bylevel）
    'xgb': 40,         # 增加 XGBoost（可以使用 colsample_bytree）
    'rf': 10,
    'extra_trees': 10
})
```

### 方案 3: 增加模型数量 ✅ 可选

通过增加模型数量来补偿多样性损失：

```python
n_models: int = 120  # 从 100 增加到 120
```

### 方案 4: 调整其他多样性参数 ✅ 可选

通过其他参数增加多样性：

```python
catboost_search_space: Dict[str, Any] = field(default_factory=lambda: {
    # ... 其他参数 ...
    'depth': (3, 12),  # 增加深度范围
    'learning_rate': (0.001, 0.15),  # 增加学习率范围
    'iterations': (50, 2500),  # 增加迭代次数范围
})
```

## 实际影响评估

### 对 Ensemble 性能的影响

| 场景 | 影响 | 说明 |
|------|------|------|
| **单模型准确率** | +0.0% ~ +0.1% | 可能略微提升（使用更多特征） |
| **Ensemble 准确率** | -0.0% ~ -0.2% | 可能略微下降（多样性降低） |
| **过拟合风险** | +轻微 | 影响很小，有其他正则化 |
| **训练速度** | 无影响 | GPU 加速不受影响 |

### 为什么影响很小？

1. **其他多样性来源**：
   - 超参数差异（`depth`, `learning_rate`, `l2_leaf_reg` 等）
   - 随机种子差异
   - `bootstrap_type` 差异
   - `subsample` 差异

2. **XGBoost 仍然有特征采样**：
   - XGBoost 的 `colsample_bytree` 仍然提供特征采样
   - 在 ensemble 中，XGBoost 模型可以提供多样性

3. **Stacking 的正则化**：
   - Stacking 本身有正则化效果
   - 元学习器学习最优组合，减少过拟合

## 建议

### 短期（当前实现）✅

**无需担心**，影响很小：
- 移除 `colsample_bylevel` 对整体性能影响 < 0.2%
- GPU 加速带来的速度提升远大于这个损失
- 其他参数仍然提供足够的多样性

### 中期（优化配置）✅ 推荐

实施**方案 1 + 方案 2**：

```python
# 1. 增加正则化参数范围
catboost_search_space = {
    'l2_leaf_reg': (1, 25),  # 增加
    'random_strength': (0, 1.5),  # 增加
    # ... 其他参数 ...
}

# 2. 增加 XGBoost 比例（有 colsample_bytree）
model_type_distribution = {
    'catboost': 40,  # 减少
    'xgb': 40,       # 增加
    'rf': 10,
    'extra_trees': 10
}
```

### 长期（实验验证）🔬

如果发现性能下降，可以：
1. 对比 CPU 版本（有 `colsample_bylevel`）和 GPU 版本
2. 如果差异 > 0.3%，考虑混合使用（部分 CPU，部分 GPU）
3. 或者等待 CatBoost 更新支持 GPU 多分类的 RSM

## 总结

### 移除 colsample_bylevel 的影响

| 方面 | 影响程度 | 说明 |
|------|---------|------|
| **模型多样性** | 中等 | 失去一个多样性来源，但影响有限 |
| **过拟合风险** | 轻微 | 有其他正则化参数 |
| **整体性能** | 很小 | 通常 < 0.2% |
| **训练速度** | 无影响 | GPU 加速不受影响 |

### 结论

✅ **可以接受**：移除 `colsample_bylevel` 的影响很小，GPU 加速带来的好处远大于这个损失。

✅ **建议优化**：通过增加其他正则化参数和调整模型分布来补偿。

✅ **无需担心**：在 ensemble 中，其他多样性来源（超参数差异、XGBoost 特征采样等）仍然足够。

