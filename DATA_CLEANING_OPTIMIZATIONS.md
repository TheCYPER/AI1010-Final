# 数据清理优化建议

## 当前实现分析

### 已有的功能
1. **缺失值处理**：
   - 数值特征：使用 `median` 填充 + missing indicator
   - 类别特征：使用 `constant` 填充（`__MISSING__`）
   - 业务缺失：`ConferenceRoomQuality` 有特殊处理

2. **特征工程**：
   - Log transform（如 `PlotSize`）
   - Wide features（交互特征）
   - Statistical aggregation
   - Frequency encoding 和 Target encoding

3. **异常值处理**：
   - 有 `OutlierHandler` 类，但可能没有在 pipeline 中使用

## 优化建议

### 1. 缺失值填充优化 ⭐⭐⭐

**当前问题**：
- 只使用 `median` 和 `constant`，可能不够精确

**优化方案**：

#### A. KNN Imputation（推荐）
```python
from sklearn.impute import KNNImputer

# 对于数值特征，KNN imputation 通常比 median 更准确
# 因为它考虑了特征之间的关系
numeric_imputer = KNNImputer(n_neighbors=5, add_indicator=True)
```

**优点**：
- 考虑特征之间的相关性
- 对于有多个缺失值的样本更准确
- 可以保留数据分布

**缺点**：
- 计算成本较高（但可以接受）

#### B. Iterative Imputation（备选）
```python
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor

# 使用随机森林进行迭代填充
iterative_imputer = IterativeImputer(
    estimator=RandomForestRegressor(n_estimators=10, random_state=42),
    max_iter=10,
    random_state=42,
    add_indicator=True
)
```

**优点**：
- 非常准确
- 可以处理复杂的缺失模式

**缺点**：
- 计算成本很高
- 可能过拟合

#### C. 按列选择策略
```python
# 不同列使用不同策略
# - 对于缺失率高的列：使用 constant 或 mode
# - 对于缺失率低的列：使用 KNN 或 median
# - 对于有业务含义的缺失：保留为单独类别
```

### 2. 异常值处理集成 ⭐⭐

**当前问题**：
- `OutlierHandler` 存在但没有在 pipeline 中使用
- 树模型对异常值不敏感，但线性模型（SVM, Ridge）敏感

**优化方案**：

#### A. 条件性异常值处理
```python
# 只在非树模型前使用异常值处理
if model_type in ['svm', 'ridge', 'logistic', 'mlp']:
    # 使用 RobustScaler 或 clip outliers
    outlier_handler = OutlierHandler(method='iqr', threshold=3.0, action='clip')
```

#### B. RobustScaler（推荐）
```python
from sklearn.preprocessing import RobustScaler

# RobustScaler 使用 median 和 IQR，对异常值不敏感
# 适合 SVM, Ridge 等线性模型
robust_scaler = RobustScaler()
```

**优点**：
- 自动处理异常值
- 不需要手动检测和裁剪

### 3. 特征缩放优化 ⭐⭐

**当前问题**：
- 使用 `StandardScaler`，对所有模型都标准化
- 树模型不需要标准化，但线性模型需要

**优化方案**：

```python
# 根据模型类型选择缩放方法
if model_type in ['tree', 'ensemble']:
    # 树模型：不需要缩放，或使用 MinMaxScaler（0-1范围）
    scaler = None  # 或 MinMaxScaler()
elif model_type in ['svm', 'ridge', 'logistic', 'mlp', 'knn']:
    # 线性模型：StandardScaler 或 RobustScaler
    scaler = RobustScaler()  # 对异常值更稳健
```

### 4. 缺失值模式分析 ⭐

**优化方案**：
```python
# 分析缺失值模式
# - 完全随机缺失（MCAR）
# - 随机缺失（MAR）
# - 非随机缺失（MNAR）

# 对于 MNAR，可能需要特殊处理
# 例如：ConstructionYear 缺失可能意味着老建筑
```

### 5. 特征选择 ⭐⭐

**优化方案**：
```python
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import VarianceThreshold

# A. 移除低方差特征
variance_threshold = VarianceThreshold(threshold=0.01)

# B. 基于统计的特征选择
selector = SelectKBest(f_classif, k=100)  # 选择 top 100 特征

# C. 基于模型的特征选择（在训练后）
# 使用 feature_importances_ 选择重要特征
```

**优点**：
- 减少过拟合
- 加快训练速度
- 提高模型可解释性

### 6. 类别特征编码优化 ⭐

**当前问题**：
- 使用 One-Hot Encoding，可能产生高维稀疏特征

**优化方案**：

#### A. 高基数类别特征
```python
# 对于高基数类别（如 BuildingType），考虑：
# 1. Target Encoding（已有）
# 2. Frequency Encoding（已有）
# 3. Embedding（深度学习模型）
```

#### B. 类别特征组合
```python
# 对于相关类别特征，可以创建组合特征
# 例如：ZoningClassification + BuildingType
```

### 7. 数据质量检查 ⭐

**优化方案**：
```python
# 添加数据质量检查
# - 检查重复行
# - 检查特征之间的相关性（避免多重共线性）
# - 检查类别不平衡
# - 检查数据漂移（train vs test）
```

## 实施优先级

### 高优先级（立即实施）
1. ✅ **KNN Imputation** - 对数值特征使用 KNN 填充
2. ✅ **RobustScaler** - 对线性模型使用 RobustScaler
3. ✅ **条件性异常值处理** - 根据模型类型选择是否处理异常值

### 中优先级（有时间时实施）
4. **特征选择** - 移除低方差特征，选择重要特征
5. **缺失值模式分析** - 分析缺失值类型，针对性处理
6. **类别特征优化** - 优化高基数类别特征的处理

### 低优先级（可选）
7. **数据质量检查** - 添加自动化数据质量检查
8. **Iterative Imputation** - 如果 KNN 不够，尝试迭代填充

## 实施建议

### 方案1：快速优化（推荐）
```python
# 在 preprocessor.py 中修改
# 1. 数值特征：SimpleImputer(strategy='median') -> KNNImputer(n_neighbors=5)
# 2. 线性模型：StandardScaler() -> RobustScaler()
# 3. 添加异常值处理（仅对线性模型）
```

### 方案2：完整优化
```python
# 创建新的 AdvancedPreprocessor
# - 根据模型类型选择不同的预处理策略
# - 添加特征选择
# - 添加数据质量检查
```

## 预期效果

- **KNN Imputation**: +0.5-1% 准确率提升
- **RobustScaler**: +0.3-0.5% 准确率提升（线性模型）
- **特征选择**: +0.2-0.5% 准确率提升，训练速度提升 20-30%
- **异常值处理**: +0.1-0.3% 准确率提升（线性模型）

**总计预期提升**: +1-2% 准确率

