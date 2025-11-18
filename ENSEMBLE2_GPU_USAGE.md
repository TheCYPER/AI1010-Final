# Ensemble2 GPU 使用指南

## 概述

`Ensemble2GPU` 是 `Ensemble2` 的 GPU 加速版本，专门为利用 GPU 训练而优化。它使用 GPU 加速 CatBoost 和 XGBoost 模型，可以显著加快训练速度。

## GPU 支持

### 支持的模型

- **CatBoost**: 使用 `task_type='GPU'` 启用 GPU 加速
  - **注意**: GPU 模式下，`colsample_bylevel` (RSM) 参数不支持多分类，会自动移除
- **XGBoost**: 使用 `tree_method='gpu_hist'` 启用 GPU 加速
- **RandomForest/ExtraTrees**: CPU 训练（sklearn 不支持 GPU）

### GPU 模式限制

CatBoost 在 GPU 模式下有一些限制：
- `colsample_bylevel` (RSM) 仅支持 pairwise 模式，多分类问题会自动移除此参数
- `bootstrap_type='MVS'` 不支持多分类 GPU 模式，会自动替换为 `'Bernoulli'`
- 其他参数（如 `subsample`, `bagging_temperature`）根据 `bootstrap_type` 自动处理

**支持的 bootstrap_type（GPU 多分类）**：
- `'Bayesian'`: 支持 `bagging_temperature`，不支持 `subsample`
- `'Bernoulli'`: 支持 `subsample`，不支持 `bagging_temperature`
- `'MVS'`: **不支持**，会自动替换为 `'Bernoulli'`

### GPU 要求

- NVIDIA GPU（支持 CUDA）
- 已安装 CUDA 和 cuDNN
- CatBoost 和 XGBoost 的 GPU 版本

## 使用方法

### 1. 在配置文件中设置

在 `configs/config.py` 中修改：

```python
model_type: str = "ensemble2_gpu"
```

### 2. 运行训练

```bash
python main.py --mode train
```

### 3. 自定义配置

```python
from configs.ensemble2_config import Ensemble2Config
from modeling.ensemble2_gpu import create_ensemble2_gpu

config = Ensemble2Config(
    n_models=100,  # 100 个模型
    model_type_distribution={
        'catboost': 50,    # 50 个 CatBoost 模型（GPU）
        'xgb': 30,         # 30 个 XGBoost 模型（GPU）
        'rf': 10,          # 10 个 RandomForest 模型（CPU）
        'extra_trees': 10  # 10 个 ExtraTrees 模型（CPU）
    },
    stacking_config={
        'cv': 3,  # 3 折交叉验证
        'final_estimator': 'logistic'
    },
    n_jobs=2  # GPU 模型共享 GPU，减少并行度避免冲突
)

ensemble = create_ensemble2_gpu(config=config, num_classes=5)
```

## GPU 优化特性

### 1. 自动 GPU 配置

- CatBoost 自动使用 GPU（`task_type='GPU'`）
- XGBoost 自动使用 GPU（`tree_method='gpu_hist'`）
- 自动检测并使用第一个 GPU（`devices='0'`）

### 2. 并行策略优化

- 减少并行度（`n_jobs=2`）以避免 GPU 内存冲突
- GPU 模型共享同一个 GPU，串行训练更稳定

### 3. 内存管理

- 每个模型使用 `thread_count=1`（GPU 处理计算）
- 禁用中间文件写入（`allow_writing_files=False`）

## 性能对比

### 训练时间（估算）

| 配置 | 模型数 | CV 折数 | 总任务数 | 预计时间（CPU） | 预计时间（GPU） |
|------|--------|---------|----------|----------------|----------------|
| 小规模 | 20 | 3 | 60 | ~30 分钟 | ~10 分钟 |
| 中等规模 | 50 | 3 | 150 | ~75 分钟 | ~25 分钟 |
| 大规模 | 100 | 3 | 300 | ~150 分钟 | ~50 分钟 |

**注意**: 实际时间取决于数据大小、模型复杂度、GPU 性能等因素。

### GPU 加速比

- **CatBoost**: 通常 2-5x 加速（取决于数据大小）
- **XGBoost**: 通常 3-10x 加速（取决于数据大小和树深度）

## 注意事项

### 1. GPU 内存

- 多个 GPU 模型共享同一个 GPU
- 如果 GPU 内存不足，可能需要：
  - 减少并行度（`n_jobs=1`）
  - 减少模型数量
  - 减少每个模型的复杂度

### 2. 并行训练

- GPU 模型不适合高度并行训练
- 建议 `n_jobs=2` 或更少
- 太多并行任务可能导致 GPU 内存溢出

### 3. 混合模型

- CatBoost 和 XGBoost 使用 GPU
- RandomForest 和 ExtraTrees 使用 CPU
- 训练时间取决于 GPU 和 CPU 模型的混合比例

### 4. 检查 GPU 可用性

```bash
# 检查 GPU
nvidia-smi

# 检查 CatBoost GPU 支持
python -c "from catboost import CatBoostClassifier; print('CatBoost GPU available')"

# 检查 XGBoost GPU 支持
python -c "import xgboost as xgb; print('XGBoost GPU available')"
```

## 故障排除

### 问题 1: GPU 内存不足

**错误**: `CUDA out of memory`

**解决方案**:
1. 减少并行度：`n_jobs=1`
2. 减少模型数量：`n_models=50`
3. 减少 CV 折数：`cv=2`

### 问题 2: GPU 未被使用

**检查**:
1. 确认 GPU 可用：`nvidia-smi`
2. 确认 CatBoost/XGBoost GPU 版本已安装
3. 查看训练日志，确认显示 "GPU-accelerated models"

### 问题 3: 训练速度没有提升

**可能原因**:
1. 数据量太小（GPU 优势不明显）
2. 模型太简单（GPU 优势不明显）
3. GPU 和 CPU 模型混合（CPU 模型拖慢整体速度）

**解决方案**:
1. 只使用 GPU 模型（移除 RF/ExtraTrees）
2. 增加模型复杂度
3. 使用更大的数据集

## 最佳实践

1. **先测试小规模**：
   ```python
   config = Ensemble2Config(n_models=20, stacking_config={'cv': 2})
   ```

2. **监控 GPU 使用**：
   ```bash
   watch -n 1 nvidia-smi
   ```

3. **逐步增加规模**：
   - 先测试 20 个模型
   - 如果成功，增加到 50 个
   - 最后增加到 100 个

4. **优化模型分布**：
   - 增加 GPU 模型比例（CatBoost + XGBoost）
   - 减少 CPU 模型比例（RF + ExtraTrees）

## 示例配置

### 纯 GPU 配置（最快）

```python
config = Ensemble2Config(
    n_models=80,
    model_type_distribution={
        'catboost': 50,  # 全部使用 GPU
        'xgb': 30,       # 全部使用 GPU
        'rf': 0,         # 不使用 CPU 模型
        'extra_trees': 0
    },
    n_jobs=2  # 低并行度避免 GPU 冲突
)
```

### 混合配置（平衡）

```python
config = Ensemble2Config(
    n_models=100,
    model_type_distribution={
        'catboost': 50,    # GPU
        'xgb': 30,         # GPU
        'rf': 10,          # CPU
        'extra_trees': 10  # CPU
    },
    n_jobs=2
)
```

## 总结

`Ensemble2GPU` 是充分利用 GPU 资源的理想选择，特别适合：
- 有 NVIDIA GPU 的用户
- 需要快速训练大量模型的场景
- 数据量较大的项目

通过合理配置，可以获得 2-5x 的训练速度提升。

