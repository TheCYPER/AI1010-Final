# Ensemble2 训练状态说明

## 终端消息含义

### `[Parallel(n_jobs=-1)]: Done 5 out of 5 | elapsed: X.Xmin finished`

这个消息表示：
- **一个基模型完成了 5 折交叉验证**
- `Done 5 out of 5` = 完成了 5 折中的 5 折（全部完成）
- `elapsed: X.Xmin` = 这个模型花了 X.X 分钟

### 训练过程

1. **Stacking 的工作原理**：
   - 对每个基模型，使用交叉验证生成预测
   - 100 个模型 × 5 折 CV = **500 个训练任务**
   - 每个任务都需要训练一个模型

2. **为什么这么慢**：
   - 每个模型都要训练 5 次（5 折交叉验证）
   - 100 个模型 × 5 次 = 500 次训练
   - 即使并行，也需要很长时间

3. **内存占用**：
   - 100 个模型同时存在内存中
   - 每个模型都需要存储
   - 可能导致内存不足（SIGKILL）

## 当前状态判断

### 如何判断训练是否完成？

**训练完成的标志**：
```
Ensemble2 training completed!
```

**如果看到**：
- 很多 `Done 5 out of 5` 消息 → 正在训练中
- 进程被 SIGKILL 终止 → 可能内存不足，训练未完成
- 没有 "training completed" 消息 → 训练未完成

### 检查进程状态

```bash
# 检查是否有 Python 进程在运行
ps aux | grep python

# 检查内存使用
free -h

# 检查 CPU 使用
top
```

## 优化建议

### 1. 减少模型数量（已优化）

**默认配置已改为**：
- `n_models: 20`（从 100 减少到 20）
- `cv: 3`（从 5 减少到 3）
- 总任务数：20 × 3 = **60 个任务**（从 500 减少到 60）

### 2. 进一步优化

如果还是太慢，可以：

```python
from configs.ensemble2_config import Ensemble2Config

config = Ensemble2Config(
    n_models=10,  # 进一步减少到 10 个模型
    stacking_config={
        'cv': 3,  # 使用 3 折 CV
        # ...
    }
)
```

### 3. 减少每个模型的复杂度

在 `ensemble2_config.py` 中修改超参数范围：

```python
xgb_search_space = {
    'n_estimators': (50, 500),  # 减少树的数量范围
    # ...
}
```

### 4. 使用更少的并行任务

```python
config = Ensemble2Config(
    n_jobs=4,  # 使用 4 个核心而不是全部
)
```

## 训练时间估算

### 当前配置（20 模型，3 折 CV）

- 总任务数：20 × 3 = 60
- 每个任务：~0.5-2 分钟
- **预计总时间：30-120 分钟**（0.5-2 小时）

### 原配置（100 模型，5 折 CV）

- 总任务数：100 × 5 = 500
- 每个任务：~0.5-2 分钟
- **预计总时间：250-1000 分钟**（4-16 小时）

## 如果训练被终止

### 原因

1. **内存不足**：
   - 100 个模型占用大量内存
   - 系统 OOM (Out of Memory) 杀死进程

2. **手动终止**：
   - 用户按 Ctrl+C 或 kill 命令

3. **超时**：
   - 某些系统有进程超时限制

### 解决方案

1. **减少模型数量**（已优化）
2. **增加系统内存**（如果可能）
3. **使用更少的并行任务**：
   ```python
   config.n_jobs = 2  # 减少并行度
   ```

## 监控训练进度

### 查看日志

训练开始时会显示：
```
Starting Ensemble2 training:
  - Number of base models: 20
  - Cross-validation folds: 3
  - Total training tasks: 60 (this will take a while...)
  - Estimated time: ~30.0 minutes (rough estimate)
```

### 进度估算

- 如果看到 10 个 `Done 3 out of 3` 消息 → 完成了 10/20 = 50%
- 如果看到 20 个 `Done 3 out of 3` 消息 → 完成了 20/20 = 100%（基模型训练完成）
- 之后还需要训练元学习器

## 建议

1. **先测试小规模**：
   - 使用 10-20 个模型
   - 使用 3 折 CV
   - 验证效果后再增加

2. **监控资源**：
   - 使用 `htop` 或 `top` 监控 CPU/内存
   - 确保有足够资源

3. **耐心等待**：
   - Stacking 训练需要时间
   - 每个 `Done` 消息都是进展

4. **如果太慢**：
   - 考虑使用 Voting 而不是 Stacking
   - 或者减少模型数量到 10 个以下

