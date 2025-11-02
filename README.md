# Office Category Prediction - 模块化机器学习项目

一个完整的、模块化的机器学习训练框架，用于办公室类别分类任务。

---

## 📦 安装

```bash
# 创建并激活虚拟环境
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

---

## 🚀 快速开始

### 1. 完整训练流程（推荐新手）

```bash
# 步骤 1: 数据探索（了解数据）
python main.py --mode eda

# 步骤 2: 训练模型
python main.py --mode train

# 步骤 3: 生成预测
python main.py --mode predict

# 步骤 4: 特征审计（分析特征重要性）
python main.py --mode audit
```

**输出文件：**
- 模型：`outputs/models/pipeline.joblib`
- 预测：`outputs/predictions/submission.csv`
- 指标：`outputs/metrics/metrics.json`

### 2. 高级训练选项

```bash
# 交叉验证（更可靠的性能评估）
python main.py --mode cv

# 超参数调优（寻找最佳参数）
python main.py --mode tune
```

---

## 📂 项目结构

```
AI1010Final/
├── configs/                    # ⚙️ 配置文件
│   └── config.py              # 所有参数都在这里
│
├── feature_engineering/       # 🔧 特征工程
│   ├── encoders.py           # 编码器（频率、目标编码）
│   ├── wide_features.py      # 派生特征（40+ 个）
│   ├── transformers.py       # 转换器（log、缺失值）
│   └── preprocessor.py       # 预处理管道组装
│
├── modeling/                  # 🤖 模型定义
│   ├── xgboost_model.py      # XGBoost 模型
│   └── ensemble.py           # 集成模型
│
├── training/                  # 🎓 训练逻辑
│   ├── trainer.py            # 单次训练
│   └── cross_validator.py    # 交叉验证
│
├── hyperparameter_tuning/    # 🔍 超参数优化
│   └── tuner.py              # Optuna 调优
│
├── data_exploration/         # 📊 数据分析
│   ├── exploratory_analysis.py  # EDA
│   └── feature_audit.py         # 特征审计
│
├── data_cleaning/            # 🧹 数据清洗
├── utils/                    # 🛠️ 工具函数
├── main.py                   # 🚪 主入口
└── datasets/                 # 📁 数据目录
```

---

## 🎯 训练工作流程

### 方案 A: 快速实验（单次训练）

```
数据加载 → 特征工程 → 训练模型 → 评估 → 预测
   ↓           ↓          ↓        ↓      ↓
office_   preprocessor  XGBoost  metrics  submission.csv
train.csv  pipeline               .json
```

**命令：**
```bash
python main.py --mode train   # 训练
python main.py --mode predict # 预测
```

**适用场景：** 快速迭代、测试新特征

---

### 方案 B: 可靠评估（交叉验证）

```
数据加载 → 5折交叉验证 → 聚合结果
   ↓           ↓            ↓
office_    每折训练+评估   mean ± std
train.csv                  metrics
```

**命令：**
```bash
python main.py --mode cv
```

**适用场景：** 最终模型选择、性能报告

---

### 方案 C: 参数优化（超参数调优）

```
定义搜索空间 → Optuna 优化 → 找到最佳参数 → 重新训练
      ↓              ↓              ↓            ↓
  config.py     100 trials    best_params   final model
```

**命令：**
```bash
python main.py --mode tune  # 调优
# 然后将最佳参数复制到 configs/config.py
python main.py --mode train # 用最佳参数训练
```

**适用场景：** 性能调优、竞赛提分

---

## ⚙️ 配置管理

**所有参数都在 `configs/config.py` 中集中管理！**

### 常用配置示例

```python
# configs/config.py

# 1. 修改训练参数
class TrainConfig:
    test_size = 0.2        # 验证集比例
    n_splits = 5           # 交叉验证折数
    use_early_stopping = True  # 早停

# 2. 修改模型参数
class XGBParams:
    n_estimators = 1500    # 树的数量
    learning_rate = 0.06   # 学习率
    max_depth = 4          # 树深度
    subsample = 0.75       # 样本采样
    
# 3. 修改特征工程
class Columns:
    # 使用频率编码的列
    freq_encoding_cols = ['RoofType', 'ExteriorCovering1']
    
    # 使用目标编码的列
    target_encoding_cols = ['ZoningClassification', 'BuildingType']
```

**修改配置后无需改代码，直接运行即可！**

---

## 🔧 如何添加新功能

### 1️⃣ 添加新特征

**位置：** `feature_engineering/wide_features.py`

```python
class WideFeatureBuilder(BaseEstimator, TransformerMixin):
    def _add_custom_features(self, df: pd.DataFrame, out: Dict[str, Any]):
        """在这里添加你的自定义特征"""
        
        # 示例 1: 简单交互特征
        out["QualityTimesArea"] = df["OverallQual"] * df["GrLivArea"]
        
        # 示例 2: 条件特征
        out["HasPool"] = (df["PoolArea"] > 0).astype(int)
        
        # 示例 3: 比率特征
        total_area = df["TotalBsmtSF"] + df["GrLivArea"]
        out["BasementRatio"] = df["TotalBsmtSF"] / (total_area + 1e-6)
        
        # 示例 4: 领域知识特征
        out["PricePerSqft"] = df["SalePrice"] / (df["GrLivArea"] + 1)
```

**然后运行：**
```bash
python main.py --mode train  # 新特征会自动使用
```

---

### 2️⃣ 添加新模型

**步骤 1:** 在 `modeling/` 创建新文件，如 `lightgbm_model.py`

```python
from modeling import BaseModel
import lightgbm as lgb

class LightGBMModel(BaseModel):
    def build_model(self, **params):
        """初始化模型"""
        self.model_ = lgb.LGBMClassifier(**params)
        return self.model_
    
    def fit(self, X, y, **kwargs):
        """训练"""
        eval_set = kwargs.get('eval_set', None)
        self.model_.fit(
            X, y,
            eval_set=eval_set,
            callbacks=[lgb.early_stopping(50)]
        )
        return self
    
    def predict(self, X):
        """预测"""
        return self.model_.predict(X)
```

**步骤 2:** 在 `main.py` 中添加对新模型的支持

```python
# main.py 中找到模型创建部分
if config.models.model_type == "xgboost":
    model = XGBoostModel(config=config.models.xgb_params)
elif config.models.model_type == "lightgbm":  # 新增
    from modeling.lightgbm_model import LightGBMModel
    model = LightGBMModel(config=config.models.lgb_params)
```

**步骤 3:** 在 `configs/config.py` 添加配置

```python
@dataclass
class ModelsConfig:
    model_type: str = "lightgbm"  # 修改这里
    lgb_params: dict = field(default_factory=lambda: {
        'n_estimators': 1000,
        'learning_rate': 0.05,
        ...
    })
```

---

### 3️⃣ 添加新编码器

**位置：** `feature_engineering/encoders.py`

```python
from sklearn.base import BaseEstimator, TransformerMixin

class MyCustomEncoder(BaseEstimator, TransformerMixin):
    """自定义编码器"""
    
    def __init__(self, columns=None):
        self.columns = columns
        self.mapping_ = {}
    
    def fit(self, X, y=None):
        """学习编码映射"""
        for col in self.columns:
            # 你的编码逻辑
            self.mapping_[col] = X[col].value_counts().to_dict()
        return self
    
    def transform(self, X):
        """应用编码"""
        X = X.copy()
        for col in self.columns:
            X[col] = X[col].map(self.mapping_[col]).fillna(0)
        return X
```

**然后在 `feature_engineering/preprocessor.py` 中使用：**

```python
from .encoders import MyCustomEncoder

# 在 build_preprocessor 函数中添加
transformers.append((
    "my_encoder",
    MyCustomEncoder(columns=['MyColumn']),
    ['MyColumn']
))
```

---

### 4️⃣ 修改超参数搜索空间

**位置：** `hyperparameter_tuning/tuner.py`

```python
def _suggest_xgb_params(self, trial: optuna.Trial) -> dict:
    """定义搜索空间"""
    return {
        'n_estimators': trial.suggest_int('n_estimators', 500, 3000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        
        # 添加新参数
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0, 5),
    }
```

---

## 📊 输出文件说明

```
outputs/
├── models/
│   ├── pipeline.joblib           # 完整训练管道（包含预处理+模型）
│   └── cv/
│       ├── fold_1.joblib          # 各折模型
│       └── cv_summary.json        # CV 结果汇总
│
├── metrics/
│   └── metrics.json               # 评估指标（精度、召回、F1等）
│
├── predictions/
│   └── submission.csv             # 测试集预测结果
│
├── eda_report.json                # 数据探索报告
├── feature_audit.json             # 特征重要性分析
└── tuning_results.json            # 超参数调优结果
```

---

## 💡 最佳实践

### 典型工作流程

```bash
# 1. 第一次训练
python main.py --mode eda      # 了解数据
python main.py --mode train    # 快速训练
python main.py --mode audit    # 分析特征

# 2. 改进特征（修改 wide_features.py）
python main.py --mode train    # 测试新特征

# 3. 参数调优
python main.py --mode tune     # 找最佳参数
# 将最佳参数复制到 configs/config.py

# 4. 最终训练
python main.py --mode cv       # 交叉验证评估
python main.py --mode train    # 训练最终模型
python main.py --mode predict  # 生成提交文件
```

### 调试技巧

```bash
# 1. 检查数据
python main.py --mode eda

# 2. 检查特征重要性
python main.py --mode audit

# 3. 测试新特征（修改 configs/config.py 中的 test_size）
# test_size = 0.5  # 加快训练速度
python main.py --mode train
```

---

## 🔍 常见问题

**Q: 如何快速测试新特征？**
```python
# configs/config.py
class TrainConfig:
    test_size = 0.5  # 减少训练数据，加快速度
    
class XGBParams:
    n_estimators = 100  # 减少树的数量
```

**Q: 训练太慢怎么办？**
```python
# configs/config.py
class XGBParams:
    n_jobs = -1  # 使用所有 CPU 核心
    tree_method = 'hist'  # 使用更快的算法
```

**Q: 如何保存多个模型版本？**
```bash
python main.py --mode train
# 手动重命名模型
mv outputs/models/pipeline.joblib outputs/models/pipeline_v1.joblib
```

---

## 📚 核心特征

- ✅ **模块化设计** - 每个功能独立，易于修改
- ✅ **配置驱动** - 修改参数无需改代码
- ✅ **Sklearn 兼容** - 所有转换器遵循标准接口
- ✅ **完整管道** - 预处理 + 模型一体化
- ✅ **40+ 特征** - 涵盖年龄、面积、质量、交互等
- ✅ **多种训练模式** - 单次/交叉验证/调优
- ✅ **特征分析** - 重要性/漂移/相关性

---

**开始训练吧！** 🚀

```bash
python main.py --mode train
```