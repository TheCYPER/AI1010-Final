# Office Category Prediction - Modular Machine Learning Project

A complete, modular machine learning training framework for office category classification.

---

## 📦 Installation

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Quick Start

### 1. Complete Training Pipeline (Recommended for Beginners)

```bash
# Step 1: Data exploration
python main.py --mode eda

# Step 2: Train model
python main.py --mode train

# Step 3: Generate predictions
python main.py --mode predict

# Step 4: Feature audit (analyze feature importance)
python main.py --mode audit
```

**Output Files:**
- Model: `outputs/models/pipeline.joblib`
- Predictions: `outputs/predictions/submission.csv`
- Metrics: `outputs/metrics/metrics.json`

### 2. Advanced Training Options

```bash
# Cross-validation (more reliable performance evaluation)
python main.py --mode cv

# Hyperparameter tuning
python main.py --mode tune
```

---

## 📂 Project Structure

```
AI1010Final/
├── configs/                    # ⚙️ Configuration files
│   ├── config.py              # Main configuration
│   └── ensemble2_config.py    # Ensemble2 configuration
│
├── feature_engineering/       # 🔧 Feature engineering
│   ├── encoders.py           # Encoders (frequency, target encoding)
│   ├── wide_features.py      # Derived features (40+ features)
│   ├── transformers.py       # Transformers (log, missing values)
│   └── preprocessor.py       # Preprocessing pipeline assembly
│
├── modeling/                  # 🤖 Model definitions
│   ├── base_model.py         # Base model interface
│   ├── xgboost_model.py      # XGBoost model
│   ├── catboost_model.py     # CatBoost model
│   ├── ensemble.py           # Voting ensemble
│   ├── ensemble2.py          # Stacking ensemble (CPU)
│   └── ensemble2_gpu.py      # Stacking ensemble (GPU)
│
├── training/                  # 🎓 Training logic
│   ├── trainer.py            # Single training
│   └── cross_validator.py    # Cross-validation
│
├── hyperparameter_tuning/    # 🔍 Hyperparameter optimization
│   └── tuner.py              # Optuna tuning
│
├── data_exploration/         # 📊 Data analysis
│   ├── exploratory_analysis.py  # EDA
│   └── feature_audit.py         # Feature audit
│
├── data_cleaning/            # 🧹 Data cleaning
├── utils/                    # 🛠️ Utility functions
├── main.py                   # 🚪 Main entry point
└── datasets/                 # 📁 Data directory
```

---

## 🎯 Training Workflows

### Option A: Quick Experiment (Single Training)

```
Data Loading → Feature Engineering → Train Model → Evaluate → Predict
   ↓                ↓                    ↓            ↓          ↓
office_        preprocessor          XGBoost      metrics    submission.csv
train.csv      pipeline                          .json
```

**Commands:**
```bash
python main.py --mode train   # Train
python main.py --mode predict # Predict
```

**Use Case:** Quick iteration, testing new features

---

### Option B: Reliable Evaluation (Cross-Validation)

```
Data Loading → 5-Fold CV → Aggregate Results
   ↓              ↓              ↓
office_      Train+Eval      mean ± std
train.csv    per fold        metrics
```

**Command:**
```bash
python main.py --mode cv
```

**Use Case:** Final model selection, performance reporting

---

### Option C: Parameter Optimization (Hyperparameter Tuning)

```
Define Search Space → Optuna Optimization → Best Params → Retrain
      ↓                    ↓                    ↓            ↓
  config.py           100 trials          best_params   final model
```

**Commands:**
```bash
python main.py --mode tune  # Tune
# Copy best parameters to configs/config.py
python main.py --mode train # Train with best parameters
```

**Use Case:** Performance optimization, competition scoring

---

## ⚙️ Configuration Management

**All parameters are centrally managed in `configs/config.py`!**

### Common Configuration Examples

```python
# configs/config.py

# 1. Modify training parameters
class TrainConfig:
    test_size = 0.2        # Validation set ratio
    n_splits = 5           # Cross-validation folds
    use_early_stopping = True  # Early stopping

# 2. Modify model parameters
class XGBParams:
    n_estimators = 1500    # Number of trees
    learning_rate = 0.06   # Learning rate
    max_depth = 4          # Tree depth
    subsample = 0.75       # Sample sampling
    
# 3. Modify feature engineering
class Columns:
    # Columns using frequency encoding
    freq_encoding_cols = ['RoofType', 'ExteriorCovering1']
    
    # Columns using target encoding
    target_encoding_cols = ['ZoningClassification', 'BuildingType']
```

**After modifying configuration, run directly without code changes!**

---

## 🔧 How to Add New Features

### 1️⃣ Add New Features

**Location:** `feature_engineering/wide_features.py`

```python
class WideFeatureBuilder(BaseEstimator, TransformerMixin):
    def _add_custom_features(self, df: pd.DataFrame, out: Dict[str, Any]):
        """Add your custom features here"""
        
        # Example 1: Simple interaction feature
        out["QualityTimesArea"] = df["OverallQual"] * df["GrLivArea"]
        
        # Example 2: Conditional feature
        out["HasPool"] = (df["PoolArea"] > 0).astype(int)
        
        # Example 3: Ratio feature
        total_area = df["TotalBsmtSF"] + df["GrLivArea"]
        out["BasementRatio"] = df["TotalBsmtSF"] / (total_area + 1e-6)
```

**Then run:**
```bash
python main.py --mode train  # New features will be used automatically
```

---

### 2️⃣ Add New Model

**Step 1:** Create new file in `modeling/`, e.g., `lightgbm_model.py`

```python
from modeling.base_model import BaseModel
import lightgbm as lgb

class LightGBMModel(BaseModel):
    def build_model(self, **params):
        self.model_ = lgb.LGBMClassifier(**params)
        return self.model_
    
    def fit(self, X, y, **kwargs):
        eval_set = kwargs.get('eval_set', None)
        self.model_.fit(
            X, y,
            eval_set=eval_set,
            callbacks=[lgb.early_stopping(50)]
        )
        return self
    
    def predict(self, X):
        return self.model_.predict(X)
```

**Step 2:** Add model support in `training/trainer.py` and `training/cross_validator.py`

**Step 3:** Add configuration in `configs/config.py`

---

### 3️⃣ Add New Encoder

**Location:** `feature_engineering/encoders.py`

```python
from sklearn.base import BaseEstimator, TransformerMixin

class MyCustomEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns
        self.mapping_ = {}
    
    def fit(self, X, y=None):
        for col in self.columns:
            self.mapping_[col] = X[col].value_counts().to_dict()
        return self
    
    def transform(self, X):
        X = X.copy()
        for col in self.columns:
            X[col] = X[col].map(self.mapping_[col]).fillna(0)
        return X
```

**Then use in `feature_engineering/preprocessor.py`**

---

### 4️⃣ Modify Hyperparameter Search Space

**Location:** `hyperparameter_tuning/tuner.py`

```python
def _suggest_xgb_params(self, trial: optuna.Trial) -> dict:
    return {
        'n_estimators': trial.suggest_int('n_estimators', 500, 3000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
    }
```

---

## 📊 Output Files

```
outputs/
├── models/
│   ├── pipeline.joblib           # Complete training pipeline
│   └── cv/
│       ├── fold_1.joblib          # Per-fold models
│       └── cv_summary.json        # CV results summary
│
├── metrics/
│   └── metrics.json               # Evaluation metrics
│
├── predictions/
│   └── submission.csv             # Test set predictions
│
├── eda_report.json                # Data exploration report
├── feature_audit.json             # Feature importance analysis
└── tuning_results.json            # Hyperparameter tuning results
```

---

## 💡 Best Practices

### Typical Workflow

```bash
# 1. First training
python main.py --mode eda      # Understand data
python main.py --mode train    # Quick training
python main.py --mode audit    # Analyze features

# 2. Improve features (modify wide_features.py)
python main.py --mode train    # Test new features

# 3. Parameter tuning
python main.py --mode tune     # Find best parameters
# Copy best parameters to configs/config.py

# 4. Final training
python main.py --mode cv       # Cross-validation evaluation
python main.py --mode train    # Train final model
python main.py --mode predict  # Generate submission
```

### Debugging Tips

```bash
# 1. Check data
python main.py --mode eda

# 2. Check feature importance
python main.py --mode audit

# 3. Test new features (modify test_size in configs/config.py)
# test_size = 0.5  # Speed up training
python main.py --mode train
```

---

## 🔍 FAQ

**Q: How to quickly test new features?**
```python
# configs/config.py
class TrainConfig:
    test_size = 0.5  # Reduce training data for speed
    
class XGBParams:
    n_estimators = 100  # Reduce number of trees
```

**Q: Training is too slow?**
```python
# configs/config.py
class XGBParams:
    n_jobs = -1  # Use all CPU cores
    tree_method = 'hist'  # Use faster algorithm
```

**Q: How to save multiple model versions?**
```bash
python main.py --mode train
# Manually rename model
mv outputs/models/pipeline.joblib outputs/models/pipeline_v1.joblib
```

---

## 📚 Key Features

- ✅ **Modular Design** - Each component is independent and easy to modify
- ✅ **Configuration-Driven** - Modify parameters without code changes
- ✅ **Sklearn Compatible** - All transformers follow standard interfaces
- ✅ **Complete Pipeline** - Preprocessing + model integration
- ✅ **Multiple Training Modes** - Single/CV/Tuning
- ✅ **Feature Analysis** - Importance/drift/correlation
- ✅ **Ensemble Support** - Voting and stacking ensembles with GPU acceleration

---

**Start Training!** 🚀

```bash
python main.py --mode train
```
