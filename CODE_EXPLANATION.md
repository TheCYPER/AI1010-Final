# Detailed Code Explanation

This document provides a comprehensive explanation of the Office Category Prediction machine learning codebase.

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Main Entry Point (`main.py`)](#main-entry-point-mainpy)
4. [Configuration System](#configuration-system)
5. [Data Pipeline](#data-pipeline)
6. [Feature Engineering](#feature-engineering)
7. [Model Definitions](#model-definitions)
8. [Training Workflow](#training-workflow)
9. [Key Components Deep Dive](#key-components-deep-dive)

---

## 🎯 Project Overview

This is a **multi-class classification** project that predicts office building categories (0-4) based on 79 features. The codebase follows a **modular, configuration-driven architecture** that makes it easy to experiment with different models, features, and hyperparameters.

**Key Design Principles:**
- **Modularity**: Each component (data cleaning, feature engineering, modeling) is independent
- **Configuration-driven**: All parameters centralized in `configs/config.py`
- **Sklearn-compatible**: All transformers follow sklearn's `BaseEstimator`/`TransformerMixin` interface
- **Reproducible**: Fixed random seeds and deterministic operations

---

## 🏗️ Architecture

### High-Level Flow

```
┌─────────────┐
│  main.py    │  ← Entry point, routes to different modes
└──────┬──────┘
       │
       ├───→ EDA Mode → ExploratoryAnalysis
       ├───→ Train Mode → Trainer → Model
       ├───→ CV Mode → CrossValidator
       ├───→ Tune Mode → OptunaTuner
       ├───→ Predict Mode → Load pipeline → Predict
       └───→ Audit Mode → FeatureAuditor
```

### Component Structure

```
configs/              # Configuration management
  └── config.py       # All parameters in one place

data_cleaning/        # Data preprocessing
  ├── column_types.py      # Infer numeric vs categorical
  ├── missing_handler.py   # Handle missing values
  └── outlier_handler.py   # Handle outliers

feature_engineering/  # Feature creation
  ├── preprocessor.py      # Main preprocessing pipeline
  ├── encoders.py          # Frequency, target encoding
  ├── wide_features.py     # 40+ derived features
  ├── transformers.py      # Log transform, missing indicators
  └── statistical_features.py  # Group-by aggregations

modeling/             # Model definitions
  ├── base_model.py        # Abstract base class
  ├── xgboost_model.py     # XGBoost wrapper
  ├── catboost_model.py   # CatBoost wrapper
  ├── ensemble2_gpu.py    # GPU-accelerated ensemble
  └── ... (other models)

training/             # Training logic
  ├── trainer.py          # Single train/val split
  └── cross_validator.py  # K-fold cross-validation

utils/                # Utilities
  ├── logger.py           # Logging setup
  └── metrics.py         # Evaluation metrics
```

---

## 🚪 Main Entry Point (`main.py`)

The `main.py` file is the **command-line interface** that routes different operations.

### Key Functions

#### 1. **`main()`** - Argument Parser
```python
parser.add_argument('--mode', choices=['eda', 'train', 'cv', 'tune', 'predict', 'audit'])
```
- Parses command-line arguments
- Loads configuration
- Routes to appropriate function based on mode

#### 2. **`run_eda(config)`** - Exploratory Data Analysis
- Loads data
- Generates statistics, distributions, correlations
- Saves report to `outputs/eda_report.json`

#### 3. **`run_train(config)`** - Single Training Run
- Creates `Trainer` instance
- Runs complete pipeline: load → split → preprocess → train → evaluate → save
- Outputs: `pipeline.joblib`, `metrics.json`

#### 4. **`run_cross_validation(config)`** - K-Fold CV
- Creates `CrossValidator` instance
- Trains model on each fold
- Aggregates results (mean ± std)
- More reliable performance estimate

#### 5. **`run_hyperparameter_tuning(config)`** - Optuna Optimization
- Uses Optuna to search hyperparameter space
- Evaluates each trial with cross-validation
- Saves best parameters to `tuning_results.json`

#### 6. **`run_prediction(config, model_path)`** - Test Predictions
- Loads trained pipeline
- Loads test data
- Generates predictions
- Saves `submission.csv` with format: `Id, OfficeCategory`

#### 7. **`run_feature_audit(config)`** - Feature Analysis
- Loads trained model
- Computes feature importance
- Permutation importance
- Adversarial validation (train/val distribution check)
- Correlation analysis

### Environment Setup

```python
os.environ.setdefault("JOBLIB_MULTIPROCESSING", "0")  # Threading backend
os.environ.setdefault("MPLCONFIGDIR", "outputs/mpl_config")  # Matplotlib cache
```

---

## ⚙️ Configuration System

All parameters are centralized in `configs/config.py` using **dataclasses**.

### Configuration Classes

#### 1. **`PathConfig`** - File Paths
```python
train_csv: str = "datasets/office_train.csv"
test_csv: str = "datasets/office_test.csv"
models_dir: str = "outputs/models"
```
- Defines all input/output paths
- Auto-creates directories in `__post_init__`

#### 2. **`ColumnConfig`** - Column Management
```python
target: str = "OfficeCategory"
drop_columns: List[str] = [...]  # Low-quality/redundant columns
business_missing_col: str = "ConferenceRoomQuality"  # Special missing handling
```
- Specifies which columns to drop
- Identifies target variable
- Business-logic missing values

#### 3. **`ModelConfig`** - Model Selection & Parameters
```python
model_type: str = "ensemble2_gpu"  # Current best model
xgb_params: Dict = {...}          # XGBoost hyperparameters
catboost_params: Dict = {...}      # CatBoost hyperparameters
```
- **Model selection**: Choose which model to use
- **Hyperparameters**: All model-specific parameters
- Supports: XGBoost, CatBoost, LightGBM, TabNet, MLP, KNN, Logistic, SVM, Naive Bayes, Ridge, Extra Trees, Ensembles

#### 4. **`FeatureEngineeringConfig`** - Feature Engineering
```python
freq_encoding_cols: List[str] = ['RoofType', ...]      # Frequency encoding
target_encoding_cols: List[str] = ['ZoningClassification', ...]  # Target encoding
log_transform_cols: List[str] = ['PlotSize']           # Log transform
```
- Encoding strategies
- Statistical aggregations
- Transformations

#### 5. **`TrainingConfig`** - Training Settings
```python
test_size: float = 0.2              # Validation split
n_splits: int = 5                   # CV folds
use_class_weights: bool = True      # Handle class imbalance
use_early_stopping: bool = True    # Prevent overfitting
```
- Train/validation split
- Cross-validation settings
- Class balancing
- Early stopping

#### 6. **`HyperparameterTuningConfig`** - Tuning Settings
```python
method: str = "optuna"              # Optimization method
n_trials: int = 50                  # Number of trials
cv_folds: int = 5                   # CV for each trial
search_space: Dict = {...}         # Parameter ranges
```

### Usage Example

```python
from configs import Config

config = Config()
config.models.model_type = "xgboost"
config.models.xgb_params['n_estimators'] = 2000
config.training.test_size = 0.15
```

**Benefits:**
- ✅ No code changes needed to modify parameters
- ✅ Easy to experiment with different settings
- ✅ Version control friendly (config changes are clear)

---

## 📊 Data Pipeline

### 1. Data Loading (`Trainer.load_data()`)

```python
df = pd.read_csv(config.paths.train_csv)
X = df.drop(columns=[target])
y = df[target]
```

- Loads CSV into pandas DataFrame
- Separates features (X) and target (y)
- Logs data shape and target distribution

### 2. Column Type Inference (`data_cleaning/column_types.py`)

```python
def infer_column_types(df, target):
    cat_cols = [c for c in features.columns 
                if features[c].dtype == "object"]
    num_cols = [c for c in features.columns 
                if c not in cat_cols]
    return num_cols, cat_cols
```

**Logic:**
- **Categorical**: `object` or `category` dtype
- **Numeric**: Everything else (int, float)

### 3. Data Splitting (`Trainer.split_data()`)

```python
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y  # Maintain class distribution
)
```

- **Stratified split**: Ensures train/val have same class distribution
- **Reproducible**: Fixed random seed

---

## 🔧 Feature Engineering

The feature engineering pipeline is built in `feature_engineering/preprocessor.py` using sklearn's `ColumnTransformer`.

### Pipeline Structure

```
Input DataFrame
    │
    ├─── Numeric Columns ───→ [KNN Imputer] → [Log Transform] → [Scaler]
    │
    ├─── Categorical Columns ───→ [Missing Imputer] → [OneHot Encoder]
    │
    ├─── Wide Features ───→ [WideFeatureBuilder] → [Imputer]
    │
    ├─── Frequency Encoding ───→ [FrequencyEncoder]
    │
    ├─── Target Encoding ───→ [MultiClassTargetEncoder] (needs y)
    │
    ├─── Statistical Aggregation ───→ [StatisticalAggregator]
    │
    └─── Business Missing ───→ [BusinessMissingIndicator]
    
    ↓
    
Transformed Array (numpy)
```

### Key Components

#### 1. **Numeric Pipeline**

**KNN Imputation** (`transformers.py`):
```python
KNNImputerWithIndicators(n_neighbors=5)
```
- Uses KNN to impute missing values (considers feature relationships)
- Adds missing indicators (binary flags for missingness)

**Log Transform** (`transformers.py`):
```python
Log1pTransformer(col_names=['PlotSize'])
```
- Applies `log1p(x) = log(1 + x)` to handle skewed distributions
- Only applied to specified columns

**Scaler**:
- **RobustScaler**: For linear models (SVM, Ridge, Logistic, MLP, KNN) - robust to outliers
- **StandardScaler**: For tree models (XGBoost, CatBoost, LightGBM) and deep learning (TabNet)

#### 2. **Categorical Pipeline**

**Missing Imputation**:
```python
SimpleImputer(strategy="constant", fill_value="__MISSING__")
```
- Replaces missing with special token

**One-Hot Encoding**:
```python
OneHotEncoder(handle_unknown="ignore", sparse_output=False)
```
- Converts categories to binary columns
- Handles unseen categories gracefully

#### 3. **Wide Features** (`wide_features.py`)

The `WideFeatureBuilder` creates **40+ derived features**:

**Age Features:**
- `BuildingAge = YearListed - ConstructionYear`
- `YearsSinceRenovation = YearListed - RenovationYear`
- `IsRenovated = (RenovationYear > ConstructionYear)`

**Area Features:**
- `TotalLivingArea = GroundFloorArea + UpperFloorArea`
- `TotalBasementArea = sum of basement components`

**Ratio Features:**
- `BasementFinishRatio = Finished / Total`
- `OfficeSpaceRatio = OfficeSpace / TotalLivingArea`
- `ParkingRatio = ParkingSpots / TotalRooms`

**Quality Combinations:**
- `OverallQuality = BuildingGrade + BuildingCondition`
- `ExteriorScore = ExteriorQuality + ExteriorCondition`
- `BasementScore = BasementQuality + BasementCondition`

**Temporal Features:**
- `BuildingLifeStage = binned(BuildingAge)`
- `RenovationEffectiveness = YearsSinceRenovation / BuildingAge`

**Interaction Features:**
- `QualityAreaProximity = OverallQuality * TotalLivingArea * ProximityScore`
- `BasementEfficiency = FinishedBasementArea / TotalBasementArea`

**Domain Knowledge Features:**
- `RoomSizeAdequacy = TotalLivingArea / TotalRooms`
- `ParkingAdequacy = ParkingSpots / TotalRooms`

#### 4. **Frequency Encoding** (`encoders.py`)

```python
FrequencyEncoder(columns=['RoofType', 'ExteriorCovering1'])
```

**How it works:**
- Counts frequency of each category in training data
- Maps categories to their frequency values
- Useful for high-cardinality categoricals

**Example:**
```
RoofType: "Gable" → 0.45 (appears in 45% of data)
RoofType: "Hip" → 0.30
```

#### 5. **Target Encoding** (`encoders.py`)

```python
MultiClassTargetEncoder(columns=['ZoningClassification'], alpha=10.0)
```

**How it works:**
- For each category, computes mean target value (smoothed)
- Uses formula: `(count * mean + alpha * global_mean) / (count + alpha)`
- **Smoothing (alpha)**: Prevents overfitting to rare categories

**Example:**
```
ZoningClassification="Residential":
  mean(OfficeCategory) = 2.3
  count = 1000
  global_mean = 2.0
  alpha = 10.0
  encoded_value = (1000 * 2.3 + 10 * 2.0) / (1000 + 10) ≈ 2.3
```

**⚠️ Important**: Target encoding requires `y` during `fit()`, so it's applied in the preprocessor pipeline.

#### 6. **Statistical Aggregation** (`statistical_features.py`)

```python
StatisticalAggregator(
    groupby_cols=('ZoningClassification', 'BuildingType'),
    agg_cols=('TotalLivingArea', 'BuildingAge', 'OverallQuality')
)
```

**How it works:**
- Groups by categorical columns
- Computes statistics (mean, std, min, max) for numeric columns
- Creates features like: `mean_TotalLivingArea_by_ZoningClassification`

**Example:**
```
ZoningClassification="Commercial":
  mean(TotalLivingArea) = 2500
  std(TotalLivingArea) = 500
  → Features: [2500, 500, ...]
```

#### 7. **Business Missing Indicator** (`transformers.py`)

```python
BusinessMissingIndicator()
```

- For `ConferenceRoomQuality`: Missing might mean "no conference room"
- Creates binary indicator: `1` if missing, `0` otherwise

### Model-Specific Adjustments

**TabNet (Deep Learning):**
- **Simplified feature engineering**: TabNet can learn feature interactions automatically
- Disables wide features and statistical aggregation
- Uses StandardScaler (required for neural networks)

**Tree Models (XGBoost, CatBoost, LightGBM):**
- Full feature engineering enabled
- StandardScaler (not required but harmless)

**Linear Models (SVM, Ridge, Logistic, MLP, KNN):**
- Full feature engineering enabled
- RobustScaler (more robust to outliers)

---

## 🤖 Model Definitions

All models inherit from `BaseModel` (abstract base class) in `modeling/base_model.py`.

### BaseModel Interface

```python
class BaseModel(ABC):
    def build_model(self, **kwargs):  # Abstract
    def fit(self, X, y, **kwargs):    # Abstract
    def predict(self, X):             # Abstract
    def predict_proba(self, X):       # Optional
    def get_feature_importance(self): # Optional
```

### Model Implementations

#### 1. **XGBoost** (`modeling/xgboost_model.py`)

```python
class XGBoostModel(BaseModel):
    def build_model(self, num_classes):
        self.model_ = XGBClassifier(
            objective='multi:softprob',
            n_estimators=1500,
            learning_rate=0.01,
            max_depth=7,
            ...
        )
```

**Features:**
- Gradient boosting with tree-based learners
- Supports early stopping
- Feature importance available
- GPU support via `tree_method='gpu_hist'`

#### 2. **CatBoost** (`modeling/catboost_model.py`)

```python
class CatBoostModel(BaseModel):
    def build_model(self, num_classes):
        self.model_ = CatBoostClassifier(
            iterations=1700,
            learning_rate=0.01,
            depth=7,
            task_type='GPU',  # GPU acceleration
            ...
        )
```

**Features:**
- Handles categorical features natively
- GPU acceleration
- Robust to overfitting

#### 3. **Ensemble2 GPU** (`modeling/ensemble2_gpu.py`)

**Architecture:**
```
Base Models (Level 0):
  ├── CatBoost (GPU) × 3 (different hyperparameters)
  ├── XGBoost (GPU) × 3
  ├── RandomForest × 2
  └── ExtraTrees × 2

Meta-Learner (Level 1):
  └── LogisticRegression (combines base model predictions)
```

**How Stacking Works:**
1. **Train base models** on training data
2. **Generate predictions** on validation data (out-of-fold)
3. **Train meta-learner** on base model predictions
4. **Final prediction** = meta-learner(base_model_predictions)

**Benefits:**
- **Diversity**: Different algorithms (CatBoost, XGBoost, RF, ExtraTrees)
- **Hyperparameter diversity**: Multiple configurations per algorithm
- **GPU acceleration**: CatBoost and XGBoost use GPU
- **Stacking**: Learns optimal combination of base models

**Implementation:**
```python
base_models = [
    ('catboost_0', CatBoostClassifier(...)),
    ('catboost_1', CatBoostClassifier(...)),
    ('xgb_0', XGBClassifier(...)),
    ...
]

meta_learner = LogisticRegression()

ensemble = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_learner,
    cv=5,  # Cross-validation for out-of-fold predictions
    n_jobs=-1
)
```

#### 4. **Other Models**

- **LightGBM**: Fast gradient boosting
- **TabNet**: Deep learning for tabular data
- **MLP**: Multi-layer perceptron (neural network)
- **KNN**: K-nearest neighbors
- **Logistic Regression**: Linear classifier
- **SVM**: Support vector machine
- **Naive Bayes**: Probabilistic classifier
- **Ridge**: Linear classifier with L2 regularization
- **Extra Trees**: Extremely randomized trees

### Model Creation (`Trainer.create_model()`)

```python
model_type = config.models.model_type.lower()

if model_type == "xgboost":
    model = XGBoostModel(config=config.models.xgb_params)
elif model_type == "ensemble2_gpu":
    model = create_ensemble2_gpu(config=ensemble2_config, num_classes=num_classes)
...
```

---

## 🎓 Training Workflow

### Single Training Run (`Trainer.run()`)

```python
def run(self):
    # 1. Load data
    X, y = self.load_data()
    
    # 2. Split data
    X_train, X_val, y_train, y_val = self.split_data(X, y)
    
    # 3. Create model
    model = self.create_model(num_classes=len(np.unique(y)))
    
    # 4. Train
    results = self.train(model, X_train, y_train, X_val, y_val)
    
    # 5. Save artifacts
    self.save_artifacts(results)
    
    return results
```

### Training Process (`Trainer.train()`)

```python
def train(self, model, X_train, y_train, X_val, y_val):
    # 1. Build preprocessor
    self.build_preprocessor(X_train)
    
    # 2. Fit preprocessor on training data
    X_train_transformed = self.preprocessor_.fit_transform(X_train, y_train)
    X_val_transformed = self.preprocessor_.transform(X_val)
    
    # 3. Compute sample weights (class balancing)
    sample_weight = self.compute_sample_weights(y_train)
    
    # 4. Train model
    model.fit(
        X=X_train_transformed,
        y=y_train,
        eval_set=[(X_val_transformed, y_val)],
        sample_weight=sample_weight,
        early_stopping_rounds=80
    )
    
    # 5. Evaluate
    results = evaluate_model(model, X_train_transformed, y_train, 
                            X_val_transformed, y_val)
    
    # 6. Build pipeline (preprocessor + model)
    self.pipeline_ = Pipeline([
        ("preprocessor", self.preprocessor_),
        ("model", model.model_)
    ])
    
    return results
```

### Class Balancing

```python
def compute_sample_weights(self, y):
    cls, cnt = np.unique(y, return_counts=True)
    total = len(y)
    wmap = {c: (total / cnt[i]) ** power for i, c in enumerate(cls)}
    w = y.map(wmap).astype(float).values
    w = w / np.mean(w)  # Normalize to mean=1
    return w
```

**How it works:**
- Rare classes get higher weights
- Formula: `weight = (total / class_count) ** power`
- Normalized so mean weight = 1

### Early Stopping

```python
model.fit(
    eval_set=[(X_val_transformed, y_val)],
    early_stopping_rounds=80
)
```

- Monitors validation performance
- Stops training if no improvement for 80 rounds
- Prevents overfitting

### Pipeline Saving

```python
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])
joblib.dump(pipeline, "outputs/models/pipeline.joblib")
```

**Benefits:**
- **Single file**: Preprocessor + model together
- **Reproducible**: Same preprocessing applied to test data
- **Easy deployment**: Load and predict

---

## 🔍 Key Components Deep Dive

### 1. Cross-Validation (`training/cross_validator.py`)

```python
cv = CrossValidator(config)
results = cv.run()
```

**Process:**
1. Split data into K folds (default: 5)
2. For each fold:
   - Train on K-1 folds
   - Evaluate on held-out fold
3. Aggregate results: mean ± std

**Output:**
```json
{
  "accuracy_mean": 0.8550,
  "accuracy_std": 0.0012,
  "f1_mean": 0.8520,
  ...
}
```

### 2. Hyperparameter Tuning (`hyperparameter_tuning/tuner.py`)

**Optuna Integration:**
```python
def objective(trial):
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        ...
    }
    model = XGBoostModel(config=params)
    score = cross_val_score(model, X, y, cv=5)
    return score.mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)
```

**Search Strategies:**
- **TPE (Tree-structured Parzen Estimator)**: Bayesian optimization
- **Random**: Random search
- **Grid**: Exhaustive search (small spaces only)

### 3. Feature Audit (`data_exploration/feature_audit.py`)

**Analyses:**
1. **Feature Importance**: Tree-based models provide this
2. **Permutation Importance**: Shuffle feature, measure performance drop
3. **Adversarial Validation**: Train classifier to distinguish train/val
   - If easy to distinguish → distribution shift (bad)
4. **Correlation**: High correlation → redundancy

### 4. Evaluation Metrics (`utils/metrics.py`)

```python
def evaluate_model(model, X_train, y_train, X_val, y_val):
    return {
        'train_metrics': {
            'accuracy': accuracy_score(y_train, y_pred_train),
            'f1_macro': f1_score(y_train, y_pred_train, average='macro'),
            ...
        },
        'val_metrics': {...},
        'confusion_matrix': confusion_matrix(y_val, y_pred_val),
        'val_report': classification_report(y_val, y_pred_val)
    }
```

**Metrics:**
- **Accuracy**: Overall correctness
- **F1 Score**: Harmonic mean of precision and recall
- **Precision/Recall**: Per-class metrics
- **Confusion Matrix**: Per-class error analysis

---

## 🎯 Usage Examples

### Example 1: Quick Training

```bash
python main.py --mode train
```

**What happens:**
1. Loads `office_train.csv`
2. Splits 80/20 train/val
3. Builds preprocessor (feature engineering)
4. Trains XGBoost (or configured model)
5. Saves `pipeline.joblib` and `metrics.json`

### Example 2: Cross-Validation

```bash
python main.py --mode cv
```

**What happens:**
1. Loads data
2. 5-fold cross-validation
3. Trains 5 models (one per fold)
4. Aggregates results
5. Saves CV summary

### Example 3: Hyperparameter Tuning

```bash
python main.py --mode tune
```

**What happens:**
1. Loads data
2. Runs 50 Optuna trials
3. Each trial: 5-fold CV
4. Saves best parameters to `tuning_results.json`

### Example 4: Prediction

```bash
python main.py --mode predict --model_path outputs/models/pipeline.joblib
```

**What happens:**
1. Loads trained pipeline
2. Loads `office_test.csv`
3. Applies preprocessing
4. Generates predictions
5. Saves `submission.csv`

### Example 5: Feature Analysis

```bash
python main.py --mode audit
```

**What happens:**
1. Loads trained model
2. Computes feature importance
3. Permutation importance
4. Adversarial validation
5. Correlation analysis
6. Saves `feature_audit.json`

---

## 🔑 Key Design Decisions

### 1. **Why ColumnTransformer?**

- **Modular**: Each feature type processed independently
- **Efficient**: Parallel processing possible
- **Maintainable**: Easy to add/remove transformers

### 2. **Why KNN Imputation?**

- **Better accuracy**: Considers feature relationships
- **Missing indicators**: Captures missingness as signal

### 3. **Why Multiple Encoding Strategies?**

- **Frequency encoding**: For high-cardinality categoricals
- **Target encoding**: For predictive categoricals
- **One-hot**: For low-cardinality categoricals

### 4. **Why Ensemble2 GPU?**

- **Diversity**: Multiple algorithms reduce variance
- **Stacking**: Learns optimal combination
- **GPU acceleration**: Faster training

### 5. **Why Configuration-Driven?**

- **Experimentation**: Easy to try different settings
- **Reproducibility**: Config changes are version-controlled
- **No code changes**: Modify config, not code

---

## 📝 Summary

This codebase implements a **production-ready machine learning pipeline** with:

✅ **Modular architecture** - Easy to extend and modify  
✅ **Configuration-driven** - All parameters in one place  
✅ **Comprehensive feature engineering** - 40+ derived features  
✅ **Multiple model support** - 12+ algorithms  
✅ **GPU acceleration** - For CatBoost and XGBoost  
✅ **Ensemble methods** - Stacking for improved performance  
✅ **Robust evaluation** - Cross-validation, metrics, feature analysis  
✅ **Reproducible** - Fixed seeds, deterministic operations  

The code follows **best practices** for ML pipelines:
- Sklearn-compatible interfaces
- Proper train/validation/test splits
- Early stopping to prevent overfitting
- Class balancing for imbalanced data
- Comprehensive logging and error handling

---

**For questions or issues, refer to the README.md or check the code comments.**

