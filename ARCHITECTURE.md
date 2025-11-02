# Architecture Documentation

## 🏗️ System Architecture

This document describes the architecture of the ML pipeline, design decisions, and how components interact.

## Overview

The pipeline follows a **modular, layered architecture** inspired by production ML systems:

```
┌─────────────────────────────────────────────────────────┐
│                      main.py                             │
│                  (Orchestration Layer)                   │
└─────────────────────────────────────────────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
        ▼                  ▼                  ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│   configs/   │   │   training/  │   │   modeling/  │
│              │   │              │   │              │
│ • config.py  │   │ • trainer    │   │ • xgboost    │
│              │   │ • cv         │   │ • ensemble   │
└──────────────┘   └──────────────┘   └──────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
        ▼                  ▼                  ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│   feature_   │   │    data_     │   │hyperparameter│
│ engineering/ │   │  cleaning/   │   │   _tuning/   │
│              │   │              │   │              │
│ • encoders   │   │ • missing    │   │ • optuna     │
│ • wide_feat  │   │ • outliers   │   │              │
│ • stat_feat  │   │              │   │              │
└──────────────┘   └──────────────┘   └──────────────┘
                           │
                           ▼
                   ┌──────────────┐
                   │    utils/    │
                   │              │
                   │ • logger     │
                   │ • metrics    │
                   └──────────────┘
```

## Design Principles

### 1. **Separation of Concerns**

Each module has a single, well-defined responsibility:

- **configs**: Configuration management only
- **data_cleaning**: Data quality operations
- **feature_engineering**: Feature creation and transformation
- **modeling**: Model definitions and wrappers
- **training**: Training orchestration
- **hyperparameter_tuning**: HPO logic
- **data_exploration**: EDA and auditing
- **utils**: Cross-cutting utilities

### 2. **Sklearn Compatibility**

All transformers follow the sklearn API:
- `fit(X, y)` - Learn from training data
- `transform(X)` - Apply transformation
- `fit_transform(X, y)` - Shortcut for both
- `get_feature_names_out()` - Return feature names

This enables:
- Easy integration with `ColumnTransformer`
- Pipeline composition
- Serialization with joblib
- Cross-validation compatibility

### 3. **Configuration-Driven**

All hyperparameters and settings live in `configs/config.py`:

```python
@dataclass
class Config:
    paths: PathConfig
    columns: ColumnConfig
    models: ModelConfig
    features: FeatureEngineeringConfig
    training: TrainingConfig
    tuning: HyperparameterTuningConfig
```

Benefits:
- Easy experimentation (change config, not code)
- Reproducibility (save config with model)
- No hardcoded values scattered across codebase

### 4. **Composability**

Components are designed to work independently or together:

```python
# Use preprocessor alone
preprocessor = build_preprocessor(num_cols, cat_cols)
X_transformed = preprocessor.fit_transform(X, y)

# Use model alone
model = XGBoostModel(config=params)
model.fit(X_transformed, y)

# Compose into pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', model.model_)
])
```

### 5. **Extensibility**

Adding new features is straightforward:

**New Feature Type:**
```python
# feature_engineering/my_features.py
class MyCustomFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        # Learn from data
        return self
    
    def transform(self, X):
        # Create features
        return new_features
```

**New Model:**
```python
# modeling/my_model.py
class MyModel(BaseModel):
    def build_model(self, **kwargs):
        # Initialize model
        pass
    
    def fit(self, X, y, **kwargs):
        # Train
        pass
    
    def predict(self, X):
        # Predict
        pass
```

## Component Details

### Feature Engineering Pipeline

The feature engineering pipeline is the most complex component:

```
Input DataFrame
       │
       ├─→ Numeric Branch
       │      ├─→ Median Imputation + Indicators
       │      └─→ Log1p Transform (PlotSize)
       │
       ├─→ Categorical Branch (One-Hot)
       │      └─→ Missing → "__MISSING__" → OneHot
       │
       ├─→ Frequency Encoding Branch
       │      └─→ Map categories → frequency
       │
       ├─→ Target Encoding Branch
       │      └─→ Map categories → P(y|category) with smoothing
       │
       ├─→ Wide Features Branch
       │      ├─→ Age features (BuildingAge, etc.)
       │      ├─→ Area features (TotalLivingArea, etc.)
       │      ├─→ Ratio features (PlotCoverage, etc.)
       │      ├─→ Quality features (OverallQuality, etc.)
       │      ├─→ Temporal features (SeasonListed, etc.)
       │      ├─→ Interaction features (QualityAreaProximity)
       │      └─→ Domain knowledge (RoomSizeAdequacy, etc.)
       │
       ├─→ Statistical Aggregation Branch
       │      ├─→ Group by ZoningClassification
       │      ├─→ Group by BuildingType
       │      ├─→ Compute z-scores within groups
       │      └─→ Compute relative shifts
       │
       └─→ Business Missing Indicator
              └─→ ConferenceRoomQuality missing → 0/1
       
       ↓
Column Transformer concatenates all branches
       ↓
Final Feature Matrix (300-400 features)
```

### Training Flow

**Single Split Training:**

```
1. Load data
2. Train/val split (stratified)
3. Build preprocessor
4. Fit preprocessor on train only
5. Transform train and val
6. Compute sample weights (if enabled)
7. Train model with eval_set
8. Evaluate on both sets
9. Save pipeline + metrics
```

**Cross-Validation:**

```
1. Load data
2. Create StratifiedKFold splitter
3. For each fold:
   a. Split data
   b. Build preprocessor
   c. Fit on fold train
   d. Transform fold train and val
   e. Train model
   f. Evaluate
   g. Store results
4. Aggregate results (mean ± std)
5. Save summary
```

### Hyperparameter Tuning Flow

```
1. Load data
2. Build preprocessor (fit once)
3. Create Optuna study
4. For each trial:
   a. Sample hyperparameters
   b. Build model with sampled params
   c. Cross-validate on train set
   d. Return mean CV score
5. Select best parameters
6. Save results
```

## Data Flow

### Training Data Flow

```
CSV File
   │
   ▼
DataFrame (X, y)
   │
   ├─→ Train (80%)
   │      │
   │      ├─→ Preprocessor.fit(X_train, y)
   │      │      │
   │      │      └─→ Learn encodings, statistics, etc.
   │      │
   │      ├─→ Preprocessor.transform(X_train)
   │      │      │
   │      │      └─→ X_train_transformed (300-400 features)
   │      │
   │      └─→ Model.fit(X_train_transformed, y_train)
   │
   └─→ Val (20%)
          │
          └─→ Preprocessor.transform(X_val)
                 │
                 └─→ X_val_transformed
                        │
                        └─→ Model.predict(X_val_transformed)
                               │
                               └─→ Evaluation Metrics
```

### Test Prediction Flow

```
Test CSV
   │
   ▼
Test DataFrame
   │
   └─→ Pipeline.predict(X_test)
          │
          ├─→ Preprocessor.transform(X_test)
          │      │
          │      └─→ X_test_transformed
          │
          └─→ Model.predict(X_test_transformed)
                 │
                 └─→ Predictions
                        │
                        └─→ submission.csv
```

## Key Design Decisions

### 1. Why ColumnTransformer?

**Decision:** Use sklearn's `ColumnTransformer` for preprocessing

**Rationale:**
- Handles different column types elegantly
- Preserves column names (with `set_output(transform="pandas")`)
- Integrates seamlessly with `Pipeline`
- Serializable with joblib

**Trade-off:** Slightly more verbose than custom code, but much more maintainable

### 2. Why Separate Encoding Strategies?

**Decision:** Multiple encoding branches (frequency, target, one-hot)

**Rationale:**
- Different cardinality → different optimal encoding
- Target encoding for high-cardinality predictive features
- Frequency encoding for medium-cardinality
- One-hot for low-cardinality
- Prevents over-parameterization

### 3. Why Wide Features?

**Decision:** Create 40+ derived features upfront

**Rationale:**
- Tree models benefit from explicit interactions
- Domain knowledge >> automatic feature learning
- Interpretability (know what model uses)
- Faster than neural auto-feature learning

**Alternative:** Could use auto-feature engineering (autofeat), but:
- Less interpretable
- Can create too many features
- Computationally expensive

### 4. Why Multiple Imputation Strategies?

**Decision:** Median for numeric, constant for categorical

**Rationale:**
- Median robust to outliers
- Constant ("__MISSING__") preserves signal in missingness
- Add indicators to capture missing pattern importance

### 5. Why Sample Weighting?

**Decision:** Optional class-weighted training

**Rationale:**
- Office categories may be imbalanced
- Weighting helps model focus on rare classes
- Configurable (can enable/disable)

### 6. Why Optuna for Tuning?

**Decision:** Use Optuna over GridSearch/RandomSearch

**Rationale:**
- Bayesian optimization more efficient
- Supports pruning (early stopping of bad trials)
- Beautiful visualizations
- Can resume interrupted searches

**Trade-off:** Additional dependency, but worth it for speed

## Performance Considerations

### Memory Efficiency

1. **Avoid duplicate data**
   - Use `copy()` only when necessary
   - Transform in-place where possible

2. **Sparse matrices**
   - OneHotEncoder can use sparse (disabled for simplicity)
   - Could enable for very high cardinality

3. **Batch processing**
   - For very large datasets, could add batch processing
   - Current: assumes data fits in memory

### Computational Efficiency

1. **Parallel processing**
   - XGBoost: `n_jobs=-1` (uses all cores)
   - Cross-validation: could parallelize folds (not implemented)
   - Hyperparameter tuning: `n_jobs=-1` in CV

2. **Early stopping**
   - Prevents unnecessary training
   - Monitors validation loss

3. **Caching**
   - Preprocessor fit once, transform multiple times
   - Could add more aggressive caching

## Testing Strategy

### Unit Tests (Recommended)

```python
# tests/test_encoders.py
def test_frequency_encoder():
    X = pd.DataFrame({'col': ['a', 'b', 'a', 'c']})
    enc = FrequencyEncoder(cols=['col'])
    enc.fit(X)
    result = enc.transform(X)
    assert result.shape == (4, 1)
    assert result[0, 0] == 0.5  # 'a' appears 2/4 times
```

### Integration Tests

```python
# tests/test_pipeline.py
def test_full_pipeline():
    # Load data
    # Build pipeline
    # Train
    # Predict
    # Assert accuracy > threshold
```

### Property-Based Tests

```python
# tests/test_transformers.py
def test_transformer_shape_preservation():
    # Given any valid input
    # When transformed
    # Then n_rows unchanged
```

## Future Improvements

### Short Term

1. **Add more encoders**
   - Weight of Evidence (WOE)
   - Count encoding
   - Hashing

2. **Feature selection**
   - Automated feature selection based on importance
   - Remove correlated features

3. **More models**
   - LightGBM
   - CatBoost
   - Neural networks

### Medium Term

1. **Experiment tracking**
   - MLflow integration
   - Weights & Biases

2. **Model monitoring**
   - Drift detection in production
   - Performance degradation alerts

3. **Automated retraining**
   - Scheduled retraining
   - Trigger-based retraining

### Long Term

1. **Cloud deployment**
   - Docker containers
   - Kubernetes orchestration
   - API serving

2. **Real-time inference**
   - Streaming predictions
   - Low-latency serving

3. **AutoML integration**
   - Automated architecture search
   - Automated feature engineering

## Comparison to Original WXYVer Code

### Improvements

| Aspect | WXYVer | New Architecture |
|--------|--------|------------------|
| **Organization** | Single monolithic files | Modular structure |
| **Reusability** | Hard to reuse components | Easy to mix & match |
| **Testability** | Difficult to test | Each component testable |
| **Configuration** | Scattered parameters | Centralized config |
| **Documentation** | Inline comments | Comprehensive docs |
| **Extensibility** | Hard to extend | Plugin-like architecture |

### Preserved Features

- ✅ All feature engineering logic
- ✅ Target encoding with smoothing
- ✅ Statistical aggregations
- ✅ Wide feature builder
- ✅ Class weighting
- ✅ Cross-validation support

### Migration Path

Old code still works in `WXYVer/`. New code coexists:

```
AI1010Final/
├── WXYVer/           # Original code (unchanged)
│   ├── src/
│   └── models/
│
├── configs/          # New modular architecture
├── feature_engineering/
├── modeling/
├── training/
└── main.py
```

Users can:
1. Keep using WXYVer for experiments
2. Migrate gradually to new architecture
3. Compare results between both

---

**Questions or suggestions?** Feel free to extend this architecture!

