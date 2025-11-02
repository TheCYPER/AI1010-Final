# Quick Start Guide

Get up and running in 5 minutes! 🚀

## Prerequisites

- Python 3.8+
- Virtual environment (recommended)

## Installation

```bash
# 1. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
```

## Your First Training Run

### Option 1: Quick Train (Recommended for First Time)

```bash
# Train with single split (fastest)
python main.py --mode train
```

**Expected output:**
```
======================================================================
STARTING TRAINING
======================================================================
Loading data from datasets/office_train.csv
Data loaded: (35001, 79)
Train set: (28000, 79)
Validation set: (7001, 79)
...
Train Accuracy: 0.8234
Val Accuracy:   0.7456
======================================================================
TRAINING COMPLETE
======================================================================
```

**Outputs:**
- `outputs/models/pipeline.joblib` - Trained model
- `outputs/metrics/metrics.json` - Performance metrics
- `outputs/logs/main.log` - Detailed logs

### Option 2: Cross-Validation (More Robust)

```bash
# 5-fold cross-validation
python main.py --mode cv
```

**Expected output:**
```
======================================================================
STARTING CROSS-VALIDATION
======================================================================
Using 5-fold cross-validation
...
[fold 1] acc=0.7456
[fold 2] acc=0.7489
[fold 3] acc=0.7423
[fold 4] acc=0.7501
[fold 5] acc=0.7467
...
Final Results:
  Accuracy: 0.7467 ± 0.0028
======================================================================
```

**Outputs:**
- `outputs/models/cv/cv_summary.json` - Aggregated results

## Making Predictions

After training, predict on test set:

```bash
python main.py --mode predict
```

**Expected output:**
```
Loading model from outputs/models/pipeline.joblib
Loading test data from datasets/office_test.csv
Making predictions...
Predictions saved to outputs/predictions/submission.csv
Number of predictions: 15001
```

**Submission file format:**
```csv
Id,OfficeCategory
0,2
1,1
2,3
...
```

## Exploring Your Data

Before training, run EDA:

```bash
python main.py --mode eda
```

This generates:
- Missing value analysis
- Target distribution
- Feature cardinality
- Numeric feature statistics

**Output:** `outputs/eda_report.json`

## Common Workflows

### Workflow 1: Quick Iteration

```bash
# 1. Explore data
python main.py --mode eda

# 2. Train quickly
python main.py --mode train

# 3. Make predictions
python main.py --mode predict
```

### Workflow 2: Robust Evaluation

```bash
# 1. Cross-validate
python main.py --mode cv

# 2. Feature audit
python main.py --mode audit

# 3. Retrain on all data (edit config)
python main.py --mode train
```

### Workflow 3: Hyperparameter Tuning

```bash
# 1. Quick baseline
python main.py --mode train

# 2. Tune hyperparameters (slow!)
python main.py --mode tune

# 3. Update config with best params

# 4. Retrain with best params
python main.py --mode train
```

## Customizing Your Pipeline

### Adjust Model Parameters

Edit `configs/config.py`:

```python
@dataclass
class ModelConfig:
    xgb_params: Dict[str, Any] = field(default_factory=lambda: {
        'n_estimators': 2000,      # Increase trees
        'learning_rate': 0.05,     # Reduce learning rate
        'max_depth': 5,            # Deeper trees
        ...
    })
```

Then retrain:

```bash
python main.py --mode train
```

### Add Custom Features

Edit `feature_engineering/wide_features.py`:

```python
def _add_custom_features(self, df: pd.DataFrame, out: Dict[str, Any]):
    """Add your features here."""
    out["MyFeature"] = df["Col1"] * df["Col2"]
```

Add to `transform()`:

```python
def transform(self, X: pd.DataFrame):
    ...
    self._add_custom_features(df, out)  # Add this line
    ...
```

### Change Train/Val Split

Edit `configs/config.py`:

```python
@dataclass
class TrainingConfig:
    test_size: float = 0.3  # Use 30% for validation
    ...
```

## Troubleshooting

### Problem: Import Error

```
ModuleNotFoundError: No module named 'xgboost'
```

**Solution:**
```bash
pip install -r requirements.txt
```

### Problem: Out of Memory

**Solution:** Reduce features or use sampling:

```python
# In main.py or your script
X_sample = X.sample(n=10000, random_state=42)
```

### Problem: Low Accuracy

**Checklist:**
- [ ] Did you run EDA to understand data?
- [ ] Are there many missing values?
- [ ] Is target distribution balanced?
- [ ] Try tuning hyperparameters
- [ ] Check feature importance (run `--mode audit`)

### Problem: Training Too Slow

**Solutions:**
1. Reduce `n_estimators` in config
2. Enable early stopping (already default)
3. Use single split instead of CV for iteration
4. Reduce hyperparameter tuning trials

## Next Steps

### Learn More

- Read `README.md` for detailed documentation
- Check `ARCHITECTURE.md` for design details
- Explore `configs/config.py` for all options

### Improve Your Model

1. **Feature Engineering**
   - Add domain-specific features
   - Create interaction terms
   - Try polynomial features

2. **Model Selection**
   - Try ensemble models
   - Experiment with different algorithms
   - Stack multiple models

3. **Hyperparameter Tuning**
   - Run `--mode tune` with more trials
   - Use cross-validation for evaluation
   - Document best parameters

4. **Feature Selection**
   - Run feature audit
   - Remove low-importance features
   - Test impact on performance

### Advanced Usage

**Custom training script:**

```python
from configs import Config
from training import Trainer
from modeling import XGBoostModel

config = Config()

# Customize config
config.training.n_splits = 10
config.models.xgb_params['n_estimators'] = 3000

# Train
trainer = Trainer(config)
results = trainer.run()

print(f"Accuracy: {results['val_metrics']['accuracy']:.4f}")
```

**Ensemble multiple models:**

```python
from modeling import EnsembleModel, create_xgb_rf_ensemble

ensemble = create_xgb_rf_ensemble(
    xgb_params=config.models.xgb_params,
    rf_params=config.models.rf_params
)

ensemble.fit(X_train, y_train)
predictions = ensemble.predict(X_test)
```

## Tips & Tricks

### Speed Up Iteration

```bash
# Use a subset of data during development
head -n 5000 datasets/office_train.csv > datasets/office_train_small.csv

# Update config to use smaller file
# Then iterate faster
```

### Save Experiment Results

```bash
# Add timestamp to outputs
python main.py --mode train
mv outputs/models/pipeline.joblib outputs/models/pipeline_v1.joblib

# Compare different configs
```

### Batch Predictions

```python
import pandas as pd
import joblib

pipeline = joblib.load('outputs/models/pipeline.joblib')

# Predict on multiple files
for test_file in ['test1.csv', 'test2.csv']:
    X = pd.read_csv(test_file)
    preds = pipeline.predict(X)
    pd.DataFrame({'pred': preds}).to_csv(f'preds_{test_file}')
```

## Getting Help

1. Check logs: `outputs/logs/main.log`
2. Review error messages
3. Read architecture docs
4. Inspect intermediate outputs

---

**Happy coding! 🎉**

Need more help? Check the full README or architecture docs!

