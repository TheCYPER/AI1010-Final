# Office Category Prediction - ML Pipeline

A modular, production-ready machine learning pipeline for office category classification.

## 📁 Project Structure

```
AI1010Final/
├── configs/                    # Configuration management
│   ├── __init__.py
│   └── config.py              # Centralized configuration
│
├── data_cleaning/             # Data cleaning utilities
│   ├── __init__.py
│   ├── column_types.py        # Column type inference
│   ├── missing_handler.py     # Missing value handling
│   └── outlier_handler.py     # Outlier detection & handling
│
├── data_exploration/          # EDA and feature auditing
│   ├── __init__.py
│   ├── exploratory_analysis.py # Basic EDA
│   └── feature_audit.py       # Feature importance & drift analysis
│
├── feature_engineering/       # Feature engineering modules
│   ├── __init__.py
│   ├── encoders.py           # Frequency & target encoding
│   ├── wide_features.py      # Derived feature builder
│   ├── statistical_features.py # Statistical aggregations
│   ├── transformers.py       # Log transforms, etc.
│   └── preprocessor.py       # Main preprocessor assembly
│
├── modeling/                  # Model definitions
│   ├── __init__.py
│   ├── base_model.py         # Abstract base class
│   ├── xgboost_model.py      # XGBoost wrapper
│   └── ensemble.py           # Ensemble methods
│
├── training/                  # Training logic
│   ├── __init__.py
│   ├── trainer.py            # Single split trainer
│   └── cross_validator.py   # K-fold cross-validation
│
├── hyperparameter_tuning/    # Hyperparameter optimization
│   ├── __init__.py
│   └── tuner.py             # Optuna-based tuning
│
├── utils/                    # Utility functions
│   ├── __init__.py
│   ├── logger.py            # Logging utilities
│   └── metrics.py           # Evaluation metrics
│
├── datasets/                 # Data directory
│   ├── office_train.csv
│   └── office_test.csv
│
├── outputs/                  # Output directory (created automatically)
│   ├── models/              # Trained models
│   ├── metrics/             # Evaluation metrics
│   ├── predictions/         # Test predictions
│   └── logs/               # Log files
│
├── main.py                  # Main entry point
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Exploratory Data Analysis

```bash
python main.py --mode eda
```

This will:
- Analyze missing values
- Show target distribution
- Compute cardinality
- Generate statistics
- Save report to `outputs/eda_report.json`

### 3. Train Model (Single Split)

```bash
python main.py --mode train
```

This will:
- Load and split data (80/20)
- Build preprocessing pipeline
- Train XGBoost model
- Evaluate on validation set
- Save model to `outputs/models/pipeline.joblib`
- Save metrics to `outputs/metrics/metrics.json`

### 4. Cross-Validation Training

```bash
python main.py --mode cv
```

This will:
- Perform 5-fold stratified cross-validation
- Train model on each fold
- Aggregate results with mean ± std
- Save summary to `outputs/models/cv/cv_summary.json`

### 5. Hyperparameter Tuning

```bash
python main.py --mode tune
```

This will:
- Use Optuna for Bayesian optimization
- Run 100 trials (configurable)
- Save best parameters to `outputs/tuning_results.json`

### 6. Feature Audit

```bash
python main.py --mode audit
```

This will:
- Compute feature importance
- Run permutation importance
- Check for train/val drift (adversarial validation)
- Identify highly correlated features
- Save report to `outputs/feature_audit.json`

### 7. Make Predictions

```bash
python main.py --mode predict --model_path outputs/models/pipeline.joblib
```

This will:
- Load trained model
- Make predictions on test set
- Save to `outputs/predictions/submission.csv`

## 🔧 Configuration

All configuration is centralized in `configs/config.py`. Key sections:

### Paths
```python
train_csv = "datasets/office_train.csv"
test_csv = "datasets/office_test.csv"
output_dir = "outputs"
```

### Model Parameters
```python
xgb_params = {
    'n_estimators': 1500,
    'learning_rate': 0.06,
    'max_depth': 4,
    'subsample': 0.75,
    'colsample_bytree': 0.55,
    'reg_lambda': 10.0,
    'reg_alpha': 3.0,
    ...
}
```

### Feature Engineering
```python
freq_encoding_cols = ['RoofType', 'ExteriorCovering1', 'FoundationType']
target_encoding_cols = ['ZoningClassification', 'BuildingType', ...]
```

### Training
```python
test_size = 0.2
n_splits = 5
use_class_weights = True
use_early_stopping = True
```

## 🧪 Feature Engineering Pipeline

The pipeline includes:

1. **Missing Value Handling**
   - Median imputation for numeric features
   - Constant imputation for categorical features
   - Missing indicators

2. **Encoding**
   - Frequency encoding for high-cardinality features
   - Target encoding with Laplace smoothing
   - One-hot encoding for low-cardinality features

3. **Wide Features** (40+ derived features)
   - Age features: BuildingAge, YearsSinceRenovation
   - Area features: TotalLivingArea, TotalBasementArea
   - Ratio features: PlotCoverage, RoomDensity, etc.
   - Quality combinations: OverallQuality, ExteriorScore
   - Temporal features: SeasonListed, BuildingLifeStage
   - Interaction features: QualityAreaProximity
   - Domain knowledge: RoomSizeAdequacy, ParkingAdequacy

4. **Statistical Aggregations**
   - Group-level z-scores
   - Relative shifts from group mean

5. **Transformations**
   - Log1p for skewed features (PlotSize)

## 📊 Model Performance

The pipeline is optimized for accuracy with:
- Stratified sampling
- Class weighting for imbalanced data
- Early stopping to prevent overfitting
- Comprehensive regularization (L1/L2)

Expected validation accuracy: **~75-80%** (depending on feature selection and tuning)

## 🔬 Advanced Usage

### Custom Configuration

Create a custom config and pass it:

```python
from configs import Config

config = Config()
config.models.xgb_params['n_estimators'] = 2000
config.training.n_splits = 10

# Use in your code
from training import Trainer
trainer = Trainer(config)
trainer.run()
```

### Programmatic API

You can also use the modules programmatically:

```python
from configs import Config
from training import Trainer
from modeling import XGBoostModel

# Setup
config = Config()
trainer = Trainer(config)

# Load data
X, y = trainer.load_data()
X_train, X_val, y_train, y_val = trainer.split_data(X, y)

# Build preprocessor
trainer.build_preprocessor(X_train)

# Train
model = XGBoostModel(config=config.models.xgb_params)
results = trainer.train(model, X_train, y_train, X_val, y_val)
```

### Adding New Features

Extend `WideFeatureBuilder` in `feature_engineering/wide_features.py`:

```python
def _add_custom_features(self, df: pd.DataFrame, out: Dict[str, Any]):
    """Add your custom features."""
    # Example: Interaction between two features
    out["CustomFeature"] = df["Feature1"] * df["Feature2"]
```

## 📝 Development Notes

### Design Principles

1. **Modularity**: Each component is self-contained and reusable
2. **Extensibility**: Easy to add new features, models, or strategies
3. **Configurability**: Centralized configuration for easy experimentation
4. **Sklearn Compatibility**: All transformers follow sklearn API
5. **Production Ready**: Proper logging, error handling, serialization

### Testing

```bash
# Run tests (if you add them)
pytest tests/

# Test individual modules
python -c "from configs import Config; print(Config())"
```

### Adding New Models

1. Create model class in `modeling/` inheriting from `BaseModel`
2. Implement `build_model()`, `fit()`, `predict()` methods
3. Update `main.py` to support new model

Example:

```python
from modeling import BaseModel

class MyCustomModel(BaseModel):
    def build_model(self, **kwargs):
        # Your model initialization
        pass
    
    def fit(self, X, y, **kwargs):
        # Training logic
        pass
    
    def predict(self, X):
        # Prediction logic
        pass
```

## 🐛 Troubleshooting

### Common Issues

1. **Import errors**: Ensure virtual environment is activated
2. **Memory issues**: Reduce `n_estimators` or use sampling
3. **Optuna not found**: Install with `pip install optuna`

### Performance Tips

- Use `n_jobs=-1` for parallel processing
- Enable early stopping to save time
- Start with fewer CV folds (e.g., 3) during development
- Use hyperparameter tuning sparingly (time-consuming)

## 📚 References

- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Scikit-learn](https://scikit-learn.org/)
- [Optuna](https://optuna.org/)

## 🤝 Contributing

Feel free to:
- Add new feature engineering techniques
- Implement additional models
- Improve hyperparameter search spaces
- Add visualization utilities
- Write tests

## 📄 License

This project is for educational purposes.

---

**Happy Modeling! 🎉**

