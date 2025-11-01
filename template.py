#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🏢 OFFICE CATEGORY PREDICTION - BASELINE TEMPLATE

This script provides a simple baseline for the Office Classification challenge.

Your task: Improve upon this baseline by trying different approaches!

💡 Hints on What to Try:
1. Feature Engineering - Create interactions, polynomials, ratios
2. Different Models - Random Forest, XGBoost, Neural Networks
3. Hyperparameter Tuning - Optimize model parameters
4. Ensemble Methods - Combine multiple models
5. Handle Missing Values Better - Try different imputation strategies
6. Encode Categoricals Differently - One-hot encoding, target encoding

Good luck! 🚀
"""

# ============================================================================
# STEP 1: IMPORT LIBRARIES
# ============================================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# STEP 2: LOAD DATA
# ============================================================================

def load_data():
    """Load training data and separate features from target."""
    # Load training data
    train = pd.read_csv('datasets/office_train.csv')
    
    # Separate features and target
    X = train.drop('OfficeCategory', axis=1)
    y = train['OfficeCategory']
    
    print("Dataset loaded successfully!")
    print(f"Shape: {X.shape}")
    print(f"Target distribution:\n{y.value_counts().sort_index()}")
    
    # TODO: Explore the data here
    # - Check for missing values: X.isnull().sum()
    # - Look at feature distributions: X.describe()
    # - Visualize relationships: Use matplotlib/seaborn
    # - Understand which features matter most
    
    return X, y


# ============================================================================
# STEP 3: SIMPLE PREPROCESSING
# ============================================================================

def simple_preprocess(X_train, X_test=None):
    """
    Basic preprocessing: Fill missing values and encode categoricals
    
    TODO: Improve this function!
    Ideas:
    - Try different imputation strategies (mean, mode, KNN)
    - Create new features (interactions, ratios, polynomials)
    - Try one-hot encoding instead of label encoding
    - Handle outliers
    - Scale/normalize features
    """
    
    # Make copies
    X_train = X_train.copy()
    if X_test is not None:
        X_test = X_test.copy()
    
    # Identify feature types
    numeric_features = X_train.select_dtypes(include=[np.number]).columns
    categorical_features = X_train.select_dtypes(include=['object']).columns
    
    print(f"Numeric features: {len(numeric_features)}")
    print(f"Categorical features: {len(categorical_features)}")
    
    # Fill missing values - NUMERIC (median)
    for col in numeric_features:
        median_val = X_train[col].median()
        X_train[col] = X_train[col].fillna(median_val)
        if X_test is not None:
            X_test[col] = X_test[col].fillna(median_val)
    
    # Fill missing values - CATEGORICAL (mode)
    for col in categorical_features:
        mode_val = X_train[col].mode()[0] if len(X_train[col].mode()) > 0 else 'Missing'
        X_train[col] = X_train[col].fillna(mode_val)
        if X_test is not None:
            X_test[col] = X_test[col].fillna(mode_val)
    
    # Encode categorical features (label encoding)
    for col in categorical_features:
        le = LabelEncoder()
        if X_test is not None:
            # Fit on combined data
            combined = pd.concat([X_train[col].astype(str), X_test[col].astype(str)])
            le.fit(combined)
            X_train[col] = le.transform(X_train[col].astype(str))
            X_test[col] = le.transform(X_test[col].astype(str))
        else:
            X_train[col] = le.fit_transform(X_train[col].astype(str))
    
    # Final safety check
    X_train = X_train.fillna(0)
    if X_test is not None:
        X_test = X_test.fillna(0)
    
    if X_test is not None:
        return X_train, X_test
    return X_train


# ============================================================================
# STEP 4: TRAIN MODEL
# ============================================================================

def train_model(X_train, y_train, X_val, y_val):
    """
    Train baseline model (Logistic Regression)
    
    TODO: Try different models!
    - RandomForestClassifier
    - XGBClassifier
    - GradientBoostingClassifier
    - Neural Networks (MLPClassifier)
    - Ensemble methods (VotingClassifier, StackingClassifier)
    """
    
    print("\n" + "="*70)
    print("TRAINING BASELINE MODEL: LOGISTIC REGRESSION")
    print("="*70)
    
    # Scale features (important for logistic regression)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Train logistic regression
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    return model, scaler, X_train_scaled, X_val_scaled


# ============================================================================
# STEP 5: EVALUATE MODEL
# ============================================================================

def evaluate_model(model, X_train_scaled, y_train, X_val_scaled, y_val):
    """Evaluate model performance on train and validation sets."""
    
    # Predictions
    y_train_pred = model.predict(X_train_scaled)
    y_val_pred = model.predict(X_val_scaled)
    
    # Calculate accuracy
    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc = accuracy_score(y_val, y_val_pred)
    
    print(f"\nTrain Accuracy: {train_acc*100:.2f}%")
    print(f"Validation Accuracy: {val_acc*100:.2f}%")
    
    # Detailed classification report
    print("\nClassification Report (Validation):")
    print(classification_report(y_val, y_val_pred))
    
    # TODO: Add more evaluation metrics
    # - Confusion matrix
    # - Per-class accuracy
    # - Cross-validation scores
    # - Feature importance (for tree models)
    
    return val_acc


# ============================================================================
# STEP 6: MAKE PREDICTIONS ON TEST SET
# ============================================================================

def make_predictions(model, scaler, X_original):
    """Make predictions on test set and save to submission file."""
    
    print("\n" + "="*70)
    print("MAKING PREDICTIONS ON TEST SET")
    print("="*70)
    
    # Load test data
    test = pd.read_csv('datasets/office_test.csv')
    
    # Preprocess test data (use same preprocessing as training)
    X_test_processed = simple_preprocess(X_original, test)[1]
    
    # Scale test data
    X_test_scaled = scaler.transform(X_test_processed)
    
    # Make predictions
    test_predictions = model.predict(X_test_scaled)
    
    # Save predictions
    submission = pd.DataFrame({
        'Id': range(len(test_predictions)),
        'OfficeCategory': test_predictions
    })
    submission.to_csv('submission.csv', index=False)
    
    print("Predictions saved to submission.csv")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution pipeline."""
    
    # Load data
    X, y = load_data()
    
    # Apply preprocessing
    X_processed = simple_preprocess(X)
    
    print(f"\nAfter preprocessing:")
    print(f"Shape: {X_processed.shape}")
    print(f"Missing values: {X_processed.isnull().sum().sum()}")
    
    # Train-validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X_processed, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    
    print(f"\nTrain set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    
    # Train model
    model, scaler, X_train_scaled, X_val_scaled = train_model(
        X_train, y_train, X_val, y_val
    )
    
    # Evaluate model
    val_acc = evaluate_model(
        model, X_train_scaled, y_train, X_val_scaled, y_val
    )
    
    # Make predictions on test set
    make_predictions(model, scaler, X)
    
    print("\n" + "="*70)
    print("DONE! Check submission.csv for predictions.")
    print("="*70)


if __name__ == "__main__":
    main()


# ============================================================================
# 🎯 IDEAS TO TRY - IMPROVE YOUR MODEL!
# ============================================================================

"""
## 1. 🔧 Feature Engineering

Create new features that capture relationships between variables:

# Interaction features (Quality × Size effect)
X['Quality_Size'] = X['BuildingGrade'] * X['OfficeSpace']

# Polynomial features (Non-linear relationships)
X['OfficeSpace_squared'] = X['OfficeSpace'] ** 2
X['BuildingGrade_squared'] = X['BuildingGrade'] ** 2

# Ratio features (Relative measurements)
X['Space_Plot_Ratio'] = X['OfficeSpace'] / (X['PlotSize'] + 1)
X['Restroom_Meeting_Ratio'] = X['Restrooms'] / (X['MeetingRooms'] + 1)

# Aggregated features
X['TotalArea'] = X['OfficeSpace'] + X['BasementArea'] + X['ParkingArea']


## 2. 🌲 Different Models

Try tree-based models (often better than linear models for this type of data):

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Random Forest
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# XGBoost (usually best for tabular data)
model = XGBClassifier(
    n_estimators=200,
    max_depth=8,
    learning_rate=0.1,
    random_state=42
)
model.fit(X_train, y_train)


## 3. 🎛️ Hyperparameter Tuning

Optimize your model's parameters:

from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 15, 20, 25],
    'min_samples_split': [5, 10, 20]
}

# Grid search with cross-validation
grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
grid.fit(X_train, y_train)

# Best parameters
print(f"Best parameters: {grid.best_params_}")
print(f"Best CV score: {grid.best_score_:.4f}")


## 4. 🤝 Ensemble Methods

Combine multiple models for better predictions:

from sklearn.ensemble import VotingClassifier

# Create individual models
model1 = RandomForestClassifier(n_estimators=200, random_state=42)
model2 = XGBClassifier(n_estimators=200, random_state=42)

# Voting ensemble (combines predictions)
ensemble = VotingClassifier(
    estimators=[('rf', model1), ('xgb', model2)],
    voting='soft',  # Average probabilities
    weights=[1, 1.2]  # Slightly favor XGBoost
)
ensemble.fit(X_train, y_train)


## 5. 🎯 Feature Selection

Select only the most important features:

from sklearn.feature_selection import SelectKBest, f_classif

# Select top 50 features
selector = SelectKBest(f_classif, k=50)
X_train_selected = selector.fit_transform(X_train, y_train)
X_val_selected = selector.transform(X_val)

# Get selected feature names
selected_features = X_train.columns[selector.get_support()]
print(f"Selected features: {list(selected_features)}")


## 6. 📈 Cross-Validation

Get more reliable performance estimates:

from sklearn.model_selection import cross_val_score

# 5-fold cross-validation
scores = cross_val_score(
    model,
    X_train,
    y_train,
    cv=5,
    scoring='accuracy'
)

print(f"CV Accuracy: {scores.mean():.4f} (±{scores.std():.4f})")
"""

