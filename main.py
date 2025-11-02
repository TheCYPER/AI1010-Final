#!/usr/bin/env python3
"""
Main entry point for the ML pipeline.

This script provides a clean interface for running different pipeline tasks:
- Data exploration
- Model training (single split or cross-validation)
- Hyperparameter tuning
- Prediction on test set

Usage examples:
    # Run EDA
    python main.py --mode eda
    
    # Train model with single train/val split
    python main.py --mode train
    
    # Train with cross-validation
    python main.py --mode cv
    
    # Hyperparameter tuning
    python main.py --mode tune
    
    # Make predictions on test set
    python main.py --mode predict --model_path outputs/models/pipeline.joblib
"""

import argparse
import sys
from pathlib import Path

from configs import Config
from utils.logger import setup_logger

# Setup logger
logger = setup_logger(
    'main',
    log_file='outputs/logs/main.log',
    level='INFO'
)


def run_eda(config: Config):
    """
    Run exploratory data analysis.
    
    Args:
        config: Configuration object
    """
    from data_exploration import ExploratoryAnalysis
    
    logger.info("Running Exploratory Data Analysis...")
    
    eda = ExploratoryAnalysis(config)
    eda.load_data()
    report = eda.generate_report(
        save_path=f"{config.paths.output_dir}/eda_report.json"
    )
    
    logger.info("EDA complete!")


def run_train(config: Config):
    """
    Run training with single train/validation split.
    
    Args:
        config: Configuration object
    """
    from training import Trainer
    
    logger.info("Starting training (single split)...")
    
    trainer = Trainer(config)
    results = trainer.run()
    
    logger.info(f"Training complete! Val Accuracy: {results['val_metrics']['accuracy']:.4f}")


def run_cross_validation(config: Config):
    """
    Run cross-validation training.
    
    Args:
        config: Configuration object
    """
    from training import CrossValidator
    
    logger.info("Starting cross-validation training...")
    
    cv = CrossValidator(config)
    results = cv.run(save_per_fold=False)
    
    logger.info(
        f"CV complete! Accuracy: {results['accuracy_mean']:.4f} ± {results['accuracy_std']:.4f}"
    )


def run_hyperparameter_tuning(config: Config):
    """
    Run hyperparameter tuning.
    
    Args:
        config: Configuration object
    """
    from hyperparameter_tuning import OptunaTuner
    from training import Trainer
    from modeling import XGBoostModel
    
    logger.info("Starting hyperparameter tuning...")
    
    # Load and prepare data
    trainer = Trainer(config)
    X, y = trainer.load_data()
    X_train, X_val, y_train, y_val = trainer.split_data(X, y)
    
    # Build and fit preprocessor
    trainer.build_preprocessor(X_train)
    X_train_transformed = trainer.preprocessor_.fit_transform(X_train, y_train)
    
    # Run tuning
    tuner = OptunaTuner(config)
    
    def model_builder(params):
        return XGBoostModel(config=params)
    
    results = tuner.tune(
        X_train,
        y_train,
        trainer.preprocessor_,
        model_builder
    )
    
    # Save results
    tuner.save_results()
    
    logger.info("Hyperparameter tuning complete!")
    logger.info(f"Best score: {results['best_score']:.4f}")


def run_prediction(config: Config, model_path: str, output_path: str = None):
    """
    Make predictions on test set.
    
    Args:
        config: Configuration object
        model_path: Path to trained pipeline
        output_path: Path to save predictions
    """
    import joblib
    import pandas as pd
    import numpy as np
    
    logger.info(f"Loading model from {model_path}...")
    pipeline = joblib.load(model_path)
    
    logger.info(f"Loading test data from {config.paths.test_csv}...")
    test_df = pd.read_csv(config.paths.test_csv)
    
    logger.info("Making predictions...")
    predictions = pipeline.predict(test_df)
    
    # Create submission
    submission = pd.DataFrame({
        'Id': range(len(predictions)),
        'OfficeCategory': predictions
    })
    
    if output_path is None:
        output_path = f"{config.paths.predictions_dir}/submission.csv"
    
    submission.to_csv(output_path, index=False)
    logger.info(f"Predictions saved to {output_path}")
    logger.info(f"Number of predictions: {len(predictions)}")
    logger.info(f"Prediction distribution:\n{pd.Series(predictions).value_counts().sort_index()}")


def run_feature_audit(config: Config):
    """
    Run feature importance audit on trained model.
    
    Args:
        config: Configuration object
    """
    import joblib
    import pandas as pd
    from data_exploration import FeatureAuditor
    from training import Trainer
    
    logger.info("Running feature audit...")
    
    # Load model
    model_path = f"{config.paths.models_dir}/pipeline.joblib"
    if not Path(model_path).exists():
        logger.error(f"Model not found at {model_path}. Train a model first.")
        return
    
    pipeline = joblib.load(model_path)
    
    # Load and prepare data
    trainer = Trainer(config)
    X, y = trainer.load_data()
    X_train, X_val, y_train, y_val = trainer.split_data(X, y)
    
    # Transform
    X_train_transformed = pipeline.named_steps['preprocessor'].fit_transform(X_train, y_train)
    X_val_transformed = pipeline.named_steps['preprocessor'].transform(X_val)
    
    # Audit
    auditor = FeatureAuditor(config)
    
    # Feature importance
    auditor.compute_feature_importance(pipeline.named_steps['model'])
    
    # Permutation importance
    auditor.compute_permutation_importance(
        pipeline.named_steps['model'],
        X_val_transformed,
        y_val
    )
    
    # Adversarial validation
    auditor.adversarial_validation(X_train_transformed, X_val_transformed)
    
    # Correlation
    auditor.check_correlation(pd.DataFrame(X_train_transformed))
    
    # Save report
    auditor.save_audit_report()
    
    logger.info("Feature audit complete!")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Office Category Prediction ML Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=['eda', 'train', 'cv', 'tune', 'predict', 'audit'],
        help='Pipeline mode to run'
    )
    
    parser.add_argument(
        '--model_path',
        type=str,
        default=None,
        help='Path to trained model (for predict mode)'
    )
    
    parser.add_argument(
        '--output_path',
        type=str,
        default=None,
        help='Path to save predictions (for predict mode)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to custom config file (optional)'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        logger.info(f"Loading config from {args.config}")
        # TODO: Implement config loading from file
        config = Config()
    else:
        config = Config()
    
    logger.info("="*70)
    logger.info(f"Running in mode: {args.mode.upper()}")
    logger.info("="*70)
    
    try:
        # Route to appropriate function
        if args.mode == 'eda':
            run_eda(config)
        
        elif args.mode == 'train':
            run_train(config)
        
        elif args.mode == 'cv':
            run_cross_validation(config)
        
        elif args.mode == 'tune':
            run_hyperparameter_tuning(config)
        
        elif args.mode == 'predict':
            if args.model_path is None:
                args.model_path = f"{config.paths.models_dir}/pipeline.joblib"
            
            if not Path(args.model_path).exists():
                logger.error(f"Model not found at {args.model_path}")
                sys.exit(1)
            
            run_prediction(config, args.model_path, args.output_path)
        
        elif args.mode == 'audit':
            run_feature_audit(config)
        
        logger.info("="*70)
        logger.info("✓ SUCCESS")
        logger.info("="*70)
    
    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

