"""
Script to tune model hyperparameters using cross-validation
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
import xgboost as xgb

from src.config.config import Config
from src.data.preprocessing import DataPreprocessor
from src.utils.logger import setup_logger

def parse_args():
    parser = argparse.ArgumentParser(description='Tune model hyperparameters')
    parser.add_argument('--config', type=str, default='src/config/config.yaml',
                      help='Path to configuration file')
    parser.add_argument('--data', type=str, required=True,
                      help='Path to processed data')
    parser.add_argument('--output', type=str, default='models/best_params.json',
                      help='Path to save best parameters')
    parser.add_argument('--n-iter', type=int, default=20,
                      help='Number of parameter settings to try')
    return parser.parse_args()

def get_param_grid():
    """Define the hyperparameter search space"""
    return {
        'max_depth': [3, 4, 5, 6, 7, 8],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'n_estimators': [50, 100, 200],
        'min_child_weight': [1, 3, 5],
        'subsample': [0.6, 0.7, 0.8, 0.9],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
        'gamma': [0, 0.1, 0.2]
    }

def main():
    args = parse_args()
    logger = setup_logger('tune_hyperparameters', 'logs/tuning.log')
    
    try:
        # Load configuration and data
        config = Config(args.config)
        data = np.load(args.data)
        X, y = data['X'], data['y']
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=config.training_config['validation_size'],
            random_state=42
        )
        
        # Create base model
        base_model = xgb.XGBClassifier(
            tree_method='hist',
            enable_categorical=True,
            random_state=42
        )
        
        # Set up randomized search
        param_grid = get_param_grid()
        random_search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_grid,
            n_iter=args.n_iter,
            scoring='f1_weighted',
            cv=5,
            verbose=2,
            random_state=42,
            n_jobs=-1
        )
        
        # Perform search
        logger.info("Starting hyperparameter search...")
        random_search.fit(X_train, y_train)
        
        # Log results
        logger.info("Best parameters found:")
        for param, value in random_search.best_params_.items():
            logger.info(f"{param}: {value}")
        
        logger.info(f"Best cross-validation score: {random_search.best_score_:.4f}")
        
        # Evaluate on validation set
        val_score = random_search.score(X_val, y_val)
        logger.info(f"Validation set score: {val_score:.4f}")
        
        # Save best parameters
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(random_search.best_params_, f, indent=2)
            
        logger.info(f"Best parameters saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Error during hyperparameter tuning: {str(e)}", exc_info=True)
        raise

if __name__ == '__main__':
    main()
