"""
Script to train the career guidance model
"""

import argparse
import logging
from pathlib import Path
from typing import Tuple

import pandas as pd
import yaml
from sklearn.preprocessing import LabelEncoder
import numpy as np

from src.config.config import Config
from src.data.preprocessing import DataPreprocessor
from src.models.gradient_boosting_model import GradientBoostingModel
from src.utils.logger import setup_logger

def parse_args():
    parser = argparse.ArgumentParser(description='Train career guidance model')
    parser.add_argument('--config', type=str, default='src/config/config.yaml',
                      help='Path to configuration file')
    parser.add_argument('--data', type=str, required=True,
                      help='Path to training data')
    parser.add_argument('--output', type=str, default='models',
                      help='Directory to save trained model')
    return parser.parse_args()

def preprocess_data(df: pd.DataFrame, config: Config, preprocessor: DataPreprocessor) -> Tuple[pd.DataFrame, np.ndarray]:
    """Preprocess features and targets"""
    # Preprocess features
    X = preprocessor.preprocess_features(df)
    
    # Handle categorical columns
    categorical_columns = ['al_stream', 'university_program']
    for col in categorical_columns:
        if col in X.columns:
            # Convert to categorical and encode
            X[col] = pd.Categorical(X[col]).codes
    
    # Remove target columns from features if present
    target_columns = config.data_config['target_columns']
    X = X.drop(columns=target_columns, errors='ignore')
    
    # Prepare target variables
    y_raw = df[target_columns].values
    
    # Initialize label encoders for each target
    label_encoders = {
        col: LabelEncoder() for col in target_columns
    }
    
    # Encode each target column
    y = np.zeros_like(y_raw, dtype=int)
    for i, col in enumerate(target_columns):
        # Convert any NaN values to 'Unknown' before encoding
        y_raw[:, i] = np.where(pd.isna(y_raw[:, i]), 'Unknown', y_raw[:, i])
        y[:, i] = label_encoders[col].fit_transform(y_raw[:, i])
    
    return X, y, label_encoders

def main():
    # Parse arguments
    args = parse_args()
    
    # Set up logging
    logger = setup_logger('train_model', 'logs/train_model.log')
    
    # Load configuration
    config = Config(args.config)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load data
        logger.info(f"Loading data from {args.data}")
        df = pd.read_csv(args.data)
        
        # Initialize preprocessor
        preprocessor = DataPreprocessor(config)
        
        # Preprocess data
        logger.info("Preprocessing features")
        X, y, label_encoders = preprocess_data(df, config, preprocessor)
        
        # Initialize and train model
        logger.info("Training model")
        model = GradientBoostingModel(config.model_config)
        model.train(X, y, feature_names=X.columns.tolist())
        
        # Save label encoders along with the model
        model.label_encoders = label_encoders
        
        # Evaluate model
        metrics = model.evaluate(X, y)
        logger.info("Model performance metrics:")
        for metric, value in metrics.items():
            logger.info(f"{metric}: {value:.4f}")
        
        # Get feature importance
        importance = model.get_feature_importance()
        logger.info("Top features by importance:")
        for target, scores in importance.items():
            logger.info(f"\nTarget: {target}")
            # Sort features by weight importance
            sorted_features = sorted(
                scores['weight'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
            for feature, score in sorted_features:
                logger.info(f"{feature}: {score:.4f}")
        
        # Save model
        model_path = output_dir / 'model.joblib'
        logger.info(f"Saving model to {model_path}")
        model.save(model_path)
        
        # Save preprocessor
        preprocessor_path = output_dir / 'preprocessor.joblib'
        logger.info(f"Saving preprocessor to {preprocessor_path}")
        preprocessor.save_preprocessors(preprocessor_path)
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}", exc_info=True)
        raise

if __name__ == '__main__':
    main()
