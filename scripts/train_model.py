"""
Script to train the career guidance model
"""

import argparse
import logging
from pathlib import Path

import pandas as pd
import yaml

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
        
        # Preprocess features
        logger.info("Preprocessing features")
        X = preprocessor.preprocess_features(df)
        
        # Prepare target variables
        target_columns = config.data_config['target_columns']
        y = df[target_columns].values
        
        # Initialize and train model
        logger.info("Training model")
        model = GradientBoostingModel(config.model_config)
        model.train(X, y, feature_names=X.columns.tolist())
        
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
