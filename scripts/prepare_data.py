"""
Script to prepare and preprocess data for the career guidance model
"""

import argparse
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from src.config.config import Config
from src.data.preprocessing import DataPreprocessor
from src.utils.logger import setup_logger

def parse_args():
    parser = argparse.ArgumentParser(description='Prepare data for career guidance model')
    parser.add_argument('--config', type=str, default='src/config/config.yaml',
                      help='Path to configuration file')
    parser.add_argument('--input', type=str, required=True,
                      help='Path to raw data')
    parser.add_argument('--output', type=str, default='data/processed',
                      help='Directory to save processed data')
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()
    
    # Set up logging
    logger = setup_logger('prepare_data', 'logs/prepare_data.log')
    
    # Load configuration
    config = Config(args.config)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load data
        logger.info(f"Loading data from {args.input}")
        df = pd.read_csv(args.input)
        
        # Initialize preprocessor
        preprocessor = DataPreprocessor(config)
        
        # Basic data cleaning
        logger.info("Performing basic data cleaning")
        df = df.dropna(subset=config.data_config['target_columns'])  # Remove rows with missing targets
        
        # Split data into train and test sets
        logger.info("Splitting data into train and test sets")
        train_df, test_df = train_test_split(
            df,
            test_size=config.training_config['test_size'],
            random_state=42
        )
        
        # Process training data
        logger.info("Processing training data")
        train_processed = preprocessor.preprocess_features(train_df)
        
        # Process test data using same preprocessing parameters
        logger.info("Processing test data")
        test_processed = preprocessor.preprocess_features(test_df)
        
        # Save processed datasets
        logger.info("Saving processed datasets")
        train_processed.to_csv(output_dir / 'train.csv', index=False)
        test_processed.to_csv(output_dir / 'test.csv', index=False)
        
        # Save preprocessor for later use
        logger.info("Saving preprocessor")
        preprocessor.save_preprocessors(output_dir / 'preprocessor.joblib')
        
        # Generate and save data statistics
        logger.info("Generating data statistics")
        stats = {
            'n_train_samples': len(train_processed),
            'n_test_samples': len(test_processed),
            'n_features': train_processed.shape[1],
            'feature_names': train_processed.columns.tolist(),
            'target_distribution': {
                target: train_df[target].value_counts().to_dict()
                for target in config.data_config['target_columns']
            }
        }
        
        # Save statistics
        import json
        with open(output_dir / 'data_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)
            
        logger.info("Data preparation completed successfully")
        
    except Exception as e:
        logger.error(f"Error during data preparation: {str(e)}", exc_info=True)
        raise

if __name__ == '__main__':
    main()
