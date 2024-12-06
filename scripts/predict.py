"""
Script to make career predictions using the trained model
"""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
import numpy as np

from src.config.config import Config
from src.data.preprocessing import DataPreprocessor
from src.models.gradient_boosting_model import GradientBoostingModel
from src.utils.logger import setup_logger

def parse_args():
    parser = argparse.ArgumentParser(description='Make career predictions')
    parser.add_argument('--config', type=str, default='src/config/config.yaml',
                      help='Path to configuration file')
    parser.add_argument('--model', type=str, required=True,
                      help='Path to trained model')
    parser.add_argument('--preprocessor', type=str, required=True,
                      help='Path to saved preprocessor')
    parser.add_argument('--input', type=str, required=True,
                      help='Path to input data (CSV or JSON)')
    parser.add_argument('--output', type=str, default='predictions.json',
                      help='Path to save predictions')
    return parser.parse_args()

def format_predictions(predictions: np.ndarray, probabilities: np.ndarray, 
                      target_columns: list) -> list:
    """Format predictions and probabilities into a readable format"""
    results = []
    
    for i in range(len(predictions)):
        result = {}
        for j, target in enumerate(target_columns):
            result[target] = {
                'prediction': predictions[i][j],
                'confidence': float(np.max(probabilities[j][i]))
            }
        results.append(result)
    
    return results

def main():
    # Parse arguments
    args = parse_args()
    
    # Set up logging
    logger = setup_logger('predict', 'logs/predict.log')
    
    # Load configuration
    config = Config(args.config)
    
    try:
        # Load model and preprocessor
        logger.info("Loading model and preprocessor")
        model = GradientBoostingModel(config.model_config)
        model.load(args.model)
        
        preprocessor = DataPreprocessor(config)
        preprocessor.load_preprocessors(args.preprocessor)
        
        # Load input data
        logger.info(f"Loading input data from {args.input}")
        if args.input.endswith('.json'):
            with open(args.input, 'r') as f:
                data = json.load(f)
            df = pd.DataFrame([data])
        else:
            df = pd.read_csv(args.input)
        
        # Preprocess input
        logger.info("Preprocessing input data")
        X = preprocessor.preprocess_features(df)
        
        # Make predictions
        logger.info("Making predictions")
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)
        
        # Format results
        results = format_predictions(
            predictions,
            probabilities,
            config.data_config['target_columns']
        )
        
        # Add feature importance for explanation
        importance = model.get_feature_importance()
        
        # Combine predictions with explanations
        output = {
            'predictions': results,
            'feature_importance': importance,
            'metadata': {
                'model_type': config.model_config['type'],
                'number_of_predictions': len(results)
            }
        }
        
        # Save predictions
        logger.info(f"Saving predictions to {args.output}")
        with open(args.output, 'w') as f:
            json.dump(output, f, indent=2)
            
        logger.info("Prediction completed successfully")
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}", exc_info=True)
        raise

if __name__ == '__main__':
    main()
