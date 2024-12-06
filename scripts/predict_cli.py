"""
Command-line interface for career predictions
"""

import argparse
import json
from pathlib import Path

from src.models.predictor import CareerPredictor
from src.utils.logger import setup_logger

def parse_args():
    parser = argparse.ArgumentParser(description='Make career predictions')
    parser.add_argument('--model', type=str, required=True,
                      help='Path to trained model')
    parser.add_argument('--input', type=str, required=True,
                      help='Path to input data CSV')
    parser.add_argument('--output', type=str, default='predictions.json',
                      help='Path to save predictions')
    return parser.parse_args()

def main():
    args = parse_args()
    logger = setup_logger('predict_cli', 'logs/predict.log')
    
    try:
        # Initialize predictor
        predictor = CareerPredictor(args.model)
        
        # Load input data
        import pandas as pd
        data = pd.read_csv(args.input)
        
        # Make predictions
        predictions = predictor.predict_careers(data)
        
        # Save predictions
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(predictions, f, indent=2)
            
        logger.info(f"Predictions saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        raise

if __name__ == '__main__':
    main()
