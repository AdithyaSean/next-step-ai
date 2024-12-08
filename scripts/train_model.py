"""Train and evaluate the career prediction model."""

import sys
from pathlib import Path
import pandas as pd
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.models.career_predictor import CareerPredictor
from src.data.preprocessor import DataPreprocessor

def main():
    """Train and evaluate model."""
    # Setup paths
    data_dir = project_root / "data"
    processed_dir = data_dir / "processed"
    models_dir = data_dir / "models"
    
    # Load processed data
    print("Loading data...")
    data_path = processed_dir / "processed_profiles.csv"
    data = pd.read_csv(data_path)
    
    # Initialize and train model
    print("Training model...")
    predictor = CareerPredictor()
    metrics = predictor.train(data)
    
    # Print metrics
    print("\nModel Performance:")
    for metric, value in metrics.items():
        print(f"{metric.capitalize()}: {value:.4f}")
    
    # Get feature importance
    print("\nTop 10 Most Important Features:")
    importance = predictor.get_feature_importance()
    print(importance.head(10))
    
    # Save model
    print("\nSaving model...")
    model_path = models_dir / "career_predictor"
    predictor.save(model_path)
    
    print(f"\nModel saved to: {model_path}")

if __name__ == "__main__":
    main()
