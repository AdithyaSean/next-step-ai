"""Train the career prediction model."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import json
import logging
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data.preprocessor import DataPreprocessor
from src.models.career_predictor import CareerPredictor
from src.data.generators.dataset_generator import StudentDataGenerator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_and_preprocess_data(n_samples: int = 10000) -> tuple:
    """Generate and preprocess training data.
    
    Args:
        n_samples: Number of samples to generate
        
    Returns:
        Tuple of (X, y) for training
    """
    logger.info(f"Generating {n_samples} training samples...")
    generator = StudentDataGenerator()
    data = generator.generate_dataset(n_samples)
    
    logger.info("Preprocessing data...")
    preprocessor = DataPreprocessor()
    preprocessor.fit(data)  # Fit first to ensure encoders are trained
    X, y = preprocessor.transform_with_target(data)  # Get features and target separately
    
    # Save preprocessor for inference
    preprocessor.save(project_root / 'models/saved/preprocessor')
    
    # Log feature information
    logger.info(f"Number of features: {len(preprocessor.get_feature_names())}")
    logger.info(f"Features: {preprocessor.get_feature_names()}")
    
    return X, y

def train_model(X: pd.DataFrame, y: pd.Series) -> tuple:
    """Train the career prediction model.
    
    Args:
        X: Feature matrix
        y: Target labels
        
    Returns:
        Tuple of (model, metrics)
    """
    logger.info("Splitting data into train/validation sets...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    logger.info("Training model...")
    model = CareerPredictor()
    metrics = model.train(X_train, y_train, X_val, y_val)
    
    # Generate predictions and evaluation metrics
    y_pred = model.predict(X_val)
    metrics['classification_report'] = classification_report(
        y_val, y_pred.argmax(axis=1), output_dict=True
    )
    
    # Generate SHAP explanations
    logger.info("Generating model explanations...")
    explanations = model.explain(X_val.iloc[:100])  # Sample for explanations
    
    return model, metrics, explanations

def save_artifacts(model: CareerPredictor, metrics: dict, 
                  explanations: dict, output_dir: Path):
    """Save model artifacts and metrics.
    
    Args:
        model: Trained model
        metrics: Training metrics
        explanations: Model explanations
        output_dir: Output directory
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_dir = output_dir / f'model_{timestamp}'
    
    logger.info(f"Saving model artifacts to {model_dir}...")
    model.save(model_dir)
    
    # Save metrics
    with open(model_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save SHAP values
    np.save(model_dir / 'shap_values.npy', explanations['shap_values'])
    
    logger.info("Model training complete!")

def main():
    """Main training pipeline."""
    output_dir = project_root / 'models/saved'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate and preprocess data
    X, y = generate_and_preprocess_data()
    
    # Train model
    model, metrics, explanations = train_model(X, y)
    
    # Save artifacts
    save_artifacts(model, metrics, explanations, output_dir)

if __name__ == '__main__':
    main()
