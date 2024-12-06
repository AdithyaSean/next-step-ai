"""
Script to evaluate model performance
"""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from src.config.config import Config
from src.data.preprocessing import DataPreprocessor
from src.models.gradient_boosting_model import GradientBoostingModel
from src.utils.logger import setup_logger

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate model performance')
    parser.add_argument('--config', type=str, default='src/config/config.yaml',
                      help='Path to configuration file')
    parser.add_argument('--model', type=str, required=True,
                      help='Path to trained model')
    parser.add_argument('--test-data', type=str, required=True,
                      help='Path to test data')
    parser.add_argument('--output', type=str, default='evaluation',
                      help='Directory to save evaluation results')
    return parser.parse_args()

def plot_confusion_matrix(cm, labels, output_path):
    """Plot confusion matrix"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_feature_importance(importance_dict, output_path):
    """Plot feature importance"""
    plt.figure(figsize=(12, 6))
    
    for target, scores in importance_dict.items():
        # Sort features by importance
        sorted_features = sorted(
            scores['weight'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]  # Top 10 features
        
        features, values = zip(*sorted_features)
        
        plt.subplot(len(importance_dict), 1, 
                   list(importance_dict.keys()).index(target) + 1)
        sns.barplot(x=list(values), y=list(features))
        plt.title(f'Top Features for {target}')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def main():
    # Parse arguments
    args = parse_args()
    
    # Set up logging
    logger = setup_logger('evaluate', 'logs/evaluate.log')
    
    # Load configuration
    config = Config(args.config)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load model
        logger.info("Loading model")
        model = GradientBoostingModel(config.model_config)
        model.load(args.model)
        
        # Load test data
        logger.info("Loading test data")
        test_df = pd.read_csv(args.test_data)
        
        # Split features and targets
        X = test_df.drop(columns=config.data_config['target_columns'])
        y = test_df[config.data_config['target_columns']].values
        
        # Make predictions
        logger.info("Making predictions")
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)
        
        # Calculate metrics
        logger.info("Calculating metrics")
        results = {}
        
        for i, target in enumerate(config.data_config['target_columns']):
            # Get the label encoder for this target
            encoder = None
            if hasattr(model, 'label_encoders'):
                encoder = model.label_encoders.get(target) or model.label_encoders.get(f'target_{i}')
            
            # Convert string labels to numeric if needed
            y_true = y[:, i]
            y_pred = predictions[:, i]
            
            if encoder is not None and y_true.dtype == object:
                y_true = encoder.transform(y_true)
                if y_pred.dtype == object:
                    y_pred = encoder.transform(y_pred)
            
            target_results = {
                'classification_report': classification_report(
                    y_true,
                    y_pred,
                    output_dict=True
                ),
                'confusion_matrix': confusion_matrix(
                    y_true,
                    y_pred
                ).tolist()
            }
            results[target] = target_results
            
            # Plot confusion matrix
            plot_confusion_matrix(
                target_results['confusion_matrix'],
                [f'Class {i}' for i in range(len(np.unique(y_true)))],
                output_dir / f'{target}_confusion_matrix.png'
            )
        
        # Get and plot feature importance
        importance = model.get_feature_importance()
        plot_feature_importance(importance, output_dir / 'feature_importance.png')
        
        # Save results
        with open(output_dir / 'evaluation_results.json', 'w') as f:
            json.dump(results, f, indent=2)
            
        logger.info(f"Evaluation results saved to {output_dir}")
        
        # Print summary metrics
        logger.info("\nSummary Metrics:")
        for target, metrics in results.items():
            logger.info(f"\n{target}:")
            logger.info(f"Accuracy: {metrics['classification_report']['accuracy']:.4f}")
            logger.info(f"Macro F1: {metrics['classification_report']['macro avg']['f1-score']:.4f}")
        
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}", exc_info=True)
        raise

if __name__ == '__main__':
    main()
