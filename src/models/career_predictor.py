"""Career prediction model using LightGBM."""

import lightgbm as lgb
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import joblib
from pathlib import Path
import shap

class CareerPredictor:
    """Predicts career paths based on student profiles using LightGBM."""
    
    def __init__(self):
        """Initialize the career predictor model."""
        self.model = None
        self.feature_names = None
        self.model_params = {
            'objective': 'multiclass',  # Set multiclass objective
            'metric': 'multi_logloss',  # Use multiclass log loss metric
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'verbose': -1
        }
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
             X_val: Optional[pd.DataFrame] = None,
             y_val: Optional[pd.Series] = None) -> Dict:
        """Train the model on student profile data.
        
        Args:
            X_train: Training features
            y_train: Training labels (career paths)
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            Dict containing training metrics
        """
        self.feature_names = X_train.columns.tolist()
        
        # Update model params with number of classes
        num_classes = len(np.unique(y_train))
        if num_classes <= 1:
            raise ValueError("Number of unique classes must be greater than 1")
            
        self.model_params.update({
            'num_class': num_classes,
            'verbose': -1  # Reduce verbosity
        })
        
        # Create datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data) if X_val is not None else None
        
        evals_result = {}  # Store evaluation results
        
        # Train model
        self.model = lgb.train(
            params=self.model_params,
            train_set=train_data,
            valid_sets=[train_data, val_data] if val_data else [train_data],
            valid_names=['train', 'valid'] if val_data else ['train'],
            num_boost_round=1000,
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.record_evaluation(evals_result),
                lgb.log_evaluation(period=100)  # Log every 100 iterations
            ]
        )
        
        # Get training metrics
        metrics = {
            'feature_importance': dict(zip(
                self.feature_names,
                self.model.feature_importance('gain')
            )),
            'num_classes': num_classes,
            'num_features': len(self.feature_names),
            'best_iteration': self.model.best_iteration if val_data else None,
            'train_metrics': evals_result.get('train', {}),
            'valid_metrics': evals_result.get('valid', {}) if val_data else None
        }
        
        return metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict career paths for student profiles.
        
        Args:
            X: Student profile features
            
        Returns:
            Array of predicted career path probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get probability distributions over career paths.
        
        Args:
            X: Student profile features
            
        Returns:
            Array of probability distributions over career paths
        """
        return self.predict(X)
    
    def explain(self, X: pd.DataFrame) -> Dict:
        """Generate SHAP explanations for predictions.
        
        Args:
            X: Student profiles to explain
            
        Returns:
            Dict containing SHAP values and summary plots
        """
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X)
        
        return {
            'shap_values': shap_values,
            'feature_names': self.feature_names
        }
    
    def save(self, model_dir: str):
        """Save the trained model and associated artifacts.
        
        Args:
            model_dir: Directory to save model artifacts
        """
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save LightGBM model
        self.model.save_model(str(model_dir / 'model.txt'))
        
        # Save feature names
        joblib.dump(self.feature_names, model_dir / 'feature_names.joblib')
        
        # Save model parameters
        joblib.dump(self.model_params, model_dir / 'model_params.joblib')
    
    @classmethod
    def load(cls, model_dir: str) -> 'CareerPredictor':
        """Load a trained model from disk.
        
        Args:
            model_dir: Directory containing model artifacts
            
        Returns:
            Loaded CareerPredictor instance
        """
        model_dir = Path(model_dir)
        
        # Load model parameters and create instance
        model_params = joblib.load(model_dir / 'model_params.joblib')
        predictor = cls()
        
        # Load feature names
        predictor.feature_names = joblib.load(model_dir / 'feature_names.joblib')
        
        # Load LightGBM model
        predictor.model = lgb.Booster(model_file=str(model_dir / 'model.txt'))
        
        return predictor
