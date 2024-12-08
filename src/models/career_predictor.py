"""Career prediction model using LightGBM."""

import lightgbm as lgb
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

class CareerPredictor:
    """Predict career paths based on student profiles."""
    
    def __init__(self):
        """Initialize the career predictor."""
        self.model = None
        self.feature_names = None
        self.target_encoder = None
        self.is_fitted = False
    
    def _prepare_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """Prepare data for training or prediction."""
        # Remove target if present
        X = data.copy()
        y = None
        
        if 'career_path' in X.columns:
            y = X.pop('career_path')
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        return X, y
    
    def train(self, data: pd.DataFrame, test_size: float = 0.2, 
             random_state: int = 42) -> Dict[str, float]:
        """Train the career prediction model."""
        # Prepare data
        X, y = self._prepare_data(data)
        if y is None:
            raise ValueError("Training data must include 'career_path' column")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Create dataset
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
        
        # Set parameters
        params = {
            'objective': 'multiclass',
            'num_class': len(y.unique()),
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9
        }
        
        # Train model
        self.model = lgb.train(
            params,
            train_data,
            valid_sets=[valid_data],
            num_boost_round=100,
            early_stopping_rounds=10,
            verbose_eval=False
        )
        
        # Make predictions on test set
        y_pred = self.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='weighted'
        )
        
        self.is_fitted = True
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make career predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")
        
        # Ensure correct features
        missing_features = set(self.feature_names) - set(X.columns)
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")
        
        # Reorder columns to match training data
        X = X[self.feature_names]
        
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities."""
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")
        
        # Ensure correct features
        missing_features = set(self.feature_names) - set(X.columns)
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")
        
        # Reorder columns to match training data
        X = X[self.feature_names]
        
        return self.model.predict(X, pred_contrib=True)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance scores."""
        if not self.is_fitted:
            raise ValueError("Model must be trained first")
        
        importance = self.model.feature_importance(importance_type='gain')
        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
    
    def save(self, path: str):
        """Save the trained model."""
        if not self.is_fitted:
            raise ValueError("Model must be trained before saving")
        
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = save_path / 'model.txt'
        self.model.save_model(str(model_path))
        
        # Save feature names
        features_path = save_path / 'features.joblib'
        joblib.dump(self.feature_names, features_path)
    
    @classmethod
    def load(cls, path: str) -> 'CareerPredictor':
        """Load a trained model."""
        load_path = Path(path)
        if not load_path.exists():
            raise ValueError(f"Path {path} does not exist")
        
        predictor = cls()
        
        # Load model
        model_path = load_path / 'model.txt'
        predictor.model = lgb.Booster(model_file=str(model_path))
        
        # Load feature names
        features_path = load_path / 'features.joblib'
        predictor.feature_names = joblib.load(features_path)
        
        predictor.is_fitted = True
        return predictor
