"""
Multi-label Gradient Boosting implementation for the Career Guidance System
"""

from typing import Dict, Any, List
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

from .base_model import BaseModel

class GradientBoostingModel(BaseModel):
    """Multi-label Gradient Boosting model implementation using XGBoost"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model = self._create_model()
        self.feature_names = None
        
    def _create_model(self) -> MultiOutputClassifier:
        """Create and configure the XGBoost model"""
        base_model = xgb.XGBClassifier(
            max_depth=self.config['params'].get('max_depth', 6),
            learning_rate=self.config['params'].get('learning_rate', 0.1),
            n_estimators=self.config['params'].get('n_estimators', 100),
            min_child_weight=self.config['params'].get('min_child_weight', 1),
            subsample=self.config['params'].get('subsample', 0.8),
            colsample_bytree=self.config['params'].get('colsample_bytree', 0.8),
            random_state=self.config['params'].get('random_state', 42),
            tree_method='hist',  # For faster training
            enable_categorical=True  # Native categorical support
        )
        return MultiOutputClassifier(base_model)
    
    def train(self, X: np.ndarray, y: np.ndarray, feature_names: List[str] = None) -> None:
        """
        Train the model
        
        Args:
            X: Training features
            y: Training labels (multi-label)
            feature_names: Optional list of feature names
        """
        self.feature_names = feature_names
        # Convert to DMatrix for faster training
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X = X.values
            
        self.model.fit(X, y)
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities
        
        Returns:
            Array of shape (n_samples, n_targets, n_classes)
        """
        return np.array([est.predict_proba(X) for est in self.model.estimators_])
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance
        
        Returns:
            Dictionary of evaluation metrics per target
        """
        y_pred = self.predict(X)
        
        metrics = {}
        for i, target in enumerate(self.config['target_columns']):
            metrics[f'{target}_accuracy'] = accuracy_score(y[:, i], y_pred[:, i])
            metrics[f'{target}_precision'] = precision_score(y[:, i], y_pred[:, i], average='weighted')
            metrics[f'{target}_recall'] = recall_score(y[:, i], y_pred[:, i], average='weighted')
            metrics[f'{target}_f1'] = f1_score(y[:, i], y_pred[:, i], average='weighted')
            
        # Add average metrics
        metrics['avg_accuracy'] = np.mean([v for k, v in metrics.items() if k.endswith('accuracy')])
        metrics['avg_f1'] = np.mean([v for k, v in metrics.items() if k.endswith('f1')])
        
        return metrics
    
    def get_feature_importance(self) -> Dict[str, Dict[str, float]]:
        """
        Get feature importance scores for each target
        
        Returns:
            Dictionary mapping targets to their feature importance scores
        """
        importance_dict = {}
        feature_names = self.feature_names or [f'f{i}' for i in range(self.model.estimators_[0].n_features_in_)]
        
        for i, estimator in enumerate(self.model.estimators_):
            target = self.config['target_columns'][i]
            # Get both weight and gain importance
            weight_importance = dict(zip(feature_names, estimator.feature_importances_))
            gain_importance = dict(zip(feature_names, 
                                     estimator.get_booster().get_score(importance_type='gain')))
            
            importance_dict[target] = {
                'weight': weight_importance,
                'gain': gain_importance
            }
            
        return importance_dict
    
    def save(self, path: str) -> None:
        """Save model to disk"""
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'config': self.config
        }
        joblib.dump(model_data, path)
    
    def load(self, path: str) -> None:
        """Load model from disk"""
        model_data = joblib.load(path)
        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        self.config = model_data['config']
        
    def get_early_stopping_rounds(self) -> int:
        """Get number of early stopping rounds based on n_estimators"""
        n_estimators = self.config['params'].get('n_estimators', 100)
        return max(int(n_estimators * 0.1), 10)  # 10% of n_estimators or at least 10 rounds
