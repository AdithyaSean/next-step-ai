"""
Multi-label Decision Tree implementation for the Career Guidance System
"""

from typing import Dict, Any, List
import numpy as np
import pandas as pd
from sklearn.multioutput import MultiOutputClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

from .base_model import BaseModel

class DecisionTreeModel(BaseModel):
    """Multi-label Decision Tree model implementation"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model = self._create_model()
        
    def _create_model(self) -> MultiOutputClassifier:
        """Create and configure the model"""
        base_model = DecisionTreeClassifier(
            max_depth=self.config['params']['max_depth'],
            min_samples_split=self.config['params']['min_samples_split'],
            min_samples_leaf=self.config['params']['min_samples_leaf'],
            random_state=self.config['params']['random_state']
        )
        return MultiOutputClassifier(base_model)
    
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the model"""
        self.model.fit(X, y)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities"""
        return np.array([est.predict_proba(X) for est in self.model.estimators_])
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance"""
        y_pred = self.predict(X)
        
        metrics = {}
        for i, target in enumerate(self.config['target_columns']):
            metrics[f'{target}_accuracy'] = accuracy_score(y[:, i], y_pred[:, i])
            metrics[f'{target}_precision'] = precision_score(y[:, i], y_pred[:, i], average='weighted')
            metrics[f'{target}_recall'] = recall_score(y[:, i], y_pred[:, i], average='weighted')
            metrics[f'{target}_f1'] = f1_score(y[:, i], y_pred[:, i], average='weighted')
            
        return metrics
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores"""
        importance_scores = {}
        
        for i, estimator in enumerate(self.model.estimators_):
            target = self.config['target_columns'][i]
            importance_scores[target] = dict(zip(
                self.config['feature_names'],
                estimator.feature_importances_
            ))
            
        return importance_scores
    
    def save(self, path: str) -> None:
        """Save model to disk"""
        joblib.dump(self.model, path)
    
    def load(self, path: str) -> None:
        """Load model from disk"""
        self.model = joblib.load(path)
