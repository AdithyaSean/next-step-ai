"""
Career Guidance Model implementation using XGBoost
"""

from typing import Dict, Any, Tuple, List
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.multioutput import MultiOutputClassifier

from .base_model import BaseModel

class CareerGuidanceModel(BaseModel):
    """XGBoost-based model for career and education path prediction"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._init_model()
    
    def _init_model(self):
        """Initialize XGBoost model with configuration parameters"""
        base_model = xgb.XGBClassifier(
            max_depth=self.config['model']['params']['max_depth'],
            learning_rate=self.config['model']['params']['learning_rate'],
            n_estimators=self.config['model']['params']['n_estimators'],
            min_child_weight=self.config['model']['params']['min_child_weight'],
            subsample=self.config['model']['params']['subsample'],
            colsample_bytree=self.config['model']['params']['colsample_bytree'],
            random_state=self.config['model']['params']['random_state']
        )
        # Use MultiOutputClassifier to handle both career and education predictions
        self.model = MultiOutputClassifier(base_model)
    
    def train(self, X: pd.DataFrame, career_targets: np.ndarray, education_targets: np.ndarray) -> None:
        """
        Train the model on the given data
        
        Args:
            X: Training features DataFrame
            career_targets: Career path target labels
            education_targets: Education path target labels
        """
        # Combine targets for multi-output classification
        y = np.column_stack([career_targets, education_targets])
        self.model.fit(X, y)
    
    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions for the given input
        
        Args:
            X: Input features DataFrame
            
        Returns:
            Tuple of (career_predictions, education_predictions)
        """
        predictions = self.model.predict(X)
        return predictions[:, 0], predictions[:, 1]
    
    def predict_proba(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get prediction probabilities for the given input
        
        Args:
            X: Input features DataFrame
            
        Returns:
            Tuple of (career_probabilities, education_probabilities)
        """
        probabilities = self.model.predict_proba(X)
        # Each element in probabilities is a list of probability arrays for each target
        return probabilities[0], probabilities[1]
    
    def get_feature_importance(self) -> Dict[str, List[float]]:
        """
        Get feature importance scores from the model
        
        Returns:
            Dictionary with feature importance scores for career and education predictions
        """
        importances = {
            'career': [],
            'education': []
        }
        
        # Get feature importances from each sub-model
        for idx, estimator in enumerate(self.model.estimators_):
            target = 'career' if idx == 0 else 'education'
            importances[target] = estimator.feature_importances_.tolist()
        
        return importances
    
    def fit(self, X_train, y_train):
        """
        Train the model on the given data
        
        Args:
            X_train: preprocessed feature matrix
            y_train: tuple of (career_targets, education_targets)
        """
        # If y_train is a tuple of (career_path, education_path)
        career_targets, education_targets = y_train
        
        # Combine targets and use the MultiOutputClassifier
        y = np.column_stack([career_targets, education_targets])
        self.model.fit(X_train, y)
        
        return self
