"""
Base model interface for the Career Guidance System
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd

class BaseModel(ABC):
    """Abstract base class for all career guidance models"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
    
    @abstractmethod
    def train(self, X: pd.DataFrame, career_targets: np.ndarray, education_targets: np.ndarray) -> None:
        """
        Train the model on the given data
        
        Args:
            X: Training features DataFrame
            career_targets: Career path target labels
            education_targets: Education path target labels
        """
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions for the given input
        
        Args:
            X: Input features DataFrame
            
        Returns:
            Tuple of (career_predictions, education_predictions)
        """
        pass
    
    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get prediction probabilities for the given input
        
        Args:
            X: Input features DataFrame
            
        Returns:
            Tuple of (career_probabilities, education_probabilities)
        """
        pass
    
    def get_feature_importance(self) -> Dict[str, np.ndarray]:
        """
        Get feature importance scores if the model supports it
        
        Returns:
            Dictionary with feature importance scores for career and education predictions
        """
        if hasattr(self.model, 'feature_importances_'):
            return {
                'career': self.model.feature_importances_[:len(self.config['data']['features'])],
                'education': self.model.feature_importances_[len(self.config['data']['features']):]
            }
        return {}
    
    def save_model(self, path: str) -> None:
        """Save model to disk"""
        if self.model is not None:
            import joblib
            joblib.dump(self.model, path)
    
    def load_model(self, path: str) -> None:
        """Load model from disk"""
        import joblib
        self.model = joblib.load(path)
