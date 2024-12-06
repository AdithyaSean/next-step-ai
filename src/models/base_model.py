"""
Base model interface for the Career Guidance System
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List
import numpy as np
import pandas as pd

class BaseModel(ABC):
    """Abstract base class for all career guidance models"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
    
    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the model on the given data
        
        Args:
            X: Training features
            y: Training labels
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions for the given input
        
        Args:
            X: Input features
            
        Returns:
            Predicted labels
        """
        pass
    
    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities for the given input
        
        Args:
            X: Input features
            
        Returns:
            Prediction probabilities
        """
        pass
    
    @abstractmethod
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance
        
        Args:
            X: Test features
            y: Test labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        pass
    
    def save(self, path: str) -> None:
        """Save model to disk"""
        pass
    
    def load(self, path: str) -> None:
        """Load model from disk"""
        pass
    
    @abstractmethod
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        pass
