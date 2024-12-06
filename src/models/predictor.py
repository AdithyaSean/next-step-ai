"""
Unified prediction module supporting both CLI and API usage
"""

from pathlib import Path
import json
from typing import List, Dict, Any, Union
import logging

import numpy as np
import pandas as pd
from pydantic import BaseModel

from src.config.config import Config
from src.data.preprocessing import DataPreprocessor
from src.models.gradient_boosting_model import GradientBoostingModel
from src.utils.logger import setup_logger

logger = logging.getLogger(__name__)

class StudentData(BaseModel):
    """Student data input schema"""
    ol_mathematics: float
    ol_science: float
    ol_english: float
    ol_history: float
    al_stream: str
    interests: str
    skills: str

class CareerPredictor:
    """Unified predictor class for career guidance"""
    
    def __init__(self, model_path: str, config_path: str = 'src/config/config.yaml'):
        """Initialize predictor with model and config"""
        self.config = Config(config_path)
        self.model = GradientBoostingModel(self.config.model_config)
        self.model.load(model_path)
        self.preprocessor = DataPreprocessor(self.config)
        logger.info("CareerPredictor initialized successfully")
    
    def predict_careers(self, data: Union[pd.DataFrame, StudentData]) -> Dict[str, Any]:
        """Make career predictions for input data"""
        try:
            # Convert input if needed
            if isinstance(data, StudentData):
                data = pd.DataFrame([data.dict()])
            
            # Preprocess input
            X = self.preprocessor.preprocess_features(data)
            
            # Make predictions
            predictions = self.model.predict(X)
            probabilities = self.model.predict_proba(X)
            
            # Get feature importance
            importance = self.model.get_feature_importance()
            
            # Format response
            response = {
                "predictions": [
                    {
                        target: {
                            "prediction": pred,
                            "confidence": float(prob.max())
                        }
                        for target, pred, prob in zip(
                            self.config.data_config['target_columns'],
                            predictions[0],
                            probabilities
                        )
                    }
                ],
                "feature_importance": importance
            }
            
            return response
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}", exc_info=True)
            raise
