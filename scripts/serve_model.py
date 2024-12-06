"""
Script to serve the model via a REST API
"""

import argparse
from pathlib import Path
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any

from src.config.config import Config
from src.data.preprocessing import DataPreprocessor
from src.models.gradient_boosting_model import GradientBoostingModel
from src.utils.logger import setup_logger

# Initialize FastAPI app
app = FastAPI(
    title="Next Step Career Guidance API",
    description="API for career predictions based on student data",
    version="1.0.0"
)

# Global objects
model = None
preprocessor = None
config = None
logger = None

class StudentData(BaseModel):
    """Student data input schema"""
    ol_mathematics: float
    ol_science: float
    ol_english: float
    ol_history: float
    al_stream: str
    interests: str
    skills: str

class PredictionResponse(BaseModel):
    """Prediction response schema"""
    predictions: List[Dict[str, Any]]
    feature_importance: Dict[str, Dict[str, float]]

@app.post("/predict", response_model=PredictionResponse)
async def predict(student: StudentData):
    """Make career predictions for a student"""
    try:
        # Convert input to DataFrame
        import pandas as pd
        data = pd.DataFrame([student.dict()])
        
        # Preprocess input
        X = preprocessor.preprocess_features(data)
        
        # Make predictions
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)
        
        # Get feature importance
        importance = model.get_feature_importance()
        
        # Format response
        response = {
            "predictions": [
                {
                    target: {
                        "prediction": pred,
                        "confidence": float(prob.max())
                    }
                    for target, pred, prob in zip(
                        config.data_config['target_columns'],
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
        raise HTTPException(status_code=500, detail=str(e))

def load_model(model_path: str):
    """Load the trained model and preprocessor"""
    global model, preprocessor, config, logger
    
    try:
        # Load configuration
        config = Config()
        
        # Load model
        model = GradientBoostingModel(config.model_config)
        model.load(model_path)
        
        # Initialize preprocessor
        preprocessor = DataPreprocessor(config)
        
        logger.info("Model and preprocessor loaded successfully")
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}", exc_info=True)
        raise

def parse_args():
    parser = argparse.ArgumentParser(description='Serve model via REST API')
    parser.add_argument('--model', type=str, required=True,
                      help='Path to trained model')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                      help='Host to serve on')
    parser.add_argument('--port', type=int, default=8000,
                      help='Port to serve on')
    return parser.parse_args()

def main():
    global logger
    
    # Parse arguments
    args = parse_args()
    
    # Set up logging
    logger = setup_logger('serve_model', 'logs/serve_model.log')
    
    try:
        # Load model
        load_model(args.model)
        
        # Start server
        logger.info(f"Starting server on {args.host}:{args.port}")
        uvicorn.run(app, host=args.host, port=args.port)
        
    except Exception as e:
        logger.error(f"Server error: {str(e)}", exc_info=True)
        raise

if __name__ == '__main__':
    main()
