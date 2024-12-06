"""
A/L Stream Predictor Model
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from typing import Dict, Any, List, Tuple

from .base_model import BaseModel

class ALStreamPredictor(BaseModel):
    """Predicts suitable A/L streams based on O/L results"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model = self._create_model()
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        
    def _create_model(self) -> xgb.XGBClassifier:
        """Create and configure the XGBoost model"""
        return xgb.XGBClassifier(
            max_depth=self.config['params'].get('max_depth', 6),
            learning_rate=self.config['params'].get('learning_rate', 0.1),
            n_estimators=self.config['params'].get('n_estimators', 100),
            min_child_weight=self.config['params'].get('min_child_weight', 1),
            subsample=self.config['params'].get('subsample', 0.8),
            colsample_bytree=self.config['params'].get('colsample_bytree', 0.8),
            tree_method='hist',
            random_state=42
        )
    
    def _check_ol_requirements(self, X: pd.DataFrame) -> pd.DataFrame:
        """Check if students meet minimum requirements for each stream"""
        requirements = {
            'Science': {
                'ol_mathematics': 65,
                'ol_science': 65,
                'ol_english': 50
            },
            'Commerce': {
                'ol_mathematics': 50,
                'ol_english': 50,
                'ol_commerce': 65
            },
            'Arts': {
                'ol_sinhala': 50,
                'ol_history': 50
            }
        }
        
        stream_eligibility = pd.DataFrame(index=X.index)
        for stream, reqs in requirements.items():
            eligible = True
            for subject, min_grade in reqs.items():
                if subject in X.columns:
                    eligible &= (X[subject] >= min_grade)
            stream_eligibility[f'{stream}_eligible'] = eligible
            
        return stream_eligibility
    
    def train(self, X: np.ndarray, y: np.ndarray, feature_names: List[str] = None) -> None:
        """Train the model"""
        self.feature_names = feature_names
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            eligibility = self._check_ol_requirements(X)
            X = pd.concat([X, eligibility], axis=1)
            X = X.values
        
        # Encode target labels
        y_encoded = self.label_encoder.fit_transform(y)
        self.model.fit(X, y_encoded)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if isinstance(X, pd.DataFrame):
            eligibility = self._check_ol_requirements(X)
            X = pd.concat([X, eligibility], axis=1)
            X = X.values
            
        predictions = self.model.predict(X)
        return self.label_encoder.inverse_transform(predictions)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities"""
        if isinstance(X, pd.DataFrame):
            eligibility = self._check_ol_requirements(X)
            X = pd.concat([X, eligibility], axis=1)
            X = X.values
            
        return self.model.predict_proba(X)
    
    def get_stream_recommendations(self, X: pd.DataFrame) -> List[Dict[str, Any]]:
        """Get detailed stream recommendations with confidence scores"""
        probas = self.predict_proba(X)
        predictions = self.predict(X)
        eligibility = self._check_ol_requirements(X)
        
        recommendations = []
        for i, (pred, proba) in enumerate(zip(predictions, probas)):
            streams = self.label_encoder.classes_
            stream_scores = [
                {
                    'stream': stream,
                    'confidence': float(prob),
                    'eligible': bool(eligibility.iloc[i][f'{stream}_eligible'])
                }
                for stream, prob in zip(streams, proba)
            ]
            # Sort by confidence and eligibility
            stream_scores.sort(key=lambda x: (x['eligible'], x['confidence']), reverse=True)
            recommendations.append(stream_scores)
            
        return recommendations
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance"""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        y_pred = self.predict(X)
        y_true = y
        
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1': f1_score(y_true, y_pred, average='weighted')
        }
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores"""
        importance = self.model.feature_importances_
        feature_names = self.feature_names or [f'f{i}' for i in range(len(importance))]
        return dict(zip(feature_names, importance))
