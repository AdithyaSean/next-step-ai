"""
Career Path Predictor Model using Ensemble Learning
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from typing import Dict, Any, List, Tuple

from .base_model import BaseModel

class CareerPathPredictor(BaseModel):
    """Predicts career paths using an ensemble of XGBoost and Random Forest"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.models = self._create_models()
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        
    def _create_models(self) -> Dict[str, Any]:
        """Create and configure the ensemble models"""
        xgb_config = self.config['base_models'][0]
        rf_config = self.config['base_models'][1]
        
        return {
            'xgboost': {
                'model': xgb.XGBClassifier(
                    max_depth=xgb_config['params'].get('max_depth', 7),
                    learning_rate=xgb_config['params'].get('learning_rate', 0.1),
                    n_estimators=xgb_config['params'].get('n_estimators', 120),
                    tree_method='hist',
                    random_state=42
                ),
                'weight': xgb_config['weight']
            },
            'random_forest': {
                'model': RandomForestClassifier(
                    n_estimators=rf_config['params'].get('n_estimators', 100),
                    max_depth=rf_config['params'].get('max_depth', 10),
                    random_state=42
                ),
                'weight': rf_config['weight']
            }
        }
    
    def _preprocess_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Preprocess features for career prediction"""
        # Calculate academic performance metrics
        if isinstance(X, pd.DataFrame):
            X = X.copy()
            
            # O/L performance
            ol_subjects = [col for col in X.columns if col.startswith('ol_')]
            if ol_subjects:
                X['ol_average'] = X[ol_subjects].mean(axis=1)
                
            # A/L performance
            al_subjects = [col for col in X.columns if col.startswith('al_') and col != 'al_stream']
            if al_subjects:
                X['al_average'] = X[al_subjects].mean(axis=1)
                
            # Education level completion
            X['education_level'] = 1  # Base level (O/L)
            if 'al_passed' in X.columns:
                X.loc[X['al_passed'], 'education_level'] = 2
            if 'university_completed' in X.columns:
                X.loc[X['university_completed'], 'education_level'] = 3
                
        return X
    
    def train(self, X: np.ndarray, y: np.ndarray, feature_names: List[str] = None) -> None:
        """Train the ensemble models"""
        self.feature_names = feature_names
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X = self._preprocess_features(X)
            X = X.values
            
        # Encode target labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Train each model in the ensemble
        for model_info in self.models.values():
            model_info['model'].fit(X, y_encoded)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using weighted ensemble"""
        if isinstance(X, pd.DataFrame):
            X = self._preprocess_features(X)
            X = X.values
            
        # Get predictions from each model
        predictions = []
        weights = []
        
        for name, model_info in self.models.items():
            model_pred = model_info['model'].predict(X)
            predictions.append(model_pred)
            weights.append(model_info['weight'])
            
        # Weighted voting
        weighted_pred = np.zeros_like(predictions[0], dtype=float)
        for pred, weight in zip(predictions, weights):
            weighted_pred += weight * pred
            
        final_pred = np.round(weighted_pred / sum(weights)).astype(int)
        return self.label_encoder.inverse_transform(final_pred)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities using weighted ensemble"""
        if isinstance(X, pd.DataFrame):
            X = self._preprocess_features(X)
            X = X.values
            
        # Get probabilities from each model
        probas = []
        weights = []
        
        for name, model_info in self.models.items():
            model_proba = model_info['model'].predict_proba(X)
            probas.append(model_proba)
            weights.append(model_info['weight'])
            
        # Weighted average of probabilities
        weighted_proba = np.zeros_like(probas[0])
        for proba, weight in zip(probas, weights):
            weighted_proba += weight * proba
            
        return weighted_proba / sum(weights)
    
    def get_career_recommendations(self, X: pd.DataFrame, top_k: int = 3) -> List[Dict[str, Any]]:
        """Get detailed career recommendations with confidence scores"""
        probas = self.predict_proba(X)
        
        recommendations = []
        for student_proba in probas:
            # Get top k careers
            top_indices = student_proba.argsort()[-top_k:][::-1]
            careers = self.label_encoder.classes_[top_indices]
            scores = student_proba[top_indices]
            
            career_list = [
                {
                    'career': career,
                    'confidence': float(score),
                    'requirements': self._get_career_requirements(career),
                    'skills_needed': self._get_required_skills(career)
                }
                for career, score in zip(careers, scores)
            ]
            recommendations.append(career_list)
            
        return recommendations
    
    def _get_career_requirements(self, career: str) -> Dict[str, Any]:
        """Get educational requirements for a career"""
        # This would be expanded based on actual career data
        requirements = {
            'Software Engineer': {
                'education_level': 'University',
                'preferred_streams': ['Science'],
                'preferred_programs': ['Computer Science', 'Engineering'],
                'min_gpa': 2.75
            },
            'Business Analyst': {
                'education_level': 'University',
                'preferred_streams': ['Commerce', 'Science'],
                'preferred_programs': ['Business Administration', 'Economics'],
                'min_gpa': 2.5
            }
            # Add more careers as needed
        }
        return requirements.get(career, {})
    
    def _get_required_skills(self, career: str) -> List[str]:
        """Get required skills for a career"""
        # This would be expanded based on actual career data
        skills = {
            'Software Engineer': [
                'Programming',
                'Problem Solving',
                'Software Development',
                'Database Management'
            ],
            'Business Analyst': [
                'Data Analysis',
                'Business Process Modeling',
                'Requirements Gathering',
                'Communication'
            ]
            # Add more careers as needed
        }
        return skills.get(career, [])
    
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
        """Get feature importance scores from both models"""
        importance_dict = {}
        
        # Get XGBoost feature importance
        xgb_importance = self.models['xgboost']['model'].feature_importances_
        xgb_weight = self.models['xgboost']['weight']
        
        # Get Random Forest feature importance
        rf_importance = self.models['random_forest']['model'].feature_importances_
        rf_weight = self.models['random_forest']['weight']
        
        # Combine weighted importance scores
        total_weight = xgb_weight + rf_weight
        combined_importance = (xgb_importance * xgb_weight + rf_importance * rf_weight) / total_weight
        
        feature_names = self.feature_names or [f'f{i}' for i in range(len(combined_importance))]
        return dict(zip(feature_names, combined_importance))
