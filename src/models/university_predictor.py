"""
University Program Predictor Model
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from typing import Dict, Any, List, Tuple

from .base_model import BaseModel

class UniversityPredictor(BaseModel):
    """Predicts suitable university programs based on O/L and A/L results"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model = self._create_model()
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        
    def _create_model(self) -> lgb.LGBMClassifier:
        """Create and configure the LightGBM model"""
        return lgb.LGBMClassifier(
            num_leaves=self.config['params'].get('num_leaves', 31),
            learning_rate=self.config['params'].get('learning_rate', 0.05),
            n_estimators=self.config['params'].get('n_estimators', 150),
            feature_fraction=self.config['params'].get('feature_fraction', 0.9),
            random_state=42
        )
    
    def _check_program_requirements(self, X: pd.DataFrame) -> pd.DataFrame:
        """Check if students meet minimum requirements for each program"""
        requirements = {
            'Engineering': {
                'al_stream': 'Science',
                'zscore_min': 1.8,
                'subjects': {
                    'al_physics': 65,
                    'al_chemistry': 65
                }
            },
            'Medicine': {
                'al_stream': 'Science',
                'zscore_min': 2.0,
                'subjects': {
                    'al_biology': 70,
                    'al_chemistry': 70
                }
            },
            'Business': {
                'al_stream': 'Commerce',
                'zscore_min': 1.5,
                'subjects': {
                    'al_business_studies': 65,
                    'al_economics': 65
                }
            }
        }
        
        program_eligibility = pd.DataFrame(index=X.index)
        for program, reqs in requirements.items():
            eligible = (X['al_stream'] == reqs['al_stream'])
            if 'al_zscore' in X.columns:
                eligible &= (X['al_zscore'] >= reqs['zscore_min'])
            for subject, min_grade in reqs['subjects'].items():
                if subject in X.columns:
                    eligible &= (X[subject] >= min_grade)
            program_eligibility[f'{program}_eligible'] = eligible
            
        return program_eligibility
    
    def train(self, X: np.ndarray, y: np.ndarray, feature_names: List[str] = None) -> None:
        """Train the model"""
        self.feature_names = feature_names
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            eligibility = self._check_program_requirements(X)
            X = pd.concat([X, eligibility], axis=1)
            X = X.values
            
        # Encode target labels
        y_encoded = self.label_encoder.fit_transform(y)
        self.model.fit(X, y_encoded)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if isinstance(X, pd.DataFrame):
            eligibility = self._check_program_requirements(X)
            X = pd.concat([X, eligibility], axis=1)
            X = X.values
            
        predictions = self.model.predict(X)
        return self.label_encoder.inverse_transform(predictions)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities"""
        if isinstance(X, pd.DataFrame):
            eligibility = self._check_program_requirements(X)
            X = pd.concat([X, eligibility], axis=1)
            X = X.values
            
        return self.model.predict_proba(X)
    
    def get_program_recommendations(self, X: pd.DataFrame) -> List[Dict[str, Any]]:
        """Get detailed program recommendations with success probabilities"""
        probas = self.predict_proba(X)
        predictions = self.predict(X)
        eligibility = self._check_program_requirements(X)
        
        recommendations = []
        for i, (pred, proba) in enumerate(zip(predictions, probas)):
            programs = self.label_encoder.classes_
            program_scores = [
                {
                    'program': program,
                    'success_probability': float(prob),
                    'eligible': bool(eligibility.iloc[i][f'{program}_eligible']),
                    'requirements_met': self._get_requirements_status(X.iloc[i], program)
                }
                for program, prob in zip(programs, proba)
            ]
            # Sort by eligibility and success probability
            program_scores.sort(key=lambda x: (x['eligible'], x['success_probability']), reverse=True)
            recommendations.append(program_scores)
            
        return recommendations
    
    def _get_requirements_status(self, student_data: pd.Series, program: str) -> Dict[str, Any]:
        """Get detailed status of program requirements for a student"""
        requirements = {
            'Engineering': {
                'al_stream': 'Science',
                'zscore_min': 1.8,
                'subjects': {
                    'al_physics': 65,
                    'al_chemistry': 65
                }
            },
            'Medicine': {
                'al_stream': 'Science',
                'zscore_min': 2.0,
                'subjects': {
                    'al_biology': 70,
                    'al_chemistry': 70
                }
            },
            'Business': {
                'al_stream': 'Commerce',
                'zscore_min': 1.5,
                'subjects': {
                    'al_business_studies': 65,
                    'al_economics': 65
                }
            }
        }
        
        if program not in requirements:
            return {}
            
        reqs = requirements[program]
        status = {
            'stream_match': student_data['al_stream'] == reqs['al_stream'],
            'zscore_met': student_data.get('al_zscore', 0) >= reqs['zscore_min'],
            'subject_requirements': {}
        }
        
        for subject, min_grade in reqs['subjects'].items():
            if subject in student_data:
                status['subject_requirements'][subject] = {
                    'required': min_grade,
                    'actual': student_data[subject],
                    'met': student_data[subject] >= min_grade
                }
                
        return status
    
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
