import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
import joblib

class CareerPredictor:
    def __init__(self):
        self.model = None
        self.label_encoders = {}
        self.feature_importance = None
        
    def preprocess_features(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Preprocess input features."""
        processed = data.copy()
        
        # Handle numerical features (grades)
        grade_columns = [col for col in processed.columns if col.endswith('_grade')]
        for col in grade_columns:
            processed[col] = processed[col].astype(float)
            
        # Handle categorical features (interests, skills)
        cat_columns = [col for col in processed.columns if col.endswith(('_interest', '_skill'))]
        for col in cat_columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                processed[col] = self.label_encoders[col].fit_transform(processed[col])
            else:
                processed[col] = self.label_encoders[col].transform(processed[col])
                
        return processed
        
    def train(self, X: pd.DataFrame, y: pd.Series):
        """Train the career prediction model."""
        # Preprocess features
        X_processed = self.preprocess_features(X)
        
        # LightGBM parameters optimized for career prediction
        params = {
            'objective': 'multiclass',
            'num_class': len(np.unique(y)),
            'metric': 'multi_logloss',
            'learning_rate': 0.05,
            'max_depth': 6,
            'num_leaves': 32,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'min_data_in_leaf': 10,
            'verbose': -1
        }
        
        # Create LightGBM dataset
        train_data = lgb.Dataset(X_processed, label=y)
        
        # Train model
        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=100,
            valid_sets=[train_data],
            early_stopping_rounds=10
        )
        
        # Store feature importance
        self.feature_importance = dict(zip(
            X_processed.columns,
            self.model.feature_importance()
        ))
        
    def predict(self, X: pd.DataFrame) -> List[Dict]:
        """Predict career paths with confidence scores."""
        X_processed = self.preprocess_features(X)
        probabilities = self.model.predict(X_processed)
        
        predictions = []
        for probs in probabilities:
            # Get top 3 career recommendations
            top_indices = np.argsort(probs)[-3:][::-1]
            recommendations = [
                {
                    'career': self.label_encoders['career'].inverse_transform([idx])[0],
                    'confidence': float(probs[idx]),
                    'supporting_factors': self._get_supporting_factors(X_processed, idx)
                }
                for idx in top_indices
            ]
            predictions.append(recommendations)
            
        return predictions
    
    def _get_supporting_factors(self, X: pd.DataFrame, career_idx: int) -> List[str]:
        """Get factors that influenced the career recommendation."""
        important_features = sorted(
            self.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        factors = []
        for feature, importance in important_features:
            if feature.endswith('_grade'):
                factors.append(f"Strong performance in {feature.replace('_grade', '')}")
            elif feature.endswith('_interest'):
                factors.append(f"Interest in {feature.replace('_interest', '')}")
            elif feature.endswith('_skill'):
                factors.append(f"Skill in {feature.replace('_skill', '')}")
                
        return factors
    
    def save_model(self, path: str):
        """Save model and encoders."""
        joblib.dump({
            'model': self.model,
            'encoders': self.label_encoders,
            'importance': self.feature_importance
        }, path)
    
    def load_model(self, path: str):
        """Load model and encoders."""
        data = joblib.load(path)
        self.model = data['model']
        self.label_encoders = data['encoders']
        self.feature_importance = data['importance']
