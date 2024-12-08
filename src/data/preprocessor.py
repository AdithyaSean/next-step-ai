"""Data preprocessing pipeline for the career guidance system."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

class DataPreprocessor:
    """Preprocess student data for model training and inference."""
    
    def __init__(self):
        """Initialize preprocessor with necessary encoders and scalers."""
        self.grade_scaler = StandardScaler()
        self.zscore_scaler = StandardScaler()
        self.skill_scaler = StandardScaler()
        
        # Encoders for categorical variables
        self.stream_encoder = LabelEncoder()
        self.field_encoder = LabelEncoder()
        self.interest_encoder = LabelEncoder()
        self.career_encoder = LabelEncoder()
        self.environment_encoder = LabelEncoder()  # Added encoder for work environment
        
        # Track fitted state and feature names
        self.is_fitted = False
        self.feature_names: Optional[List[str]] = None
    
    def _convert_grade_to_numeric(self, grade: str) -> float:
        """Convert letter grades to numeric values."""
        if isinstance(grade, (int, float)):
            return float(grade)
        if isinstance(grade, dict):
            return float(grade.get('grade', 0.0))
        return 0.0
    
    def _process_academic_records(self, data: Dict) -> Dict:
        """Process academic records including O/L and A/L results."""
        processed = {}
        
        # Process O/L results
        if 'ol_results' in data:
            ol_results = data['ol_results']
            for subject, grade in ol_results.items():
                if subject != 'total_subjects_passed':
                    processed[f'ol_{subject}'] = self._convert_grade_to_numeric(grade)
            processed['ol_total_passed'] = ol_results.get('total_subjects_passed', 0)
        
        # Process A/L results
        if 'al_results' in data and data['al_results']:
            al_results = data['al_results']
            processed['al_stream'] = al_results.get('stream', '')
            
            # Process subject grades
            if 'subjects' in al_results:
                for subject, grade in al_results['subjects'].items():
                    processed[f'al_{subject}'] = self._convert_grade_to_numeric(grade)
            
            # Add z-score if available
            processed['al_zscore'] = float(al_results.get('zscore', 0.0))
        else:
            processed['al_stream'] = ''
            processed['al_zscore'] = 0.0
        
        return processed
    
    def _process_university_data(self, data: Dict) -> Dict:
        """Process university-related data."""
        processed = {}
        
        if 'university_preferences' in data:
            uni_prefs = data['university_preferences']
            processed.update({
                'degree_field': uni_prefs.get('field', ''),
                'preferred_location': uni_prefs.get('location', ''),
                'max_cost': uni_prefs.get('max_cost', 0)
            })
        
        return processed
    
    def _process_preferences(self, data: Dict) -> Dict:
        """Process career preferences and constraints."""
        processed = {}
        
        if 'career_preferences' in data:
            prefs = data['career_preferences']
            processed.update({
                'min_salary': prefs.get('min_salary', 0),
                'work_environment': prefs.get('work_environment', '')
            })
        
        # Get career path directly from profile (target variable)
        processed['career_path'] = data.get('career_path', 'General Studies')
        
        # Process skills
        if 'skills_assessment' in data:  
            for skill, level in data['skills_assessment'].items():
                processed[f'skill_{skill}'] = level
        
        return processed
    
    def preprocess_single(self, data: Dict) -> Dict:
        """Preprocess a single student profile."""
        processed = {}
        
        # Process each component
        processed.update(self._process_academic_records(data))
        processed.update(self._process_university_data(data))
        processed.update(self._process_preferences(data))
        
        return processed
    
    def fit(self, data: List[Dict]) -> 'DataPreprocessor':
        """Fit preprocessor on training data."""
        # Preprocess all profiles
        processed_data = [self.preprocess_single(profile) for profile in data]
        df = pd.DataFrame(processed_data)
        
        # Store feature names
        self.feature_names = [col for col in df.columns if col != 'career_path']
        
        # Fit grade scaler
        grade_cols = [col for col in df.columns if col.startswith(('ol_', 'al_'))
                     and col not in ['ol_total_passed', 'al_stream']]
        if grade_cols:
            self.grade_scaler.fit(df[grade_cols].fillna(0))
        
        # Fit categorical encoders
        if 'al_stream' in df.columns:
            self.stream_encoder.fit(df['al_stream'].fillna(''))
        if 'degree_field' in df.columns:
            self.field_encoder.fit(df['degree_field'].fillna(''))
        if 'work_environment' in df.columns:  
            self.environment_encoder.fit(df['work_environment'].fillna(''))
        
        # Fit skill scaler
        skill_cols = [col for col in df.columns if col.startswith('skill_')]
        if skill_cols:
            self.skill_scaler.fit(df[skill_cols].fillna(0))
            
        # Fit career path encoder
        if 'career_path' in df.columns:
            self.career_encoder.fit(df['career_path'].fillna(''))
        
        self.is_fitted = True
        return self
    
    def transform(self, data: List[Dict]) -> pd.DataFrame:
        """Transform data using fitted preprocessor."""
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        # Preprocess all profiles
        processed_data = [self.preprocess_single(profile) for profile in data]
        df = pd.DataFrame(processed_data)
        
        # Transform numerical features
        grade_cols = [col for col in df.columns if col.startswith(('ol_', 'al_'))
                     and col not in ['ol_total_passed', 'al_stream']]
        if grade_cols:
            df[grade_cols] = self.grade_scaler.transform(df[grade_cols].fillna(0))
        
        # Transform categorical features
        if 'al_stream' in df.columns:
            df['al_stream'] = self.stream_encoder.transform(df['al_stream'].fillna(''))
        if 'degree_field' in df.columns:
            df['degree_field'] = self.field_encoder.transform(df['degree_field'].fillna(''))
        if 'work_environment' in df.columns:  
            df['work_environment'] = self.environment_encoder.transform(df['work_environment'].fillna(''))
        
        # Transform skills
        skill_cols = [col for col in df.columns if col.startswith('skill_')]
        if skill_cols:
            df[skill_cols] = self.skill_scaler.transform(df[skill_cols].fillna(0))
        
        # Handle target variable if present
        if 'career_path' in df.columns:
            df['career_path'] = self.career_encoder.transform(df['career_path'].fillna(''))
        
        return df
    
    def fit_transform(self, data: List[Dict]) -> pd.DataFrame:
        """Fit preprocessor and transform data."""
        return self.fit(data).transform(data)
    
    def transform_with_target(self, data: List[Dict]) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """Transform data and return features and target separately.
        
        This is a convenience method that returns both features and target variable
        separately, useful for model training scenarios.
        
        Returns:
            Tuple of (features DataFrame, target Series if available)
        """
        df = self.transform(data)
        
        # Extract target if present
        target = None
        if 'career_path' in df.columns:
            target = df['career_path']
            df = df.drop('career_path', axis=1)
            
        return df, target

    def get_feature_names(self) -> List[str]:
        """Get list of feature names after transformation.
        
        This is useful for model interpretation and feature importance analysis.
        Must be called after fitting the preprocessor.
        
        Returns:
            List of feature names
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before getting feature names")
            
        if self.feature_names is None:
            raise ValueError("Feature names not available. This should not happen if preprocessor is fitted.")
            
        return self.feature_names
    
    def save(self, path: str):
        """Save fitted preprocessor to disk."""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted preprocessor")
        
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save all fitted components
        joblib.dump(self.grade_scaler, save_path / 'grade_scaler.joblib')
        joblib.dump(self.zscore_scaler, save_path / 'zscore_scaler.joblib')
        joblib.dump(self.skill_scaler, save_path / 'skill_scaler.joblib')
        joblib.dump(self.stream_encoder, save_path / 'stream_encoder.joblib')
        joblib.dump(self.field_encoder, save_path / 'field_encoder.joblib')
        joblib.dump(self.interest_encoder, save_path / 'interest_encoder.joblib')
        joblib.dump(self.career_encoder, save_path / 'career_encoder.joblib')
        joblib.dump(self.environment_encoder, save_path / 'environment_encoder.joblib')  
        
        # Save feature names
        with open(save_path / 'feature_names.json', 'w') as f:
            json.dump(self.feature_names, f)
    
    @classmethod
    def load(cls, path: str) -> 'DataPreprocessor':
        """Load fitted preprocessor from disk."""
        load_path = Path(path)
        if not load_path.exists():
            raise ValueError(f"Path {path} does not exist")
            
        preprocessor = cls()
        try:
            # Load all components
            preprocessor.grade_scaler = joblib.load(load_path / 'grade_scaler.joblib')
            preprocessor.zscore_scaler = joblib.load(load_path / 'zscore_scaler.joblib')
            preprocessor.skill_scaler = joblib.load(load_path / 'skill_scaler.joblib')
            preprocessor.stream_encoder = joblib.load(load_path / 'stream_encoder.joblib')
            preprocessor.field_encoder = joblib.load(load_path / 'field_encoder.joblib')
            preprocessor.interest_encoder = joblib.load(load_path / 'interest_encoder.joblib')
            preprocessor.career_encoder = joblib.load(load_path / 'career_encoder.joblib')
            preprocessor.environment_encoder = joblib.load(load_path / 'environment_encoder.joblib')  
            
            # Load feature names
            with open(load_path / 'feature_names.json', 'r') as f:
                preprocessor.feature_names = json.load(f)
        except FileNotFoundError as e:
            raise ValueError(f"Failed to load preprocessor components: {e}")
        
        preprocessor.is_fitted = True
        return preprocessor
