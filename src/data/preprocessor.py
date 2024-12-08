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
        
        # Mapping dictionaries
        self.grade_mapping = {
            'A': 75, 'B': 65, 'C': 55, 'S': 85,
            'F': 35, 'W': 0
        }
        
        # Track fitted state
        self.is_fitted = False
    
    def _convert_grade_to_numeric(self, grade: str) -> float:
        """Convert letter grades to numeric values."""
        if isinstance(grade, (int, float)):
            return float(grade)
        return self.grade_mapping.get(grade.upper(), 0.0)
    
    def _process_academic_records(self, data: Dict) -> Dict:
        """Process academic records including O/L and A/L results."""
        processed = {}
        
        # Process O/L results
        if 'ol_results' in data:
            ol_results = data['ol_results']
            processed.update({
                f'ol_{k}': self._convert_grade_to_numeric(v)
                for k, v in ol_results.items()
                if k != 'total_subjects_passed'
            })
            processed['ol_total_passed'] = ol_results.get('total_subjects_passed', 0)
        
        # Process A/L results
        if 'al_results' in data:
            al_results = data['al_results']
            processed['al_stream'] = al_results.get('stream', '')
            for subj, grade in al_results.get('subjects', {}).items():
                processed[f'al_{subj}'] = self._convert_grade_to_numeric(grade)
            processed['al_zscore'] = float(al_results.get('zscore', 0))
        
        return processed
    
    def _process_university_data(self, data: Dict) -> Dict:
        """Process university-related data."""
        processed = {}
        
        if 'university_data' in data:
            uni_data = data['university_data']
            processed.update({
                'degree_field': uni_data.get('field_of_study', ''),
                'degree_year': uni_data.get('current_year', 0),
                'current_gpa': float(uni_data.get('current_gpa', 0)),
            })
            
            # Process technical skills
            tech_skills = uni_data.get('technical_competencies', {})
            for skill, level in tech_skills.items():
                processed[f'skill_{skill}'] = float(level)
            
            # Process projects and internships
            processed['num_projects'] = len(uni_data.get('significant_projects', []))
            processed['num_internships'] = len(uni_data.get('internships', []))
        
        return processed
    
    def _process_preferences(self, data: Dict) -> Dict:
        """Process career preferences and constraints."""
        processed = {}
        
        if 'career_preferences' in data:
            prefs = data['career_preferences']
            processed['preferred_roles'] = prefs.get('preferred_roles', [])
            processed['preferred_sectors'] = prefs.get('preferred_sectors', [])
            
            # Work preferences
            work_prefs = prefs.get('work_preferences', {})
            for k, v in work_prefs.items():
                processed[f'pref_{k}'] = int(v)
            
            # Career goals
            goals = prefs.get('career_goals', {})
            for k, v in goals.items():
                processed[f'goal_{k}'] = int(v)
        
        # Process constraints
        if 'constraints' in data:
            constraints = data['constraints']
            processed.update({
                'location': constraints.get('preferred_location', ''),
                'has_financial_constraints': int(constraints.get('financial_constraints', False)),
                'can_relocate': int(constraints.get('willing_to_relocate', True))
            })
        
        return processed
    
    def preprocess_single(self, data: Dict) -> Dict:
        """Preprocess a single student profile."""
        processed = {}
        
        # Process each component
        processed.update(self._process_academic_records(data))
        processed.update(self._process_university_data(data))
        processed.update(self._process_preferences(data))
        
        return processed
    
    def fit(self, data: List[Dict]):
        """Fit preprocessor on training data."""
        # Convert to DataFrame for easier processing
        processed_data = [self.preprocess_single(profile) for profile in data]
        df = pd.DataFrame(processed_data)
        
        # Fit grade scalers
        grade_cols = [col for col in df.columns if col.startswith(('ol_', 'al_')) 
                     and col not in ['ol_total_passed', 'al_stream']]
        if grade_cols:
            self.grade_scaler.fit(df[grade_cols].fillna(0))
        
        # Fit categorical encoders
        if 'al_stream' in df:
            self.stream_encoder.fit(df['al_stream'].fillna(''))
        if 'degree_field' in df:
            self.field_encoder.fit(df['degree_field'].fillna(''))
        
        # Fit skill scaler
        skill_cols = [col for col in df.columns if col.startswith('skill_')]
        if skill_cols:
            self.skill_scaler.fit(df[skill_cols].fillna(0))
        
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
        if 'al_stream' in df:
            df['al_stream'] = self.stream_encoder.transform(df['al_stream'].fillna(''))
        if 'degree_field' in df:
            df['degree_field'] = self.field_encoder.transform(df['degree_field'].fillna(''))
        
        # Transform skills
        skill_cols = [col for col in df.columns if col.startswith('skill_')]
        if skill_cols:
            df[skill_cols] = self.skill_scaler.transform(df[skill_cols].fillna(0))
        
        return df
    
    def fit_transform(self, data: List[Dict]) -> pd.DataFrame:
        """Fit preprocessor and transform data."""
        return self.fit(data).transform(data)
    
    def save(self, path: str):
        """Save fitted preprocessor to disk."""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted preprocessor")
        
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save scikit-learn objects
        joblib.dump(self.grade_scaler, save_path / 'grade_scaler.pkl')
        joblib.dump(self.zscore_scaler, save_path / 'zscore_scaler.pkl')
        joblib.dump(self.skill_scaler, save_path / 'skill_scaler.pkl')
        joblib.dump(self.stream_encoder, save_path / 'stream_encoder.pkl')
        joblib.dump(self.field_encoder, save_path / 'field_encoder.pkl')
        
        # Save mappings
        with open(save_path / 'grade_mapping.json', 'w') as f:
            json.dump(self.grade_mapping, f)
    
    @classmethod
    def load(cls, path: str) -> 'DataPreprocessor':
        """Load fitted preprocessor from disk."""
        load_path = Path(path)
        if not load_path.exists():
            raise ValueError(f"Path {path} does not exist")
        
        preprocessor = cls()
        
        # Load scikit-learn objects
        preprocessor.grade_scaler = joblib.load(load_path / 'grade_scaler.pkl')
        preprocessor.zscore_scaler = joblib.load(load_path / 'zscore_scaler.pkl')
        preprocessor.skill_scaler = joblib.load(load_path / 'skill_scaler.pkl')
        preprocessor.stream_encoder = joblib.load(load_path / 'stream_encoder.pkl')
        preprocessor.field_encoder = joblib.load(load_path / 'field_encoder.pkl')
        
        # Load mappings
        with open(load_path / 'grade_mapping.json', 'r') as f:
            preprocessor.grade_mapping = json.load(f)
        
        preprocessor.is_fitted = True
        return preprocessor
