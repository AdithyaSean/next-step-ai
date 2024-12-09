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
        processed = {
            # Initialize O/L subjects with defaults
            'ol_mathematics': 0.0,
            'ol_science': 0.0,
            'ol_english': 0.0,
            'ol_first_language': 0.0,
            'ol_ict': 0.0,
            'ol_total_passed': 0,
            'ol_core_subjects_average': 0.0,
            
            # Initialize A/L subjects with defaults
            'al_stream': '',
            'al_physics': 0.0,
            'al_chemistry': 0.0,
            'al_biology': 0.0,
            'al_combined_maths': 0.0,
            'al_business_studies': 0.0,
            'al_accounting': 0.0,
            'al_economics': 0.0,
            'al_subject1': 0.0,  # For arts stream or other subjects
            'al_subject2': 0.0,
            'al_subject3': 0.0,
            'al_zscore': 0.0
        }
        
        # Process O/L results if available
        if 'ol_results' in data:
            ol_results = data['ol_results']
            for subject, grade in ol_results.items():
                if subject != 'total_subjects_passed':
                    processed[f'ol_{subject}'] = self._convert_grade_to_numeric(grade)
            processed['ol_total_passed'] = ol_results.get('total_subjects_passed', 0)
            processed['ol_core_subjects_average'] = ol_results.get('core_subjects_average', 0.0)
        
        # Process A/L results if available
        if 'al_results' in data and data['al_results']:
            al_results = data['al_results']
            processed['al_stream'] = al_results.get('stream', '')
            
            # Process subject grades if available
            if 'subjects' in al_results:
                subjects = al_results['subjects']
                
                # Handle specific subjects based on stream
                if processed['al_stream'] == 'PHYSICAL_SCIENCE':
                    processed.update({
                        'al_physics': self._convert_grade_to_numeric(subjects.get('physics', 0)),
                        'al_chemistry': self._convert_grade_to_numeric(subjects.get('chemistry', 0)),
                        'al_combined_maths': self._convert_grade_to_numeric(subjects.get('combined_maths', 0))
                    })
                elif processed['al_stream'] == 'BIOLOGICAL_SCIENCE':
                    processed.update({
                        'al_biology': self._convert_grade_to_numeric(subjects.get('biology', 0)),
                        'al_physics': self._convert_grade_to_numeric(subjects.get('physics', 0)),
                        'al_chemistry': self._convert_grade_to_numeric(subjects.get('chemistry', 0))
                    })
                elif processed['al_stream'] == 'COMMERCE':
                    processed.update({
                        'al_business_studies': self._convert_grade_to_numeric(subjects.get('business_studies', 0)),
                        'al_accounting': self._convert_grade_to_numeric(subjects.get('accounting', 0)),
                        'al_economics': self._convert_grade_to_numeric(subjects.get('economics', 0))
                    })
                else:  # ARTS or other streams
                    # Map generic subjects to subject1, subject2, subject3
                    subject_values = list(subjects.values())
                    for i, grade in enumerate(subject_values[:3]):
                        processed[f'al_subject{i+1}'] = self._convert_grade_to_numeric(grade)
            
            # Add z-score if available
            processed['al_zscore'] = float(al_results.get('zscore', 0.0))
        
        return processed

    def _process_university_data(self, data: Dict) -> Dict:
        """Process university-related data."""
        processed = {
            'degree_type': '',
            'degree_field': '',
            'current_year': 0,
            'current_gpa': 0.0,
            'specialization': '',
            'research_experience': 0,
            'internship_experience': 0
        }
        
        if 'university_data' in data and data['university_data']:
            uni_data = data['university_data']
            processed.update({
                'degree_type': uni_data.get('degree_type', ''),
                'degree_field': uni_data.get('field_of_study', ''),
                'current_year': uni_data.get('current_year', 0),
                'current_gpa': float(uni_data.get('current_gpa', 0.0)),
                'specialization': uni_data.get('specialization', ''),
                'research_experience': len(uni_data.get('significant_projects', [])),
                'internship_experience': len(uni_data.get('internships', []))
            })
        
        return processed

    def _process_preferences(self, data: Dict) -> Dict:
        """Process career preferences and constraints."""
        processed = {
            # Initialize with defaults
            'preferred_role_technical': 0,
            'preferred_role_research': 0,
            'preferred_role_management': 0,
            'preferred_sector_technology': 0,
            'preferred_sector_education': 0,
            'preferred_sector_research': 0,
            'research_oriented': 0,
            'industry_oriented': 0,
            'entrepreneurship_interest': 0,
            'further_studies': 0,
            'industry_experience': 0,
            'startup_plans': 0
        }
        
        # Process career preferences if available
        if 'career_preferences' in data and data['career_preferences']:
            prefs = data['career_preferences']
            
            # Process preferred roles
            for role in prefs.get('preferred_roles', []):
                role_key = f'preferred_role_{role.lower().split()[0]}'
                if role_key in processed:
                    processed[role_key] = 1
            
            # Process preferred sectors
            for sector in prefs.get('preferred_sectors', []):
                sector_key = f'preferred_sector_{sector.lower()}'
                if sector_key in processed:
                    processed[sector_key] = 1
            
            # Process work preferences
            if 'work_preferences' in prefs:
                work_prefs = prefs['work_preferences']
                processed.update({
                    'research_oriented': int(work_prefs.get('research_oriented', False)),
                    'industry_oriented': int(work_prefs.get('industry_oriented', False)),
                    'entrepreneurship_interest': int(work_prefs.get('entrepreneurship_interest', False))
                })
            
            # Process career goals
            if 'career_goals' in prefs:
                goals = prefs['career_goals']
                processed.update({
                    'further_studies': int(goals.get('further_studies', False)),
                    'industry_experience': int(goals.get('industry_experience', False)),
                    'startup_plans': int(goals.get('startup_plans', False))
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
        
        # Ensure all required columns are present with default values
        required_columns = [
            # O/L subjects
            'ol_mathematics', 'ol_science', 'ol_english', 'ol_first_language', 'ol_ict',
            'ol_total_passed', 'ol_core_subjects_average',
            
            # A/L subjects and stream
            'al_stream', 'al_physics', 'al_chemistry', 'al_biology', 'al_combined_maths',
            'al_business_studies', 'al_accounting', 'al_economics',
            'al_subject1', 'al_subject2', 'al_subject3', 'al_zscore',
            
            # University data
            'degree_type', 'degree_field', 'current_year', 'current_gpa', 'specialization',
            'research_experience', 'internship_experience',
            
            # Career preferences
            'preferred_role_technical', 'preferred_role_research', 'preferred_role_management',
            'preferred_sector_technology', 'preferred_sector_education', 'preferred_sector_research',
            'research_oriented', 'industry_oriented', 'entrepreneurship_interest',
            'further_studies', 'industry_experience', 'startup_plans'
        ]
        
        # Add missing columns with default values
        for col in required_columns:
            if col not in df.columns:
                if col.startswith(('ol_', 'al_')) and not col.endswith(('stream', 'passed')):
                    df[col] = 0.0  # Numeric grades default to 0
                elif col.endswith(('_oriented', '_interest', '_studies', '_experience', '_plans')):
                    df[col] = 0  # Boolean flags default to 0
                else:
                    df[col] = ''  # Categorical variables default to empty string
        
        # Reorder columns to match training data
        df = df[required_columns]
        
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
        
        # Transform skills (already handled in _process_preferences)
        
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
