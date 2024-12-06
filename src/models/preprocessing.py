"""
Data preprocessing and feature engineering pipeline for the Career Guidance System
"""

from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

class PreprocessingPipeline:
    """Handles data preprocessing and feature engineering for all prediction stages"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.scalers = {
            'numerical': StandardScaler()
        }
        self.label_encoders = {
            'stream': LabelEncoder(),
            'program': LabelEncoder(),
            'career': LabelEncoder()
        }
        self.text_vectorizer = TfidfVectorizer(max_features=100)
        self._initialize_encoders()
    
    def _initialize_encoders(self):
        """Initialize label encoders with known categories"""
        # A/L Streams
        self.label_encoders['stream'].fit(['Unknown', 'Physical Science', 'Biological Science', 'Commerce', 'Arts'])
        
        # University Programs
        programs = [
            'Unknown', 'Computer Science', 'Engineering', 'Medicine', 'Physical Science',
            'Biological Science', 'Business Administration', 'Management',
            'Arts and Humanities', 'Professional Certificati', 'Technical Training', 'Incomplete Degree'
        ]
        self.label_encoders['program'].fit(programs)
        
        # Career Paths
        careers = [
            'Unknown', 'Software Engineer', 'Data Scientist', 'Doctor', 'Engineer',
            'Research Scientist', 'Business Analyst', 'Accountant', 'Teacher', 
            'Manager', 'Consultant', 'Sales', 'Technical Support', 'Administrative'
        ]
        self.label_encoders['career'].fit(careers)
    
    def preprocess_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Main preprocessing function that handles all feature types"""
        processed_data = pd.DataFrame()
        
        # Process core subjects (numerical)
        core_subjects = self.config['data']['features']['core_subjects']
        if any(subj in data.columns for subj in core_subjects):
            numerical_data = data[core_subjects].fillna(0)
            processed_data[core_subjects] = self.scalers['numerical'].fit_transform(numerical_data)
        
        # Process progress status (boolean)
        progress_fields = self.config['data']['features']['progress_status']
        for field in progress_fields:
            if field in data.columns:
                processed_data[field] = data[field].fillna(False).astype(int)
        
        # Process A/L info
        if 'al_stream' in data.columns:
            processed_data['al_stream'] = self.label_encoders['stream'].transform(data['al_stream'].fillna('Unknown'))
            
            # Process A/L subjects based on stream
            al_subjects = [col for col in data.columns if col.startswith('al_') and col not in ['al_stream', 'al_zscore']]
            for subject in al_subjects:
                if subject in data.columns:
                    processed_data[subject] = self.scalers['numerical'].fit_transform(data[subject].fillna(0).values.reshape(-1, 1))
        
        # Process university info
        if 'university_program' in data.columns:
            processed_data['university_program'] = self.label_encoders['program'].transform(data['university_program'].fillna('Unknown'))
        if 'university_gpa' in data.columns:
            processed_data['university_gpa'] = self.scalers['numerical'].fit_transform(data['university_gpa'].fillna(0).values.reshape(-1, 1))
        
        # Process additional information (text data)
        text_fields = ['interests', 'skills', 'extracurricular']
        for field in text_fields:
            if field in data.columns:
                # Convert list to string if necessary
                text_data = data[field].apply(lambda x: ' '.join(x) if isinstance(x, list) else str(x))
                text_features = self.text_vectorizer.fit_transform(text_data)
                text_df = pd.DataFrame(
                    text_features.toarray(),
                    columns=[f'{field}_{i}' for i in range(text_features.shape[1])]
                )
                processed_data = pd.concat([processed_data, text_df], axis=1)
        
        return processed_data
    
    def preprocess_targets(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess target variables"""
        career_path = self.label_encoders['career'].transform(data['career_path'].fillna('Unknown'))
        education_path = self.label_encoders['program'].transform(data['education_path'].fillna('Unknown'))
        return career_path, education_path
        
    def inverse_transform_predictions(self, career_preds: np.ndarray, education_preds: np.ndarray) -> Tuple[List[str], List[str]]:
        """Convert numerical predictions back to human-readable labels"""
        careers = self.label_encoders['career'].inverse_transform(career_preds)
        education = self.label_encoders['program'].inverse_transform(education_preds)
        return careers.tolist(), education.tolist()
