"""
Data preprocessing and feature engineering pipeline for the Career Guidance System
"""

from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

class PreprocessingPipeline:
    """Handles data preprocessing and feature engineering for all prediction stages"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.scalers = {
            'ol': StandardScaler(),
            'al': StandardScaler(),
            'university': StandardScaler()
        }
        self.label_encoders = {
            'stream': LabelEncoder(),
            'program': LabelEncoder(),
            'career': LabelEncoder()
        }
        self._initialize_encoders()
    
    def _initialize_encoders(self):
        """Initialize label encoders with known categories"""
        # A/L Streams
        self.label_encoders['stream'].fit(['Science', 'Commerce', 'Arts'])
        
        # University Programs
        programs = [
            'Computer Science', 'Engineering', 'Medicine', 'Business Administration',
            'Law', 'Architecture', 'Agriculture', 'Arts and Humanities'
        ]
        self.label_encoders['program'].fit(programs)
        
        # Career Paths
        careers = [
            'Software Engineer', 'Data Scientist', 'Doctor', 'Business Analyst',
            'Lawyer', 'Architect', 'Agricultural Scientist', 'Teacher',
            'Accountant', 'Marketing Manager'
        ]
        self.label_encoders['career'].fit(careers)
    
    def preprocess_ol_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess O/L results data"""
        # Select core subjects
        core_subjects = [
            'ol_mathematics', 'ol_science', 'ol_english',
            'ol_sinhala', 'ol_history'
        ]
        
        # Handle missing optional subjects
        optional_subjects = [
            'ol_religion', 'ol_commerce', 'ol_geography',
            'ol_civics', 'ol_tamil'
        ]
        
        # Create feature matrix
        X = data[core_subjects].copy()
        
        # Add optional subjects (with 0 for missing)
        for subject in optional_subjects:
            if subject in data.columns:
                X[subject] = data[subject].fillna(0)
            else:
                X[subject] = 0
        
        # Calculate aggregate features
        X['ol_average'] = X[core_subjects].mean(axis=1)
        X['ol_std'] = X[core_subjects].std(axis=1)
        X['ol_min'] = X[core_subjects].min(axis=1)
        X['ol_max'] = X[core_subjects].max(axis=1)
        
        # Scale numerical features
        numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns
        X[numerical_cols] = self.scalers['ol'].fit_transform(X[numerical_cols])
        
        return X
    
    def preprocess_al_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess A/L results data"""
        # Handle A/L subjects based on stream
        al_features = []
        
        # Stream encoding
        if 'al_stream' in data.columns:
            stream_encoded = pd.get_dummies(data['al_stream'], prefix='stream')
            al_features.append(stream_encoded)
        
        # Subject scores
        subject_cols = [col for col in data.columns if col.startswith('al_subject')]
        if subject_cols:
            subjects = data[subject_cols].copy()
            subjects = subjects.fillna(0)
            al_features.append(subjects)
        
        # Additional A/L features
        extra_features = ['al_general_english', 'al_general_knowledge', 'al_zscore']
        extra = data[extra_features].copy() if all(col in data.columns for col in extra_features) else pd.DataFrame()
        if not extra.empty:
            extra = extra.fillna(0)
            al_features.append(extra)
        
        # Combine all features
        if al_features:
            X = pd.concat(al_features, axis=1)
            
            # Scale numerical features
            numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns
            if not numerical_cols.empty:
                X[numerical_cols] = self.scalers['al'].fit_transform(X[numerical_cols])
        else:
            X = pd.DataFrame(index=data.index)
        
        return X
    
    def preprocess_university_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess university education data"""
        uni_features = []
        
        # Program encoding
        if 'university_program' in data.columns:
            program_encoded = pd.get_dummies(data['university_program'], prefix='program')
            uni_features.append(program_encoded)
        
        # GPA and completion status
        numeric_features = ['university_gpa']
        binary_features = ['university_completed']
        
        for features in [numeric_features, binary_features]:
            cols = [col for col in features if col in data.columns]
            if cols:
                feature_data = data[cols].copy().fillna(0)
                uni_features.append(feature_data)
        
        # Combine all features
        if uni_features:
            X = pd.concat(uni_features, axis=1)
            
            # Scale numerical features
            numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns
            if not numerical_cols.empty:
                X[numerical_cols] = self.scalers['university'].fit_transform(X[numerical_cols])
        else:
            X = pd.DataFrame(index=data.index)
        
        return X
    
    def preprocess_additional_info(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess additional student information"""
        info_features = []
        
        # Process interests and skills if available
        text_features = ['interests', 'skills', 'extracurricular']
        for feature in text_features:
            if feature in data.columns:
                # Convert list to string if necessary
                if isinstance(data[feature].iloc[0], list):
                    feature_data = data[feature].apply(lambda x: ', '.join(x) if x else '')
                else:
                    feature_data = data[feature].fillna('')
                
                # Create binary indicators for common categories
                categories = self.config.get(f'{feature}_categories', [])
                for category in categories:
                    col_name = f'{feature}_{category.lower()}'
                    info_features.append(
                        pd.DataFrame({
                            col_name: feature_data.str.contains(category, case=False).astype(int)
                        })
                    )
        
        # Combine all features
        if info_features:
            X = pd.concat(info_features, axis=1)
        else:
            X = pd.DataFrame(index=data.index)
        
        return X
    
    def combine_features(self, data: pd.DataFrame, stage: str) -> pd.DataFrame:
        """Combine relevant features based on prediction stage"""
        features = []
        
        # Always include O/L data
        ol_features = self.preprocess_ol_data(data)
        features.append(ol_features)
        
        if stage in ['university', 'career']:
            # Include A/L data for university and career prediction
            al_features = self.preprocess_al_data(data)
            features.append(al_features)
        
        if stage == 'career':
            # Include university data for career prediction
            uni_features = self.preprocess_university_data(data)
            additional_features = self.preprocess_additional_info(data)
            features.extend([uni_features, additional_features])
        
        # Combine all relevant features
        X = pd.concat(features, axis=1)
        
        return X
    
    def get_feature_names(self, stage: str) -> List[str]:
        """Get feature names for the specified prediction stage"""
        dummy_data = pd.DataFrame({
            'ol_mathematics': [75], 'ol_science': [80], 'ol_english': [70],
            'ol_sinhala': [85], 'ol_history': [75]
        })
        
        if stage in ['university', 'career']:
            dummy_data['al_stream'] = ['Science']
            dummy_data['al_subject1'] = [75]
            dummy_data['al_subject2'] = [80]
            dummy_data['al_subject3'] = [85]
        
        if stage == 'career':
            dummy_data['university_program'] = ['Computer Science']
            dummy_data['university_gpa'] = [3.5]
            dummy_data['interests'] = [['Technology', 'Science']]
            dummy_data['skills'] = [['Programming', 'Analysis']]
        
        X = self.combine_features(dummy_data, stage)
        return X.columns.tolist()
