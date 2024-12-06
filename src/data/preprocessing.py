"""
Data preprocessing utilities for the Career Guidance System
"""

from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from ..config.config import Config

class DataPreprocessor:
    """Handles all data preprocessing tasks"""
    
    def __init__(self, config: Config):
        self.config = config
        self.scalers = {}
        self.encoders = {}
        self.vectorizers = {}
        
    def preprocess_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess all features according to their types
        
        Args:
            df: Input DataFrame with raw features
            
        Returns:
            Preprocessed DataFrame
        """
        processed_df = df.copy()
        
        # Process numerical features
        num_features = self.config.data_config['features']['academic']
        processed_df = self._process_numerical(processed_df, num_features)
        
        # Process categorical features
        cat_features = (
            self.config.data_config['features']['interests'] +
            self.config.data_config['features']['activities']
        )
        processed_df = self._process_categorical(processed_df, cat_features)
        
        return processed_df
    
    def _process_numerical(self, df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
        """Process numerical features using specified scaling method"""
        strategy = self.config.preprocessing['numerical_features']['strategy']
        
        if strategy == 'standard_scaler':
            scaler = StandardScaler()
            df[features] = scaler.fit_transform(df[features])
            self.scalers['numerical'] = scaler
            
        return df
    
    def _process_categorical(self, df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
        """Process categorical features using specified encoding method"""
        strategy = self.config.preprocessing['categorical_features']['strategy']
        
        if strategy == 'label_encoding':
            for feature in features:
                encoder = LabelEncoder()
                df[feature] = encoder.fit_transform(df[feature])
                self.encoders[feature] = encoder
        elif strategy == 'one_hot_encoding':
            encoder = OneHotEncoder(sparse=False)
            encoded_features = encoder.fit_transform(df[features])
            feature_names = encoder.get_feature_names_out(features)
            
            # Replace original features with encoded ones
            df = df.drop(columns=features)
            for i, name in enumerate(feature_names):
                df[name] = encoded_features[:, i]
            
            self.encoders['categorical'] = encoder
            
        return df
    
    def process_text_features(self, texts: List[str]) -> np.ndarray:
        """Process text features using specified vectorization method"""
        strategy = self.config.preprocessing['text_features']['strategy']
        
        if strategy == 'tfidf':
            if 'text' not in self.vectorizers:
                self.vectorizers['text'] = TfidfVectorizer(max_features=1000)
            
            return self.vectorizers['text'].fit_transform(texts)
        
        return np.array(texts)
    
    def save_preprocessors(self, path: str):
        """Save all preprocessors for later use"""
        # Implementation for saving preprocessors (e.g., using joblib)
        pass
    
    def load_preprocessors(self, path: str):
        """Load saved preprocessors"""
        # Implementation for loading preprocessors
        pass
