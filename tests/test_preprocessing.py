"""
Tests for data preprocessing functionality
"""

import pytest
import pandas as pd
import numpy as np
from src.config.config import Config
from src.data.preprocessing import DataPreprocessor

@pytest.fixture
def sample_config():
    return {
        'data': {
            'features': {
                'academic': ['ol_mathematics', 'ol_science', 'ol_english', 'ol_history'],
                'interests': ['interests'],
                'activities': ['skills']
            }
        },
        'preprocessing': {
            'numerical_features': {'strategy': 'standard_scaler'},
            'categorical_features': {'strategy': 'label_encoding'},
            'text_features': {'strategy': 'tfidf'}
        }
    }

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'ol_mathematics': [85, 92, 78],
        'ol_science': [88, 95, 82],
        'ol_english': [75, 85, 90],
        'ol_history': [80, 88, 85],
        'interests': ['Technology', 'Healthcare', 'Business'],
        'skills': ['Programming', 'Analysis', 'Communication']
    })

def test_preprocessor_initialization(sample_config):
    config = Config()
    config.config = sample_config
    preprocessor = DataPreprocessor(config)
    assert preprocessor is not None

def test_numerical_preprocessing(sample_config, sample_data):
    config = Config()
    config.config = sample_config
    preprocessor = DataPreprocessor(config)
    
    processed_df = preprocessor.preprocess_features(sample_data)
    
    # Check if numerical features are scaled
    numerical_features = sample_config['data']['features']['academic']
    for feature in numerical_features:
        assert processed_df[feature].mean() == pytest.approx(0, abs=1e-10)
        assert processed_df[feature].std() == pytest.approx(1, abs=1e-10)

def test_categorical_preprocessing(sample_config, sample_data):
    config = Config()
    config.config = sample_config
    preprocessor = DataPreprocessor(config)
    
    processed_df = preprocessor.preprocess_features(sample_data)
    
    # Check if categorical features are encoded
    categorical_features = (
        sample_config['data']['features']['interests'] +
        sample_config['data']['features']['activities']
    )
    for feature in categorical_features:
        assert processed_df[feature].dtype in [np.int32, np.int64]
