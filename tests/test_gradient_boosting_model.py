"""
Tests for the Gradient Boosting model
"""

import pytest
import numpy as np
from src.models.gradient_boosting_model import GradientBoostingModel

@pytest.fixture
def sample_config():
    return {
        'params': {
            'max_depth': 3,
            'learning_rate': 0.1,
            'n_estimators': 50,
            'random_state': 42
        },
        'target_columns': ['career_path', 'education_path']
    }

@pytest.fixture
def sample_data():
    # Generate synthetic data for testing
    np.random.seed(42)
    X = np.random.rand(100, 5)  # 100 samples, 5 features
    y = np.random.randint(0, 3, size=(100, 2))  # 2 target columns, 3 classes each
    return X, y

def test_model_initialization(sample_config):
    model = GradientBoostingModel(sample_config)
    assert model is not None
    assert model.config == sample_config

def test_model_training(sample_config, sample_data):
    model = GradientBoostingModel(sample_config)
    X, y = sample_data
    model.train(X, y)
    
    # Test predictions
    predictions = model.predict(X)
    assert predictions.shape == y.shape
    
    # Test probabilities
    probabilities = model.predict_proba(X)
    assert len(probabilities) == len(model.config['target_columns'])

def test_model_evaluation(sample_config, sample_data):
    model = GradientBoostingModel(sample_config)
    X, y = sample_data
    model.train(X, y)
    
    metrics = model.evaluate(X, y)
    assert 'avg_accuracy' in metrics
    assert 'avg_f1' in metrics
    
    # Check if we have metrics for each target
    for target in model.config['target_columns']:
        assert f'{target}_accuracy' in metrics
        assert f'{target}_f1' in metrics

def test_feature_importance(sample_config, sample_data):
    model = GradientBoostingModel(sample_config)
    X, y = sample_data
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    model.train(X, y, feature_names=feature_names)
    
    importance = model.get_feature_importance()
    assert len(importance) == len(model.config['target_columns'])
    
    # Check if we have both weight and gain importance
    for target in importance:
        assert 'weight' in importance[target]
        assert 'gain' in importance[target]
