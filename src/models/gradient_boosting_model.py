from typing import Dict, Any, List, Tuple, Union
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.multioutput import MultiOutputClassifier

from .base_model import BaseModel

class GradientBoostingModel(BaseModel):
    """XGBoost-based model for multi-target prediction"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._init_model()
    
    def _init_model(self):
        """Initialize XGBoost model with configuration parameters"""
        base_model = xgb.XGBClassifier(
            max_depth=self.config['params']['max_depth'],
            learning_rate=self.config['params']['learning_rate'],
            n_estimators=self.config['params']['n_estimators'],
            min_child_weight=self.config['params']['min_child_weight'],
            subsample=self.config['params']['subsample'],
            colsample_bytree=self.config['params']['colsample_bytree'],
            random_state=self.config['params']['random_state'],
            enable_categorical=True
        )
        self.model = MultiOutputClassifier(base_model)
        self.feature_names = None
    
    def train(self, X: pd.DataFrame, y: np.ndarray, feature_names: List[str] = None):
        """Train the model"""
        self.feature_names = feature_names or (
            X.columns.tolist() if isinstance(X, pd.DataFrame) else None
        )
        self.model.fit(X, y)
        return self
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Make predictions on new data
        
        Args:
            X: Input features as DataFrame or numpy array
        
        Returns:
            np.ndarray: Predicted labels
        """
        if isinstance(X, pd.DataFrame):
            # Handle categorical columns
            categorical_columns = ['al_stream', 'university_program']
            X = X.copy()
            for col in categorical_columns:
                if col in X.columns:
                    X[col] = pd.Categorical(X[col]).codes
        
        return self.model.predict(X)
    
    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Get prediction probabilities
        
        Args:
            X: Input features as DataFrame or numpy array
        
        Returns:
            np.ndarray: Prediction probabilities
        """
        if isinstance(X, pd.DataFrame):
            # Handle categorical columns
            categorical_columns = ['al_stream', 'university_program']
            X = X.copy()
            for col in categorical_columns:
                if col in X.columns:
                    X[col] = pd.Categorical(X[col]).codes
        
        return self.model.predict_proba(X)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance
        
        Args:
            X: Input features
            y: True labels
        
        Returns:
            Dictionary containing evaluation metrics for each target
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        predictions = self.predict(X)
        metrics = {}
        
        # Calculate metrics for each target
        for i in range(y.shape[1]):
            # Get the label encoder for this target
            encoder = self.label_encoders[f'target_{i}'] if hasattr(self, 'label_encoders') else None
            
            # Convert string labels to numeric if needed
            y_true = y[:, i]
            y_pred = predictions[:, i]
            
            if encoder is not None and y_true.dtype == object:
                y_true = encoder.transform(y_true)
                if y_pred.dtype == object:
                    y_pred = encoder.transform(y_pred)
            
            # Calculate metrics
            metrics[f'target_{i}_accuracy'] = accuracy_score(y_true, y_pred)
            metrics[f'target_{i}_precision'] = precision_score(
                y_true, y_pred, average='weighted'
            )
            metrics[f'target_{i}_recall'] = recall_score(
                y_true, y_pred, average='weighted'
            )
            metrics[f'target_{i}_f1'] = f1_score(
                y_true, y_pred, average='weighted'
            )
        
        # Calculate average metrics across all targets
        metrics['avg_accuracy'] = np.mean([
            v for k, v in metrics.items() if k.endswith('_accuracy')
        ])
        metrics['avg_f1'] = np.mean([
            v for k, v in metrics.items() if k.endswith('_f1')
        ])
        
        return metrics
    
    def get_feature_importance(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Get feature importance scores for each target"""
        importance = {}
        
        for i, estimator in enumerate(self.model.estimators_):
            target_importance = {
                'weight': dict(zip(
                    self.feature_names or [f'f{j}' for j in range(len(estimator.feature_importances_))],
                    estimator.feature_importances_
                ))
            }
            importance[f'target_{i}'] = target_importance
            
        return importance
    
    def save(self, path: str) -> None:
        """Save model to disk"""
        import joblib
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'label_encoders': getattr(self, 'label_encoders', None)
        }
        joblib.dump(model_data, path)
    
    def load(self, path: str) -> None:
        """Load model from disk"""
        import joblib
        model_data = joblib.load(path)
        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        if 'label_encoders' in model_data:
            self.label_encoders = model_data['label_encoders']
    
    def plot_feature_importance(self, target_idx: int = 0, top_n: int = 10) -> None:
        """
        Plot feature importance for a specific target
        
        Args:
            target_idx: Index of the target to plot importance for
            top_n: Number of top features to show
        
        Returns:
            matplotlib.figure.Figure: The generated plot figure
        
        Raises:
            ImportError: If matplotlib is not installed
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            raise ImportError(
                "Visualization dependencies are required. Install them with:\n"
                "pip install matplotlib seaborn"
            )
        
        # Get feature importance for the specified target
        importance = self.get_feature_importance()
        target_importance = importance[f'target_{target_idx}']['weight']
        
        # Sort features by importance
        sorted_features = sorted(
            target_importance.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:top_n]
        
        # Prepare data for plotting
        features, scores = zip(*sorted_features)
        
        # Create bar plot
        fig = plt.figure(figsize=(10, 6))
        sns.barplot(x=list(scores), y=list(features))
        plt.title(f'Top {top_n} Feature Importance for Target {target_idx}')
        plt.xlabel('Importance Score')
        plt.ylabel('Features')
        plt.tight_layout()
        
        return fig