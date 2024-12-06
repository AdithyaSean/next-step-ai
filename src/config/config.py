"""
Configuration management for the Career Guidance System
"""

from pathlib import Path
from typing import Dict, Any
import yaml

class Config:
    """Configuration manager for the application"""
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path or str(Path(__file__).parent / 'config.yaml')
        self.config: Dict[str, Any] = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    @property
    def model_config(self) -> Dict[str, Any]:
        """Get model-specific configuration"""
        return self.config.get('model', {})
    
    @property
    def data_config(self) -> Dict[str, Any]:
        """Get data processing configuration"""
        return self.config.get('data', {})
    
    @property
    def training_config(self) -> Dict[str, Any]:
        """Get training configuration"""
        return self.config.get('training', {})
    
    @property
    def preprocessing_config(self) -> Dict[str, Any]:
        """Get preprocessing configuration"""
        return self.config.get('preprocessing', {})