"""
Unified Career Guidance System that orchestrates the three-stage prediction process
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
import yaml
from pydantic import BaseModel

import pandas as pd
import numpy as np

from .al_stream_predictor import ALStreamPredictor
from .university_predictor import UniversityPredictor
from .career_path_predictor import CareerPathPredictor

logger = logging.getLogger(__name__)

class StudentProfile(BaseModel):
    """Input schema for student data"""
    # O/L Results
    ol_mathematics: float
    ol_science: float
    ol_english: float
    ol_sinhala: float
    ol_history: float
    ol_religion: Optional[float] = None
    ol_commerce: Optional[float] = None
    ol_geography: Optional[float] = None
    ol_civics: Optional[float] = None
    ol_tamil: Optional[float] = None
    
    # A/L Results (optional)
    al_stream: Optional[str] = None
    al_subject1: Optional[float] = None
    al_subject2: Optional[float] = None
    al_subject3: Optional[float] = None
    al_general_english: Optional[float] = None
    al_general_knowledge: Optional[float] = None
    al_zscore: Optional[float] = None
    
    # University Results (optional)
    university_program: Optional[str] = None
    university_gpa: Optional[float] = None
    university_completed: Optional[bool] = None
    
    # Additional Information
    interests: Optional[List[str]] = None
    skills: Optional[List[str]] = None
    extracurricular: Optional[List[str]] = None

class CareerGuidanceSystem:
    """
    Unified Career Guidance System that manages the three-stage prediction process:
    1. A/L Stream Prediction
    2. University Program Prediction
    3. Career Path Prediction
    """
    
    def __init__(self, config_path: str = None):
        """Initialize the career guidance system"""
        self.config_path = config_path or str(Path(__file__).parent.parent / 'config' / 'model_config.yaml')
        self.config = self._load_config()
        
        # Initialize predictors
        self.al_predictor = ALStreamPredictor(self.config['al_stream_predictor'])
        self.university_predictor = UniversityPredictor(self.config['university_predictor'])
        self.career_predictor = CareerPathPredictor(self.config['career_predictor'])
        
        logger.info("Career Guidance System initialized successfully")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load model configuration"""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _validate_ol_results(self, profile: StudentProfile) -> bool:
        """Validate if O/L results meet minimum requirements"""
        required_subjects = ['ol_mathematics', 'ol_science', 'ol_english', 'ol_sinhala', 'ol_history']
        return all(getattr(profile, subject) is not None for subject in required_subjects)
    
    def predict_al_stream(self, profile: StudentProfile) -> Dict[str, Any]:
        """Predict suitable A/L streams based on O/L results"""
        if not self._validate_ol_results(profile):
            raise ValueError("Incomplete O/L results. All core subjects are required.")
        
        # Convert profile to DataFrame
        ol_data = pd.DataFrame([{
            k: v for k, v in profile.dict().items() 
            if k.startswith('ol_') and v is not None
        }])
        
        # Get stream recommendations
        recommendations = self.al_predictor.get_stream_recommendations(ol_data)[0]
        
        return {
            'recommendations': recommendations,
            'message': "Based on your O/L results, here are your recommended A/L streams."
        }
    
    def predict_university_programs(self, profile: StudentProfile) -> Dict[str, Any]:
        """Predict suitable university programs based on O/L and A/L results"""
        if not profile.al_stream:
            raise ValueError("A/L stream is required for university program prediction")
        
        if not all(getattr(profile, f'al_subject{i}') is not None for i in range(1, 4)):
            raise ValueError("All three A/L subject results are required")
        
        # Convert profile to DataFrame
        education_data = pd.DataFrame([{
            k: v for k, v in profile.dict().items()
            if (k.startswith('ol_') or k.startswith('al_')) and v is not None
        }])
        
        # Get program recommendations
        recommendations = self.university_predictor.get_program_recommendations(education_data)[0]
        
        return {
            'recommendations': recommendations,
            'message': "Based on your O/L and A/L results, here are your recommended university programs."
        }
    
    def predict_career_paths(self, profile: StudentProfile) -> Dict[str, Any]:
        """Predict suitable career paths based on complete educational profile"""
        # Convert profile to DataFrame
        full_profile = pd.DataFrame([profile.dict()])
        
        # Get career recommendations
        recommendations = self.career_predictor.get_career_recommendations(full_profile)[0]
        
        # Enhance recommendations with educational context
        context = []
        if profile.university_completed:
            context.append("university education")
        elif profile.al_stream:
            context.append("A/L studies")
        else:
            context.append("O/L results")
            
        if profile.interests:
            context.append("personal interests")
        if profile.skills:
            context.append("current skills")
            
        return {
            'recommendations': recommendations,
            'message': f"Based on your {', '.join(context)}, here are your recommended career paths."
        }
    
    def get_comprehensive_guidance(self, profile: StudentProfile) -> Dict[str, Any]:
        """Get comprehensive career guidance based on available information"""
        guidance = {}
        
        # Stage 1: A/L Stream Prediction
        if not profile.al_stream and self._validate_ol_results(profile):
            guidance['al_streams'] = self.predict_al_stream(profile)
        
        # Stage 2: University Program Prediction
        if profile.al_stream:
            try:
                guidance['university_programs'] = self.predict_university_programs(profile)
            except ValueError as e:
                logger.warning(f"Skipping university prediction: {str(e)}")
        
        # Stage 3: Career Path Prediction
        guidance['career_paths'] = self.predict_career_paths(profile)
        
        return guidance
    
    def save_models(self, directory: str) -> None:
        """Save all models to the specified directory"""
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        
        self.al_predictor.save(str(path / 'al_predictor.joblib'))
        self.university_predictor.save(str(path / 'university_predictor.joblib'))
        self.career_predictor.save(str(path / 'career_predictor.joblib'))
        
        logger.info(f"All models saved to {directory}")
    
    def load_models(self, directory: str) -> None:
        """Load all models from the specified directory"""
        path = Path(directory)
        
        self.al_predictor.load(str(path / 'al_predictor.joblib'))
        self.university_predictor.load(str(path / 'university_predictor.joblib'))
        self.career_predictor.load(str(path / 'career_predictor.joblib'))
        
        logger.info(f"All models loaded from {directory}")
