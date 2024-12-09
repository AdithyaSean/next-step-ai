"""Test script for career predictor model."""

import sys
import os
from pathlib import Path
import joblib
import numpy as np

# Add src to Python path
src_path = str(Path(__file__).parent.parent / "src")
if src_path not in sys.path:
    sys.path.append(src_path)

from models.career_predictor import CareerPredictor
from data.preprocessor import DataPreprocessor
from api.models import (
    StudentProfile, OLResults, ALResults, Stream, 
    UniversityData, CareerPreferences, WorkPreferences,
    CareerGoals
)

def create_sample_profile():
    """Create a sample student profile for testing."""
    return {
        "student_id": "TEST001",
        "education_level": "AL",  
        "ol_results": {
            "mathematics": 85.0,
            "science": 78.0,
            "english": 82.0,
            "first_language": 75.0,
            "ict": 90.0,
            "total_subjects_passed": 8,
            "core_subjects_average": 81.67
        },
        "al_results": {
            "stream": "PHYSICAL_SCIENCE",
            "subjects": {
                "physics": 75.0,
                "chemistry": 82.0,
                "combined_maths": 88.0
            },
            "zscore": 1.8523
        },
        "university_data": {
            "degree_type": "BSc",
            "field_of_study": "Computer Science",
            "current_year": 2,
            "current_gpa": 3.8,
            "specialization": "Artificial Intelligence",
            "significant_projects": [
                "Machine Learning Project",
                "Web Development Project"
            ],
            "internships": [
                "Software Engineering Intern"
            ]
        },
        "skills_assessment": {
            "programming": 4,
            "problem_solving": 4,
            "communication": 3,
            "teamwork": 4,
            "analytical": 5
        },
        "interests": [
            "technology",
            "science",
            "mathematics"
        ],
        "career_preferences": {
            "preferred_roles": ["Software Engineer", "Data Scientist", "AI Engineer"],
            "preferred_sectors": ["Technology", "Research", "Education"],
            "work_preferences": {
                "research_oriented": True,
                "industry_oriented": True,
                "entrepreneurship_interest": False
            },
            "career_goals": {
                "further_studies": True,
                "industry_experience": True,
                "startup_plans": False
            }
        },
        "constraints": {
            "location": ["local"],
            "financial": ["moderate"],
            "time_commitment": ["full_time"]
        }
    }

def load_preprocessor(preprocessor_dir: Path) -> DataPreprocessor:
    """Load the preprocessor with saved encoders and scalers."""
    preprocessor = DataPreprocessor()
    
    # Load all saved preprocessor components
    preprocessor.grade_scaler = joblib.load(preprocessor_dir / 'grade_scaler.joblib')
    preprocessor.zscore_scaler = joblib.load(preprocessor_dir / 'zscore_scaler.joblib')
    preprocessor.skill_scaler = joblib.load(preprocessor_dir / 'skill_scaler.joblib')
    preprocessor.stream_encoder = joblib.load(preprocessor_dir / 'stream_encoder.joblib')
    preprocessor.field_encoder = joblib.load(preprocessor_dir / 'field_encoder.joblib')
    preprocessor.interest_encoder = joblib.load(preprocessor_dir / 'interest_encoder.joblib')
    preprocessor.career_encoder = joblib.load(preprocessor_dir / 'career_encoder.joblib')
    preprocessor.environment_encoder = joblib.load(preprocessor_dir / 'environment_encoder.joblib')
    
    preprocessor.is_fitted = True
    return preprocessor

def main():
    # Initialize the model
    print("Initializing career predictor...")
    
    # Load saved model and preprocessors
    model_dir = Path(__file__).parent.parent / "models" / "saved"
    latest_model_dir = sorted(list(model_dir.glob("model_*")))[-1]
    preprocessor_dir = model_dir / "preprocessor"
    
    print(f"\nLoading model from: {latest_model_dir}")
    print(f"Loading preprocessors from: {preprocessor_dir}")
    
    try:
        predictor = CareerPredictor.load(str(latest_model_dir))
        preprocessor = load_preprocessor(preprocessor_dir)
    except Exception as e:
        print(f"Error loading model or preprocessor: {str(e)}")
        return
    
    # Create sample data
    print("\nCreating sample student profile...")
    profile_data = create_sample_profile()
    
    # Preprocess the profile
    print("\nPreprocessing profile...")
    try:
        features_df = preprocessor.transform([profile_data])
        print(f"Preprocessed features shape: {features_df.shape}")
    except Exception as e:
        print(f"Error preprocessing data: {str(e)}")
        return
    
    # Make prediction
    print("\nMaking career prediction...")
    try:
        raw_predictions = predictor.predict(features_df)
        
        # Get top career paths
        top_indices = np.argsort(raw_predictions[0])[::-1][:3]
        career_paths = preprocessor.career_encoder.inverse_transform(top_indices)
        probabilities = raw_predictions[0][top_indices]
        
        print("\nPrediction Results:")
        for path, prob in zip(career_paths, probabilities):
            print(f"\nCareer Path: {path}")
            print(f"Confidence: {prob:.2f}")
            
            # Add some generic advice based on the career path
            if "software" in path.lower() or "developer" in path.lower():
                print("\nRecommendations:")
                print("- Focus on strengthening programming skills")
                print("- Consider learning popular frameworks")
                print("- Build a portfolio of projects")
            elif "data" in path.lower() or "analyst" in path.lower():
                print("\nRecommendations:")
                print("- Strengthen statistical knowledge")
                print("- Learn data visualization tools")
                print("- Practice with real-world datasets")
            elif "research" in path.lower() or "scientist" in path.lower():
                print("\nRecommendations:")
                print("- Focus on research methodologies")
                print("- Consider advanced degrees")
                print("- Stay updated with latest papers")
                
    except Exception as e:
        print(f"\nError making prediction: {str(e)}")

if __name__ == "__main__":
    main()
