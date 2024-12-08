"""Check compatibility between API models and generated data."""

import sys
from pathlib import Path
from typing import Dict, List, Any
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.api.models import (
    StudentProfile, OLResults, ALResults, UniversityData,
    CareerPreferences, WorkPreferences, CareerGoals,
    Project, Internship, TechnicalCompetencies
)
from src.data.generators import StudentDataGenerator
from src.data.validator import DataValidator

class CompatibilityChecker:
    """Check compatibility between generated data and API models."""
    
    def __init__(self):
        self.generator = StudentDataGenerator(seed=42)
        self.validator = DataValidator()
        self.issues = []
    
    def check_compatibility(self, num_samples: int = 100) -> List[str]:
        """Generate samples and check compatibility with API models."""
        self.issues = []
        
        print(f"Generating {num_samples} sample profiles...")
        samples = self.generator.generate_dataset(num_samples)
        
        print("Checking compatibility with API models...")
        for i, sample in enumerate(samples):
            try:
                # Try to create Pydantic models from generated data
                self._validate_profile(sample)
            except Exception as e:
                self.issues.append(f"Sample {i} validation failed: {str(e)}")
        
        return self.issues
    
    def _validate_profile(self, data: Dict[str, Any]):
        """Validate a single profile against API models."""
        
        # Validate O/L Results
        ol_results = OLResults(**data['ol_results'])
        
        # Validate A/L Results if present
        al_results = None
        if 'al_results' in data:
            al_results = ALResults(**data['al_results'])
        
        # Validate University Data if present
        university_data = None
        if 'university_data' in data:
            # Validate Projects
            if 'significant_projects' in data['university_data']:
                projects = [
                    Project(**proj)
                    for proj in data['university_data']['significant_projects']
                ]
            
            # Validate Internships
            if 'internships' in data['university_data']:
                internships = [
                    Internship(**intern)
                    for intern in data['university_data']['internships']
                ]
            
            # Validate Technical Competencies
            if 'technical_competencies' in data['university_data']:
                tech_comp = TechnicalCompetencies(
                    **data['university_data']['technical_competencies']
                )
            
            university_data = UniversityData(**data['university_data'])
        
        # Validate Career Preferences if present
        if 'career_preferences' in data:
            work_prefs = WorkPreferences(
                **data['career_preferences']['work_preferences']
            )
            career_goals = CareerGoals(
                **data['career_preferences']['career_goals']
            )
            career_preferences = CareerPreferences(
                **data['career_preferences']
            )
        
        # Finally, validate entire profile
        profile = StudentProfile(**data)

def main():
    checker = CompatibilityChecker()
    issues = checker.check_compatibility(100)
    
    if issues:
        print("\nFound compatibility issues:")
        for issue in issues:
            print(f"- {issue}")
    else:
        print("\nNo compatibility issues found!")
        
    # Generate sample dataset
    print("\nGenerating sample dataset...")
    sample_data = checker.generator.generate_dataset(5)
    
    # Save sample for review
    output_path = project_root / "data" / "samples"
    output_path.mkdir(exist_ok=True)
    
    with open(output_path / "sample_profiles.json", "w") as f:
        json.dump(sample_data, f, indent=2)
    
    print(f"\nSample data saved to: {output_path}/sample_profiles.json")

if __name__ == "__main__":
    main()
