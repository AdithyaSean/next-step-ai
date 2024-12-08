"""Generate realistic student data for the career guidance system."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import random
from datetime import datetime, timedelta
import uuid
from . import config

class StudentDataGenerator:
    """Generate realistic student data following Sri Lankan education patterns."""
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize the generator with optional seed for reproducibility."""
        self.rng = np.random.default_rng(seed)
        self.student_count = 0
    
    def generate_ol_results(self) -> Dict:
        """Generate O/L results following realistic grade distributions."""
        results = {}
        for subject, (mean, std) in config.GRADE_DISTRIBUTIONS['OL'].items():
            grade = self.rng.normal(mean, std)
            results[subject] = min(max(round(grade, 2), 0), 100)
        
        results['total_subjects_passed'] = sum(1 for grade in results.values() if grade >= 35)
        results['core_subjects_average'] = np.mean([
            results['mathematics'],
            results['science'],
            results['english']
        ])
        return results
    
    def generate_al_results(self, stream: str) -> Dict:
        """Generate A/L results for given stream."""
        results = {'stream': stream, 'subjects': {}}
        
        for subject, (mean, std) in config.GRADE_DISTRIBUTIONS['AL'][stream].items():
            grade = self.rng.normal(mean, std)
            results['subjects'][subject] = min(max(round(grade, 2), 0), 100)
        
        # Generate realistic z-score
        results['zscore'] = round(self.rng.normal(1.2, 0.6), 2)
        return results
    
    def generate_skills_assessment(self) -> Dict[str, int]:
        """Generate skill ratings."""
        skills = {}
        for skill, (mean, std) in config.SKILL_DISTRIBUTIONS.items():
            rating = round(self.rng.normal(mean, std))
            skills[skill] = min(max(rating, 1), 5)
        return skills
    
    def generate_university_data(self, stream: str) -> Dict:
        """Generate university-related data."""
        field = random.choice(config.UNIVERSITY_FIELDS[stream])
        
        # Generate GPA
        gpa = self.rng.normal(
            config.GPA_DISTRIBUTION['mean'],
            config.GPA_DISTRIBUTION['std']
        )
        gpa = min(max(round(gpa, 2), 
                     config.GPA_DISTRIBUTION['min']),
                 config.GPA_DISTRIBUTION['max'])
        
        # Generate technical skills
        tech_skills = {}
        if field in config.TECHNICAL_SKILLS:
            for skill, (mean, std) in config.TECHNICAL_SKILLS[field].items():
                rating = round(self.rng.normal(mean, std))
                tech_skills[skill] = min(max(rating, 1), 5)
        
        # Generate projects
        num_projects = self.rng.integers(1, 4)
        projects = []
        for _ in range(num_projects):
            proj_type = random.choice(list(config.PROJECT_TYPES.keys()))
            duration = self.rng.integers(
                config.PROJECT_TYPES[proj_type]['min_duration'],
                config.PROJECT_TYPES[proj_type]['max_duration']
            )
            projects.append({
                'type': proj_type,
                'domain': field,
                'duration_months': duration
            })
        
        return {
            'degree_type': 'Bachelors',
            'current_year': self.rng.integers(1, 5),
            'field_of_study': field,
            'specialization': random.choice(config.UNIVERSITY_FIELDS[stream]),
            'current_gpa': gpa,
            'technical_competencies': tech_skills,
            'significant_projects': projects,
            'internships': self._generate_internships(field)
        }
    
    def _generate_internships(self, field: str) -> List[Dict]:
        """Generate internship experiences."""
        num_internships = self.rng.integers(0, 3)
        internships = []
        
        for _ in range(num_internships):
            internships.append({
                'field': field,
                'duration_months': self.rng.integers(1, 7),
                'role_type': random.choice(['Technical', 'Research', 'Management'])
            })
        
        return internships
    
    def generate_career_preferences(self) -> Dict:
        """Generate career preferences."""
        # Select random roles and sectors
        all_paths = list(config.CAREER_PATHS.keys())
        preferred_roles = random.sample(all_paths, min(3, len(all_paths)))
        
        all_areas = list(config.INTEREST_AREAS.keys())
        preferred_sectors = random.sample(all_areas, min(3, len(all_areas)))
        
        return {
            'preferred_roles': preferred_roles,
            'preferred_sectors': preferred_sectors,
            'work_preferences': {
                'research_oriented': random.choice([True, False]),
                'industry_oriented': random.choice([True, False]),
                'entrepreneurship_interest': random.choice([True, False])
            },
            'career_goals': {
                'further_studies': random.choice([True, False]),
                'industry_experience': random.choice([True, False]),
                'startup_plans': random.choice([True, False])
            }
        }
    
    def generate_constraints(self) -> Dict:
        """Generate student constraints."""
        return {
            'preferred_location': np.random.choice(
                list(config.DISTRICTS.keys()),
                p=list(config.DISTRICTS.values())
            ),
            'financial_constraints': np.random.choice(
                [True, False],
                p=[config.FINANCIAL_CONSTRAINTS[True], 
                   config.FINANCIAL_CONSTRAINTS[False]]
            ),
            'willing_to_relocate': np.random.choice(
                [True, False],
                p=[config.RELOCATION_WILLINGNESS[True],
                   config.RELOCATION_WILLINGNESS[False]]
            )
        }
    
    def generate_student(self) -> Dict:
        """Generate a complete student profile."""
        self.student_count += 1
        
        # Determine education level
        education_level = np.random.choice(
            list(config.EDUCATION_LEVEL_DIST.keys()),
            p=list(config.EDUCATION_LEVEL_DIST.values())
        )
        
        # Generate base profile
        profile = {
            'student_id': f"ST{str(uuid.uuid4())[:8]}",
            'education_level': education_level,
            'ol_results': self.generate_ol_results(),
            'skills_assessment': self.generate_skills_assessment(),
            'interests': random.sample(list(config.INTEREST_AREAS.keys()), 3),
            'constraints': self.generate_constraints()
        }
        
        # Add A/L results if applicable
        if education_level in ['AL', 'UNDERGRADUATE', 'GRADUATE']:
            stream = np.random.choice(
                list(config.AL_STREAM_DIST.keys()),
                p=list(config.AL_STREAM_DIST.values())
            )
            profile['al_results'] = self.generate_al_results(stream)
            
            # Add university data if applicable
            if education_level in ['UNDERGRADUATE', 'GRADUATE']:
                profile['university_data'] = self.generate_university_data(stream)
                profile['career_preferences'] = self.generate_career_preferences()
        
        return profile
    
    def generate_dataset(self, size: int) -> List[Dict]:
        """Generate a dataset of given size."""
        return [self.generate_student() for _ in range(size)]
    
    def save_dataset(self, size: int, output_path: str):
        """Generate and save dataset to JSON and CSV formats."""
        dataset = self.generate_dataset(size)
        
        # Save as JSON
        json_path = f"{output_path}/student_data_{size}.json"
        pd.DataFrame(dataset).to_json(json_path, orient='records', indent=2)
        
        # Save as CSV (flattened)
        csv_path = f"{output_path}/student_data_{size}.csv"
        df = pd.json_normalize(dataset)
        df.to_csv(csv_path, index=False)
        
        return json_path, csv_path
