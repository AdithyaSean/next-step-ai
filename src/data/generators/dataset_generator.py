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
            results[subject] = float(min(max(round(grade, 2), 0), 100))
        
        results['total_subjects_passed'] = int(sum(1 for grade in results.values() if grade >= 35))
        results['core_subjects_average'] = float(np.mean([
            results['mathematics'],
            results['science'],
            results['english']
        ]))
        return results
    
    def generate_al_results(self, stream: str) -> Dict:
        """Generate A/L results for given stream."""
        results = {'stream': stream, 'subjects': {}}
        
        for subject, (mean, std) in config.GRADE_DISTRIBUTIONS['AL'][stream].items():
            grade = self.rng.normal(mean, std)
            results['subjects'][subject] = float(min(max(round(grade, 2), 0), 100))
        
        # Generate realistic z-score
        results['zscore'] = float(round(self.rng.normal(1.2, 0.6), 2))
        return results
    
    def generate_skills_assessment(self) -> Dict[str, float]:
        """Generate skill ratings."""
        skills = {}
        for skill, (mean, std) in config.SKILL_DISTRIBUTIONS.items():
            rating = round(self.rng.normal(mean, std))
            skills[skill] = float(min(max(rating, 1), 5))
        return skills
    
    def generate_university_data(self, stream: str) -> Dict:
        """Generate university-related data."""
        field = random.choice(config.UNIVERSITY_FIELDS[stream])
        
        # Generate GPA
        gpa = self.rng.normal(
            config.GPA_DISTRIBUTION['mean'],
            config.GPA_DISTRIBUTION['std']
        )
        gpa = float(min(max(round(gpa, 2), 
                     config.GPA_DISTRIBUTION['min']),
                 config.GPA_DISTRIBUTION['max']))
        
        # Generate technical skills
        tech_skills = {}
        if field in config.TECHNICAL_SKILLS:
            for skill, (mean, std) in config.TECHNICAL_SKILLS[field].items():
                rating = round(self.rng.normal(mean, std))
                tech_skills[skill] = float(min(max(rating, 1), 5))
        
        # Generate projects
        num_projects = self.rng.integers(1, 4)
        projects = []
        for _ in range(num_projects):
            proj_type = random.choice(list(config.PROJECT_TYPES.keys()))
            duration = int(self.rng.integers(
                config.PROJECT_TYPES[proj_type]['min_duration'],
                config.PROJECT_TYPES[proj_type]['max_duration']
            ))
            projects.append({
                'type': proj_type,
                'domain': field,
                'duration_months': duration
            })
        
        return {
            'degree_type': 'Bachelors',
            'current_year': int(self.rng.integers(1, 5)),
            'field_of_study': field,
            'specialization': random.choice(config.UNIVERSITY_FIELDS[stream]),
            'current_gpa': gpa,
            'technical_competencies': tech_skills,
            'significant_projects': projects,
            'internships': self._generate_internships(field)
        }
    
    def _generate_internships(self, field: str) -> List[Dict]:
        """Generate internship experiences."""
        num_internships = int(self.rng.integers(0, 3))
        internships = []
        
        for _ in range(num_internships):
            internships.append({
                'field': field,
                'duration_months': int(self.rng.integers(1, 7)),
                'role_type': random.choice(['Technical', 'Research', 'Management'])
            })
        
        return internships
    
    def determine_career_path(self, ol_results: Dict, al_results: Optional[Dict], skills: Dict[str, float]) -> str:
        """Determine the most suitable career path based on academic performance and skills."""
        career_scores = {}
        
        for career, requirements in config.CAREER_PATHS.items():
            score = 0.0
            
            # Check O/L requirements
            if career == 'Software Engineer' and ol_results['mathematics'] >= requirements['min_ol_maths']:
                score += 0.3
            elif career == 'Doctor' and ol_results['science'] >= requirements['min_ol_science']:
                score += 0.3
            elif career == 'Business Analyst' and ol_results['mathematics'] >= requirements['min_ol_maths']:
                score += 0.3
            
            # Check A/L stream and results if available
            if al_results is not None:
                if career == 'Software Engineer':
                    if (al_results['stream'] == 'PHYSICAL_SCIENCE' and 
                        'combined_maths' in al_results['subjects'] and 
                        al_results['subjects']['combined_maths'] >= requirements['min_al_maths']):
                        score += 0.4
                elif career == 'Doctor':
                    if (al_results['stream'] == 'BIOLOGICAL_SCIENCE' and 
                        'biology' in al_results['subjects'] and 
                        al_results['subjects']['biology'] >= requirements['min_al_biology']):
                        score += 0.4
                elif career == 'Business Analyst':
                    if (al_results['stream'] == 'COMMERCE' and 
                        'accounting' in al_results['subjects'] and 
                        al_results['subjects']['accounting'] >= requirements['min_al_accounting']):
                        score += 0.4
            else:
                # For O/L only students, give partial credit for being in the right academic track
                if career == 'Software Engineer' and ol_results['mathematics'] >= 75:
                    score += 0.2
                elif career == 'Doctor' and ol_results['science'] >= 85:
                    score += 0.2
                elif career == 'Business Analyst' and ol_results['mathematics'] >= 70:
                    score += 0.2
            
            # Check skills match
            required_skills = requirements['required_skills']
            matching_skills = sum(1 for skill in required_skills if skill in skills and skills[skill] >= 3.5)
            score += (matching_skills / len(required_skills)) * 0.3
            
            career_scores[career] = score
        
        # Return career with highest score, defaulting to most general if no strong matches
        best_career = max(career_scores.items(), key=lambda x: x[1])
        return best_career[0] if best_career[1] >= 0.5 else 'General Studies'
    
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
                'research_oriented': int(random.choice([1, 0])),
                'industry_oriented': int(random.choice([1, 0])),
                'entrepreneurship_interest': int(random.choice([1, 0]))
            },
            'career_goals': {
                'further_studies': int(random.choice([1, 0])),
                'industry_experience': int(random.choice([1, 0])),
                'startup_plans': int(random.choice([1, 0]))
            }
        }
    
    def generate_constraints(self) -> Dict:
        """Generate student constraints."""
        return {
            'preferred_location': np.random.choice(
                list(config.DISTRICTS.keys()),
                p=list(config.DISTRICTS.values())
            ),
            'financial_constraints': int(np.random.choice(
                [1, 0],
                p=[config.FINANCIAL_CONSTRAINTS[True], 
                   config.FINANCIAL_CONSTRAINTS[False]]
            )),
            'willing_to_relocate': int(np.random.choice(
                [1, 0],
                p=[config.RELOCATION_WILLINGNESS[True],
                   config.RELOCATION_WILLINGNESS[False]]
            ))
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
            'constraints': self.generate_constraints(),
            'al_results': None  # Initialize as None by default
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
        
        # Determine career path
        profile['career_path'] = self.determine_career_path(
            profile['ol_results'],
            profile['al_results'],
            profile['skills_assessment']
        )
        
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
