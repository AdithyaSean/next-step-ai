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
            ol_subjects = {
                'mathematics': 'min_ol_maths',
                'science': 'min_ol_science',
                'english': 'min_ol_english'
            }
            
            for subject, req_key in ol_subjects.items():
                if req_key in requirements and ol_results[subject] >= requirements[req_key]:
                    score += 0.3
            
            # Check A/L stream and results if available
            if al_results is not None:
                al_subjects = {
                    'PHYSICAL_SCIENCE': {
                        'combined_maths': 'min_al_maths',
                        'physics': 'min_al_physics',
                        'chemistry': 'min_al_chemistry'
                    },
                    'BIOLOGICAL_SCIENCE': {
                        'biology': 'min_al_biology',
                        'chemistry': 'min_al_chemistry',
                        'physics': 'min_al_physics'
                    },
                    'COMMERCE': {
                        'business_studies': 'min_al_business',
                        'accounting': 'min_al_accounting',
                        'economics': 'min_al_economics'
                    },
                    'ARTS': {
                        'subject1': 'min_al_subject1',
                        'subject2': 'min_al_subject2',
                        'subject3': 'min_al_subject3'
                    }
                }
                
                stream = al_results['stream']
                if stream in al_subjects:
                    for subject, req_key in al_subjects[stream].items():
                        if (req_key in requirements and 
                            subject in al_results['subjects'] and 
                            al_results['subjects'][subject] >= requirements[req_key]):
                            score += 0.4
            else:
                # For O/L only students, give partial credit for being in the right academic track
                if ('min_ol_maths' in requirements and 
                    ol_results['mathematics'] >= requirements['min_ol_maths'] + 5):
                    score += 0.2
                elif ('min_ol_science' in requirements and 
                      ol_results['science'] >= requirements['min_ol_science'] + 5):
                    score += 0.2
                elif ('min_ol_english' in requirements and 
                      ol_results['english'] >= requirements['min_ol_english'] + 5):
                    score += 0.2
            
            # Check skills match
            required_skills = requirements['required_skills']
            matching_skills = sum(1 for skill in required_skills 
                                if skill in skills and skills[skill] >= 3.5)
            score += (matching_skills / len(required_skills)) * 0.3
            
            # Add significant randomness to ensure diversity (±15%)
            score *= (1 + self.rng.uniform(-0.15, 0.15))
            
            career_scores[career] = score
        
        # Get top 3 careers by score
        top_careers = sorted(career_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        
        # If any careers have a good score, randomly select from top 3
        good_careers = [(career, score) for career, score in top_careers if score >= 0.5]
        if good_careers:
            # Weight selection by scores
            careers, scores = zip(*good_careers)
            total_score = sum(scores)
            probabilities = [score/total_score for score in scores]
            return str(np.random.choice(careers, p=probabilities))
        
        return 'General Studies'
    
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
