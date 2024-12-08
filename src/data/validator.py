from typing import Dict, List, Optional
import pandas as pd

class DataValidator:
    """Validates student data according to the defined schema."""
    
    EDUCATION_LEVELS = ['OL', 'AL', 'UNDERGRADUATE', 'GRADUATE', 'MASTERS', 'PHD']
    STREAMS = ['PHYSICAL_SCIENCE', 'BIOLOGICAL_SCIENCE', 'COMMERCE', 'ARTS']
    SKILL_RANGE = range(1, 6)  # 1-5
    GRADE_RANGE = range(0, 101)  # 0-100
    GPA_RANGE = (0.0, 4.0)
    
    REQUIRED_OL_SUBJECTS = ['mathematics', 'science', 'english']
    
    DEGREE_TYPES = ['Bachelors', 'Masters', 'PhD']
    ROLE_TYPES = ['Technical', 'Research', 'Management']
    PROJECT_TYPES = ['Research', 'Project', 'Internship']
    
    def __init__(self):
        self.errors = []
    
    def validate_student_profile(self, data: Dict) -> bool:
        """
        Validates a complete student profile.
        Returns True if valid, False if invalid.
        """
        self.errors = []
        
        # Check required fields
        required_fields = ['student_id', 'education_level', 'academic_data', 
                         'skills_data', 'interests_data']
        for field in required_fields:
            if field not in data:
                self.errors.append(f"Missing required field: {field}")
                return False
        
        # Validate each section
        validations = [
            self._validate_education_level(data['education_level']),
            self._validate_academic_data(data['academic_data'], data['education_level']),
            self._validate_skills_data(data['skills_data']),
            self._validate_interests_data(data['interests_data'])
        ]
        
        if 'constraints' in data:
            validations.append(self._validate_constraints(data['constraints']))
        
        if 'career_preferences' in data:
            validations.append(self._validate_career_preferences(data['career_preferences']))
        
        return all(validations)
    
    def _validate_education_level(self, level: str) -> bool:
        """Validates education level."""
        if level not in self.EDUCATION_LEVELS:
            self.errors.append(f"Invalid education level: {level}")
            return False
        return True
    
    def _validate_academic_data(self, data: Dict, education_level: str) -> bool:
        """Validates academic data based on education level."""
        if 'ol_results' not in data:
            self.errors.append("Missing O/L results")
            return False
        
        # Validate O/L results
        ol_data = data['ol_results']
        for subject in self.REQUIRED_OL_SUBJECTS:
            if subject not in ol_data:
                self.errors.append(f"Missing required O/L subject: {subject}")
                return False
            if not isinstance(ol_data[subject], (int, float)):
                self.errors.append(f"Invalid grade type for {subject}")
                return False
            if ol_data[subject] not in self.GRADE_RANGE:
                self.errors.append(f"Invalid grade range for {subject}")
                return False
        
        # Validate A/L data if applicable
        if education_level in ['AL', 'UNDERGRADUATE', 'GRADUATE']:
            if 'al_stream' not in data:
                self.errors.append("Missing A/L stream")
                return False
            if data['al_stream'] not in self.STREAMS:
                self.errors.append(f"Invalid A/L stream: {data['al_stream']}")
                return False
            
            if 'al_results' in data:
                # Validate stream-specific subjects
                stream_data = data['al_results'].get(data['al_stream'].lower(), {})
                if not stream_data:
                    self.errors.append(f"Missing A/L results for stream: {data['al_stream']}")
                    return False
                
                # Validate grades
                for subject, grade in stream_data.items():
                    if not isinstance(grade, (int, float)):
                        self.errors.append(f"Invalid grade type for {subject}")
                        return False
                    if grade not in self.GRADE_RANGE:
                        self.errors.append(f"Invalid grade range for {subject}")
                        return False
        
        # Validate university data if applicable
        if education_level in ['UNDERGRADUATE', 'GRADUATE', 'MASTERS', 'PHD']:
            if 'university_data' not in data:
                self.errors.append("Missing university data")
                return False
                
            return self._validate_university_data(data['university_data'])
        
        return True
    
    def _validate_university_data(self, data: Dict) -> bool:
        """Validates university-specific data."""
        # Validate core information
        if 'degree_type' not in data or data['degree_type'] not in self.DEGREE_TYPES:
            self.errors.append("Invalid or missing degree type")
            return False
            
        if 'current_year' not in data or not isinstance(data['current_year'], int):
            self.errors.append("Invalid or missing current year")
            return False
            
        if 'field_of_study' not in data or not data['field_of_study']:
            self.errors.append("Missing field of study")
            return False
            
        # Validate GPA
        if 'current_gpa' in data:
            gpa = data['current_gpa']
            if not isinstance(gpa, float) or not self.GPA_RANGE[0] <= gpa <= self.GPA_RANGE[1]:
                self.errors.append("Invalid GPA value")
                return False
        
        # Validate technical competencies
        if 'technical_competencies' in data:
            for skill, rating in data['technical_competencies'].items():
                if rating not in self.SKILL_RANGE:
                    self.errors.append(f"Invalid rating for technical skill: {skill}")
                    return False
        
        # Validate projects
        if 'significant_projects' in data:
            for project in data['significant_projects']:
                if 'type' not in project or project['type'] not in self.PROJECT_TYPES:
                    self.errors.append("Invalid project type")
                    return False
                if 'duration_months' not in project or not isinstance(project['duration_months'], int):
                    self.errors.append("Invalid project duration")
                    return False
        
        # Validate internships
        if 'internships' in data:
            for internship in data['internships']:
                if 'role_type' not in internship or internship['role_type'] not in self.ROLE_TYPES:
                    self.errors.append("Invalid internship role type")
                    return False
                if 'duration_months' not in internship or not isinstance(internship['duration_months'], int):
                    self.errors.append("Invalid internship duration")
                    return False
        
        return True
    
    def _validate_skills_data(self, data: Dict) -> bool:
        """Validates skills assessment data."""
        required_skills = ['analytical_thinking', 'problem_solving', 
                         'creativity', 'communication']
        
        for skill in required_skills:
            if skill not in data:
                self.errors.append(f"Missing required skill: {skill}")
                return False
            if data[skill] not in self.SKILL_RANGE:
                self.errors.append(f"Invalid skill rating for {skill}")
                return False
        
        return True
    
    def _validate_interests_data(self, data: Dict) -> bool:
        """Validates interests data."""
        if 'primary_interests' not in data:
            self.errors.append("Missing primary interests")
            return False
        
        interests = data['primary_interests']
        if not isinstance(interests, list):
            self.errors.append("Primary interests must be a list")
            return False
        
        if len(interests) != 3:
            self.errors.append("Must specify exactly 3 primary interests")
            return False
        
        return True
    
    def _validate_constraints(self, data: Dict) -> bool:
        """Validates optional constraints data."""
        if 'preferred_location' in data and not isinstance(data['preferred_location'], str):
            self.errors.append("Invalid preferred location format")
            return False
        
        if 'financial_constraints' in data and not isinstance(data['financial_constraints'], bool):
            self.errors.append("Invalid financial constraints format")
            return False
        
        if 'willing_to_relocate' in data and not isinstance(data['willing_to_relocate'], bool):
            self.errors.append("Invalid willing to relocate format")
            return False
        
        return True
    
    def _validate_career_preferences(self, data: Dict) -> bool:
        """Validates career preferences data."""
        if not isinstance(data.get('preferred_roles', []), list) or len(data.get('preferred_roles', [])) != 3:
            self.errors.append("Must specify exactly 3 preferred roles")
            return False
            
        if not isinstance(data.get('preferred_sectors', []), list) or len(data.get('preferred_sectors', [])) != 3:
            self.errors.append("Must specify exactly 3 preferred sectors")
            return False
            
        # Validate work preferences
        work_prefs = data.get('work_preferences', {})
        required_prefs = ['research_oriented', 'industry_oriented', 'entrepreneurship_interest']
        for pref in required_prefs:
            if pref not in work_prefs or not isinstance(work_prefs[pref], bool):
                self.errors.append(f"Invalid or missing work preference: {pref}")
                return False
        
        # Validate career goals
        goals = data.get('career_goals', {})
        required_goals = ['further_studies', 'industry_experience', 'startup_plans']
        for goal in required_goals:
            if goal not in goals or not isinstance(goals[goal], bool):
                self.errors.append(f"Invalid or missing career goal: {goal}")
                return False
        
        return True
    
    def get_errors(self) -> List[str]:
        """Returns list of validation errors."""
        return self.errors
