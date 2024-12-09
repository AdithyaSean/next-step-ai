"""Configuration for dataset generation."""

# Education Level Distribution
EDUCATION_LEVEL_DIST = {
    'OL': 0.4,    # 40% O/L students
    'AL': 0.3,    # 30% A/L students
    'UNDERGRADUATE': 0.2,  # 20% undergraduates
    'GRADUATE': 0.1  # 10% graduates
}

# A/L Stream Distribution
AL_STREAM_DIST = {
    'PHYSICAL_SCIENCE': 0.25,
    'BIOLOGICAL_SCIENCE': 0.25,
    'COMMERCE': 0.3,
    'ARTS': 0.2
}

# University Field Distribution
UNIVERSITY_FIELDS = {
    'PHYSICAL_SCIENCE': [
        'Computer Science',
        'Engineering',
        'Mathematics',
        'Physics',
        'Information Technology'
    ],
    'BIOLOGICAL_SCIENCE': [
        'Medicine',
        'Dentistry',
        'Biomedical Science',
        'Agriculture',
        'Veterinary Science'
    ],
    'COMMERCE': [
        'Business Administration',
        'Finance',
        'Accounting',
        'Economics',
        'Management'
    ],
    'ARTS': [
        'Languages',
        'Social Sciences',
        'Psychology',
        'Law',
        'Education'
    ]
}

# Grade Distributions (mean, std)
GRADE_DISTRIBUTIONS = {
    'OL': {
        'mathematics': (65, 15),
        'science': (70, 12),
        'english': (75, 10),
        'first_language': (80, 8),
        'ict': (72, 13)
    },
    'AL': {
        'PHYSICAL_SCIENCE': {
            'physics': (68, 12),
            'combined_maths': (65, 15),
            'chemistry': (70, 10)
        },
        'BIOLOGICAL_SCIENCE': {
            'biology': (72, 10),
            'physics': (65, 12),
            'chemistry': (70, 10)
        },
        'COMMERCE': {
            'business_studies': (75, 8),
            'accounting': (73, 10),
            'economics': (70, 12)
        },
        'ARTS': {
            'subject1': (78, 8),
            'subject2': (75, 10),
            'subject3': (73, 12)
        }
    }
}

# Skills Distribution (mean, std) for 1-5 scale
SKILL_DISTRIBUTIONS = {
    'analytical_thinking': (3.5, 0.8),
    'problem_solving': (3.3, 0.9),
    'creativity': (3.4, 0.7),
    'communication': (3.6, 0.8)
}

# Technical Skills by Field
TECHNICAL_SKILLS = {
    'Computer Science': {
        'programming': (4.0, 0.7),
        'database_management': (3.8, 0.8),
        'web_development': (3.7, 0.9),
        'data_structures': (3.5, 0.8)
    },
    'Engineering': {
        'cad_design': (3.8, 0.7),
        'circuit_analysis': (3.7, 0.8),
        'mechanics': (3.6, 0.9),
        'materials': (3.5, 0.8)
    },
    'Medicine': {
        'clinical_skills': (4.0, 0.6),
        'patient_care': (3.9, 0.7),
        'medical_knowledge': (3.8, 0.8),
        'diagnostics': (3.7, 0.7)
    }
}

# Career Paths with Requirements
CAREER_PATHS = {
    'Software Engineer': {
        'min_ol_maths': 70,
        'min_al_maths': 75,
        'required_skills': ['analytical_thinking', 'problem_solving']
    },
    'Doctor': {
        'min_ol_science': 75,
        'min_al_biology': 80,
        'required_skills': ['analytical_thinking', 'communication']
    },
    'Business Analyst': {
        'min_ol_maths': 65,
        'min_al_accounting': 70,
        'required_skills': ['analytical_thinking', 'communication']
    },
    'Data Scientist': {
        'min_ol_maths': 75,
        'min_al_maths': 80,
        'required_skills': ['analytical_thinking', 'problem_solving']
    },
    'Biomedical Engineer': {
        'min_ol_science': 70,
        'min_al_biology': 75,
        'required_skills': ['problem_solving', 'creativity']
    },
    'Financial Analyst': {
        'min_ol_maths': 70,
        'min_al_accounting': 75,
        'required_skills': ['analytical_thinking', 'problem_solving']
    },
    'Research Scientist': {
        'min_ol_science': 80,
        'min_al_chemistry': 80,
        'required_skills': ['analytical_thinking', 'creativity']
    },
    'Teacher': {
        'min_ol_english': 70,
        'min_al_subject1': 75,
        'required_skills': ['communication', 'creativity']
    },
    'Marketing Manager': {
        'min_ol_english': 75,
        'min_al_business': 70,
        'required_skills': ['communication', 'creativity']
    },
    'Civil Engineer': {
        'min_ol_maths': 70,
        'min_al_maths': 75,
        'required_skills': ['problem_solving', 'analytical_thinking']
    }
}

# Interest Areas
INTEREST_AREAS = {
    'Technology': 0.25,
    'Healthcare': 0.2,
    'Business': 0.2,
    'Education': 0.15,
    'Research': 0.1,
    'Engineering': 0.1
}

# Work Environment Preferences
WORK_ENVIRONMENTS = {
    'Office': 0.4,
    'Remote': 0.2,
    'Field': 0.15,
    'Hospital': 0.15,
    'Laboratory': 0.1
}

# Districts with Population Distribution
DISTRICTS = {
    'Colombo': 0.3,
    'Gampaha': 0.2,
    'Kandy': 0.15,
    'Galle': 0.1,
    'Kurunegala': 0.1,
    'Jaffna': 0.08,
    'Batticaloa': 0.07
}

# Financial Constraints Distribution
FINANCIAL_CONSTRAINTS = {
    True: 0.4,  # 40% have financial constraints
    False: 0.6  # 60% don't have financial constraints
}

# GPA Distribution
GPA_DISTRIBUTION = {
    'mean': 3.2,
    'std': 0.4,
    'min': 2.0,
    'max': 4.0
}

# Project Types with Duration Ranges (in months)
PROJECT_TYPES = {
    'Research': {
        'min_duration': 3,
        'max_duration': 12
    },
    'Project': {
        'min_duration': 2,
        'max_duration': 6
    },
    'Internship': {
        'min_duration': 1,
        'max_duration': 4
    }
}

# Relocation Willingness
RELOCATION_WILLINGNESS = {
    True: 0.7,  # 70% willing to relocate
    False: 0.3
}
