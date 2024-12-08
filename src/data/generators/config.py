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

# Interest Areas with Weights
INTEREST_AREAS = {
    'Technology': 0.2,
    'Healthcare': 0.15,
    'Business': 0.2,
    'Engineering': 0.15,
    'Education': 0.1,
    'Science': 0.1,
    'Arts': 0.05,
    'Law': 0.05
}

# Career Paths with Required Skills
CAREER_PATHS = {
    'Software Engineer': {
        'required_skills': ['programming', 'problem_solving', 'analytical_thinking'],
        'preferred_streams': ['PHYSICAL_SCIENCE'],
        'min_ol_maths': 65,
        'min_al_maths': 70
    },
    'Doctor': {
        'required_skills': ['analytical_thinking', 'communication', 'clinical_skills'],
        'preferred_streams': ['BIOLOGICAL_SCIENCE'],
        'min_ol_science': 75,
        'min_al_biology': 75
    },
    'Business Analyst': {
        'required_skills': ['analytical_thinking', 'communication', 'business_knowledge'],
        'preferred_streams': ['COMMERCE'],
        'min_ol_maths': 60,
        'min_al_accounting': 70
    }
}

# Project Types with Durations
PROJECT_TYPES = {
    'Research': {'min_duration': 3, 'max_duration': 12},
    'Development': {'min_duration': 2, 'max_duration': 6},
    'Analysis': {'min_duration': 1, 'max_duration': 4}
}

# Location Distribution
DISTRICTS = {
    'Colombo': 0.25,
    'Gampaha': 0.15,
    'Kandy': 0.12,
    'Galle': 0.08,
    'Kurunegala': 0.07,
    'Other': 0.33
}

# Financial Constraints Distribution
FINANCIAL_CONSTRAINTS = {
    True: 0.4,  # 40% have financial constraints
    False: 0.6
}

# Relocation Willingness
RELOCATION_WILLINGNESS = {
    True: 0.7,  # 70% willing to relocate
    False: 0.3
}

# GPA Distribution
GPA_DISTRIBUTION = {
    'mean': 3.2,
    'std': 0.4,
    'min': 2.0,
    'max': 4.0
}
