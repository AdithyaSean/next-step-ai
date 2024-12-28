"""Configuration for data generation and preprocessing."""

# Education level mapping
EDUCATION_LEVELS = {"OL": 0, "AL": 1, "UNI": 2}

# Subject mappings
OL_SUBJECTS = {
    "Maths": 0,
    "Science": 1,
    "English": 2,
    "Sinhala": 3,
    "History": 4,
    "Religion": 5,
}

# AL stream and subject mappings
AL_STREAMS = {
    "Physical Science": 0,
    "Biological Science": 1,
    "Commerce": 2,
    "Arts": 3,
    "Technology": 4,
}

AL_SUBJECTS = {
    "Physics": 0,
    "Chemistry": 1,
    "Combined_Maths": 2,
    "Biology": 3,
    "Accounting": 4,
    "Business_Studies": 5,
    "Economics": 6,
    "History": 7,
    "Geography": 8,
    "Politics": 9,
    "Engineering_Tech": 10,
    "Science_Tech": 11,
    "ICT": 12,
}

# Career mappings
CAREERS = {
    "Engineering": 0,
    "Medicine": 1,
    "IT": 2,
    "Business": 3,
    "Teaching": 4,
    "Research": 5,
}

config = {
    "num_students": 5000,
    "data_dir": "./data/raw",
    "processed_dir": "./data/processed",
    "education_levels": EDUCATION_LEVELS,
    "education_level_dist": {
        EDUCATION_LEVELS["OL"]: 0.4,
        EDUCATION_LEVELS["AL"]: 0.5,
        EDUCATION_LEVELS["UNI"]: 0.1,
    },
    "ol_subjects": OL_SUBJECTS,
    "al_streams": AL_STREAMS,
    "al_subjects": {
        AL_STREAMS["Physical Science"]: [
            AL_SUBJECTS["Physics"],
            AL_SUBJECTS["Chemistry"],
            AL_SUBJECTS["Combined_Maths"],
        ],
        AL_STREAMS["Biological Science"]: [
            AL_SUBJECTS["Biology"],
            AL_SUBJECTS["Chemistry"],
            AL_SUBJECTS["Physics"],
        ],
        AL_STREAMS["Commerce"]: [
            AL_SUBJECTS["Accounting"],
            AL_SUBJECTS["Business_Studies"],
            AL_SUBJECTS["Economics"],
        ],
        AL_STREAMS["Arts"]: [
            AL_SUBJECTS["History"],
            AL_SUBJECTS["Geography"],
            AL_SUBJECTS["Politics"],
        ],
        AL_STREAMS["Technology"]: [
            AL_SUBJECTS["Engineering_Tech"],
            AL_SUBJECTS["Science_Tech"],
            AL_SUBJECTS["ICT"],
        ],
    },
    "careers": CAREERS,
    "career_success_ranges": {
        CAREERS["Engineering"]: (0.6, 0.95),
        CAREERS["Medicine"]: (0.65, 0.92),
        CAREERS["IT"]: (0.55, 0.90),
        CAREERS["Business"]: (0.50, 0.85),
        CAREERS["Teaching"]: (0.45, 0.80),
        CAREERS["Research"]: (0.58, 0.88),
    },
}
