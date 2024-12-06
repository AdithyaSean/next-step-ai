"""
Generate sample data for testing the career guidance system
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Define sample data characteristics
SAMPLE_SIZE = 1000

# Academic subjects
OL_SUBJECTS = ['Mathematics', 'Science', 'English', 'History']
AL_STREAMS = ['Science', 'Commerce', 'Arts']
INTERESTS = ['Technology', 'Business', 'Healthcare', 'Teaching', 'Arts']
SKILLS = ['Programming', 'Communication', 'Analysis', 'Leadership', 'Creativity']

# Career paths (target variables)
CAREER_PATHS = [
    'Software Engineer',
    'Data Scientist',
    'Business Analyst',
    'Doctor',
    'Teacher'
]

def generate_sample_data():
    """Generate synthetic student data"""
    
    np.random.seed(42)
    
    data = {
        # Generate OL results (grades between 0-100)
        **{f"ol_{subject.lower()}": np.random.randint(40, 100, SAMPLE_SIZE) 
           for subject in OL_SUBJECTS},
        
        # Generate AL stream
        'al_stream': np.random.choice(AL_STREAMS, SAMPLE_SIZE),
        
        # Generate interests (1-3 interests per student)
        'interests': [
            ', '.join(np.random.choice(INTERESTS, np.random.randint(1, 4), replace=False))
            for _ in range(SAMPLE_SIZE)
        ],
        
        # Generate skills (1-3 skills per student)
        'skills': [
            ', '.join(np.random.choice(SKILLS, np.random.randint(1, 4), replace=False))
            for _ in range(SAMPLE_SIZE)
        ],
        
        # Generate career paths (target)
        'career_path': np.random.choice(CAREER_PATHS, SAMPLE_SIZE)
    }
    
    return pd.DataFrame(data)

def main():
    # Create data directory if it doesn't exist
    data_dir = Path('data/raw')
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate and save data
    df = generate_sample_data()
    df.to_csv(data_dir / 'sample_data.csv', index=False)
    
    print(f"Generated {len(df)} sample records in data/raw/sample_data.csv")
    print("\nSample data preview:")
    print(df.head())
    print("\nData statistics:")
    print(df.describe())
    
if __name__ == '__main__':
    main()
