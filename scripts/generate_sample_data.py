"""
Generate realistic sample data for the career guidance system including complete educational journey
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Define sample data characteristics
SAMPLE_SIZE = 1000

# O/L Subjects and passing criteria
OL_SUBJECTS = {
    'Mathematics': {'weight': 1.0},
    'Science': {'weight': 1.0},
    'English': {'weight': 1.0},
    'Sinhala': {'weight': 0.8},
    'History': {'weight': 0.8},
    'Religion': {'weight': 0.7},
    'Literature': {'weight': 0.7},
    'ICT': {'weight': 0.8},
    'Commerce': {'weight': 0.8}
}

# A/L Streams and subjects
AL_STREAMS = {
    'Science': {
        'subjects': ['Physics', 'Chemistry', 'Biology/ICT/Combined Maths'],
        'required_ol_grades': {'Mathematics': 65, 'Science': 65, 'English': 50},
        'zscore_threshold': 1.2
    },
    'Commerce': {
        'subjects': ['Business Studies', 'Accounting', 'Economics'],
        'required_ol_grades': {'Mathematics': 50, 'English': 50, 'Commerce': 65},
        'zscore_threshold': 1.0
    },
    'Arts': {
        'subjects': ['Geography', 'Political Science', 'Languages'],
        'required_ol_grades': {'Sinhala': 50, 'History': 50},
        'zscore_threshold': 0.8
    }
}

# University Programs
UNIVERSITY_PROGRAMS = {
    'Science': [
        {'name': 'Engineering', 'zscore_min': 1.8, 'dropout_rate': 0.1},
        {'name': 'Medicine', 'zscore_min': 2.0, 'dropout_rate': 0.05},
        {'name': 'Computer Science', 'zscore_min': 1.6, 'dropout_rate': 0.15},
        {'name': 'Physical Science', 'zscore_min': 1.4, 'dropout_rate': 0.2}
    ],
    'Commerce': [
        {'name': 'Business Administration', 'zscore_min': 1.5, 'dropout_rate': 0.12},
        {'name': 'Finance', 'zscore_min': 1.6, 'dropout_rate': 0.1},
        {'name': 'Economics', 'zscore_min': 1.4, 'dropout_rate': 0.15},
        {'name': 'Management', 'zscore_min': 1.3, 'dropout_rate': 0.18}
    ],
    'Arts': [
        {'name': 'Psychology', 'zscore_min': 1.3, 'dropout_rate': 0.15},
        {'name': 'Sociology', 'zscore_min': 1.2, 'dropout_rate': 0.2},
        {'name': 'Languages', 'zscore_min': 1.1, 'dropout_rate': 0.18},
        {'name': 'Political Science', 'zscore_min': 1.0, 'dropout_rate': 0.22}
    ]
}

# Career paths based on education level and program
CAREER_PATHS = {
    'ol_dropout': [
        'Skilled Labor', 'Small Business Owner', 'Sales Representative', 
        'Technical Training', 'Vocational Training'
    ],
    'al_dropout': [
        'Administrative Assistant', 'Customer Service', 'Sales Manager',
        'Technical Specialist', 'Small Business Owner'
    ],
    'university_dropout': [
        'Junior Developer', 'Business Analyst', 'Marketing Associate',
        'Technical Support', 'Entrepreneur'
    ],
    'graduate': {
        'Engineering': ['Software Engineer', 'Civil Engineer', 'Systems Engineer'],
        'Medicine': ['Doctor', 'Medical Researcher'],
        'Computer Science': ['Software Developer', 'Data Scientist', 'IT Consultant'],
        'Business Administration': ['Business Analyst', 'Project Manager', 'Management Consultant'],
        'Finance': ['Financial Analyst', 'Investment Banker', 'Accountant'],
        'Psychology': ['Counselor', 'HR Manager', 'Research Assistant']
    }
}

def generate_ol_results():
    """Generate O/L results with realistic grade distribution"""
    results = {}
    for subject in OL_SUBJECTS:
        # Generate grades with a normal distribution
        grades = np.random.normal(65, 15, SAMPLE_SIZE)
        # Clip grades between 0 and 100
        grades = np.clip(grades, 0, 100)
        results[f'ol_{subject.lower()}'] = grades.round(2)
    
    # Calculate if passed O/L (minimum 3 core subjects and total 6 subjects above 35)
    core_subjects = ['Mathematics', 'Science', 'English', 'Sinhala']
    for i in range(SAMPLE_SIZE):
        core_passes = sum(1 for subj in core_subjects if results[f'ol_{subj.lower()}'][i] >= 35)
        total_passes = sum(1 for subj in OL_SUBJECTS if results[f'ol_{subj.lower()}'][i] >= 35)
        results['ol_passed'] = core_passes >= 3 and total_passes >= 6
    
    return pd.DataFrame(results)

def determine_al_stream(ol_results):
    """Determine suitable A/L stream based on O/L results"""
    streams = []
    for _, row in ol_results.iterrows():
        if not row['ol_passed']:
            streams.append('dropout')
            continue
            
        # Check qualification for each stream
        qualified_streams = []
        for stream, requirements in AL_STREAMS.items():
            qualified = all(row[f'ol_{subj.lower()}'] >= grade 
                          for subj, grade in requirements['required_ol_grades'].items())
            if qualified:
                qualified_streams.append(stream)
        
        if qualified_streams:
            # Choose stream based on best performing relevant subjects
            streams.append(np.random.choice(qualified_streams))
        else:
            streams.append('dropout')
    
    return streams

def generate_al_results(ol_results, al_streams):
    """Generate A/L results including Z-scores"""
    results = {
        'al_stream': al_streams,
        'al_attempted': [stream != 'dropout' for stream in al_streams]
    }
    
    # Generate subject grades and Z-scores
    for stream in AL_STREAMS:
        for subject in AL_STREAMS[stream]['subjects']:
            results[f'al_{subject.lower().replace("/", "_")}'] = [np.nan] * SAMPLE_SIZE
            
    results['al_zscore'] = [np.nan] * SAMPLE_SIZE
    results['al_passed'] = [False] * SAMPLE_SIZE
    
    for i, stream in enumerate(al_streams):
        if stream != 'dropout':
            # Generate grades for stream subjects
            for subject in AL_STREAMS[stream]['subjects']:
                grades = np.random.normal(65, 15)
                results[f'al_{subject.lower().replace("/", "_")}'][i] = np.clip(grades, 0, 100)
            
            # Generate Z-score
            zscore = np.random.normal(1.2, 0.4)
            results['al_zscore'][i] = np.clip(zscore, -2, 3)
            
            # Determine if passed A/L
            subject_grades = [results[f'al_{subj.lower().replace("/", "_")}'][i] 
                            for subj in AL_STREAMS[stream]['subjects']]
            results['al_passed'][i] = all(grade >= 35 for grade in subject_grades if not np.isnan(grade))
    
    return pd.DataFrame(results)

def generate_university_data(al_results):
    """Generate university education data"""
    results = {
        'university_entrance': [False] * SAMPLE_SIZE,
        'university_program': ['None'] * SAMPLE_SIZE,
        'university_completed': [False] * SAMPLE_SIZE,
        'university_gpa': [0.0] * SAMPLE_SIZE
    }
    
    for i, row in al_results.iterrows():
        if row['al_passed'] and not pd.isna(row['al_zscore']):
            stream = row['al_stream']
            if stream != 'dropout' and row['al_zscore'] >= AL_STREAMS[stream]['zscore_threshold']:
                # Qualified for university
                results['university_entrance'][i] = True
                
                # Select program based on Z-score
                qualified_programs = []
                for prog in UNIVERSITY_PROGRAMS[stream]:
                    if row['al_zscore'] >= prog['zscore_min']:
                        qualified_programs.append(prog)
                
                if qualified_programs:
                    program = np.random.choice(qualified_programs)
                    results['university_program'][i] = program['name']
                    
                    # Generate GPA and completion status
                    if np.random.random() > program['dropout_rate']:
                        results['university_completed'][i] = True
                        results['university_gpa'][i] = np.clip(np.random.normal(3.2, 0.4), 0, 4.0)
    
    return pd.DataFrame(results)

def determine_career_path(education_data):
    """Determine career path based on education journey"""
    career_paths = []
    
    for _, row in education_data.iterrows():
        if not row['ol_passed']:
            career_paths.append(np.random.choice(CAREER_PATHS['ol_dropout']))
        elif not row['al_attempted'] or not row['al_passed']:
            career_paths.append(np.random.choice(CAREER_PATHS['al_dropout']))
        elif not row['university_entrance'] or not row['university_completed']:
            career_paths.append(np.random.choice(CAREER_PATHS['university_dropout']))
        else:
            program = row['university_program']
            possible_careers = CAREER_PATHS['graduate'].get(program, ['General Professional'])
            career_paths.append(np.random.choice(possible_careers))
    
    return career_paths

def generate_sample_data():
    """Generate synthetic student data with complete educational journey"""
    
    np.random.seed(42)
    
    # Generate O/L results
    ol_data = generate_ol_results()
    
    # Determine A/L streams
    al_streams = determine_al_stream(ol_data)
    
    # Generate A/L results
    al_data = generate_al_results(ol_data, al_streams)
    
    # Generate university data
    university_data = generate_university_data(al_data)
    
    # Combine all educational data
    education_data = pd.concat([ol_data, al_data, university_data], axis=1)
    
    # Determine career paths
    education_data['career_path'] = determine_career_path(education_data)
    
    return education_data

def main():
    # Create data directory if it doesn't exist
    data_dir = Path('data/raw')
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate and save data
    df = generate_sample_data()
    output_file = data_dir / 'sample_data.csv'
    df.to_csv(output_file, index=False)
    
    print(f"Generated {len(df)} sample records in {output_file}")
    print("\nSample data preview:")
    print(df.head())
    print("\nEducation Journey Statistics:")
    print(f"O/L Pass Rate: {(df['ol_passed'].mean() * 100):.1f}%")
    print(f"A/L Attempt Rate: {(df['al_attempted'].mean() * 100):.1f}%")
    print(f"A/L Pass Rate: {(df['al_passed'].mean() * 100):.1f}%")
    print(f"University Entrance Rate: {(df['university_entrance'].mean() * 100):.1f}%")
    print(f"University Completion Rate: {(df['university_completed'].mean() * 100):.1f}%")
    print("\nCareer Path Distribution:")
    print(df['career_path'].value_counts().head())
    
if __name__ == '__main__':
    main()
