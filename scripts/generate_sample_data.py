"""
Generate realistic sample data for the career guidance system including complete educational journey
"""

import pandas as pd
import numpy as np
from pathlib import Path
import random
from typing import Dict, List

# Define sample data characteristics
SAMPLE_SIZE = 1000

# O/L Core Subjects
OL_SUBJECTS = {
    'Mathematics': {'weight': 1.0, 'pass_mark': 40},
    'Science': {'weight': 1.0, 'pass_mark': 40},
    'English': {'weight': 1.0, 'pass_mark': 40}
}

# A/L Streams and subjects
AL_STREAMS = {
    'Physical Science': {
        'subjects': ['Physics', 'Chemistry', 'Mathematics'],
        'required_ol_grades': {'Mathematics': 65, 'Science': 65},
        'zscore_threshold': 1.2,
        'career_paths': ['Software Engineer', 'Engineer', 'Data Scientist']
    },
    'Biological Science': {
        'subjects': ['Biology', 'Physics', 'Chemistry'],
        'required_ol_grades': {'Mathematics': 60, 'Science': 65},
        'zscore_threshold': 1.2,
        'career_paths': ['Doctor', 'Research Scientist']
    },
    'Commerce': {
        'subjects': ['Business Studies', 'Accounting', 'Economics'],
        'required_ol_grades': {'Mathematics': 50},
        'zscore_threshold': 1.0,
        'career_paths': ['Business Analyst', 'Accountant', 'Manager']
    },
    'Arts': {
        'subjects': [],  # Flexible subject combination
        'required_ol_grades': {},
        'zscore_threshold': 0.8,
        'career_paths': ['Teacher', 'Consultant']
    }
}

# University Programs by Stream
UNIVERSITY_PROGRAMS = {
    'Physical Science': [
        'Engineering',
        'Computer Science',
        'Physical Science'
    ],
    'Biological Science': [
        'Medicine',
        'Biological Science'
    ],
    'Commerce': [
        'Business Administration',
        'Management'
    ],
    'Arts': [
        'Arts and Humanities'
    ]
}

# Sample interests and skills by stream
INTERESTS_BY_STREAM = {
    'Physical Science': [
        'Technology', 'Mathematics', 'Physics', 'Programming',
        'Engineering', 'Problem Solving', 'Innovation'
    ],
    'Biological Science': [
        'Biology', 'Chemistry', 'Research', 'Healthcare',
        'Laboratory Work', 'Life Sciences'
    ],
    'Commerce': [
        'Business', 'Economics', 'Finance', 'Management',
        'Entrepreneurship', 'Marketing'
    ],
    'Arts': [
        'Languages', 'Social Sciences', 'History', 'Geography',
        'Literature', 'Research', 'Writing'
    ]
}

SKILLS_BY_STREAM = {
    'Physical Science': [
        'Mathematical Analysis', 'Problem Solving', 'Critical Thinking',
        'Programming', 'Data Analysis', 'Logical Reasoning'
    ],
    'Biological Science': [
        'Laboratory Skills', 'Research', 'Analysis', 'Critical Thinking',
        'Observation', 'Documentation'
    ],
    'Commerce': [
        'Financial Analysis', 'Management', 'Communication',
        'Problem Solving', 'Decision Making', 'Leadership'
    ],
    'Arts': [
        'Research', 'Writing', 'Critical Thinking', 'Analysis',
        'Communication', 'Project Management'
    ]
}

def generate_ol_results() -> pd.DataFrame:
    """Generate O/L results with realistic grade distribution"""
    data = []
    for _ in range(SAMPLE_SIZE):
        record = {}
        passed_all = True
        
        # Generate core subject results
        for subject, info in OL_SUBJECTS.items():
            # Generate mark with normal distribution
            mark = np.random.normal(65, 15)
            mark = max(min(mark, 100), 0)  # Clip between 0 and 100
            record[f'ol_{subject.lower()}'] = round(mark, 2)
            passed_all = passed_all and mark >= info['pass_mark']
        
        record['ol_passed'] = passed_all
        data.append(record)
    
    return pd.DataFrame(data)

def determine_al_stream(ol_results: pd.DataFrame) -> pd.Series:
    """Determine suitable A/L stream based on O/L results"""
    streams = []
    for _, row in ol_results.iterrows():
        if not row['ol_passed']:
            streams.append(None)
            continue
            
        # Calculate eligibility for each stream
        eligible_streams = []
        for stream, info in AL_STREAMS.items():
            eligible = True
            for subject, required_grade in info['required_ol_grades'].items():
                if row[f'ol_{subject.lower()}'] < required_grade:
                    eligible = False
                    break
            if eligible:
                eligible_streams.append(stream)
        
        # Choose stream based on grades and requirements
        if eligible_streams:
            if row['ol_mathematics'] >= 65 and row['ol_science'] >= 65:
                # Prefer Physical Science if good at math and science
                stream = 'Physical Science' if 'Physical Science' in eligible_streams else eligible_streams[0]
            elif row['ol_science'] >= 65:
                # Prefer Biological Science if good at science
                stream = 'Biological Science' if 'Biological Science' in eligible_streams else eligible_streams[0]
            else:
                stream = random.choice(eligible_streams)
        else:
            stream = 'Arts'  # Default to Arts if not eligible for others
            
        streams.append(stream)
    
    return pd.Series(streams)

def generate_al_results(ol_results: pd.DataFrame, streams: pd.Series) -> pd.DataFrame:
    """Generate A/L results including Z-scores"""
    data = []
    for (_, ol_row), stream in zip(ol_results.iterrows(), streams):
        record = {}
        
        # Copy O/L results
        for col in ol_results.columns:
            record[col] = ol_row[col]
            
        if stream is None:
            record.update({
                'al_attempted': False,
                'al_passed': False,
                'al_stream': None,
                'al_zscore': 0.0
            })
            data.append(record)
            continue
            
        record.update({
            'al_attempted': True,
            'al_stream': stream
        })
        
        # Generate subject results based on stream
        if stream != 'Arts':
            for subject in AL_STREAMS[stream]['subjects']:
                # Correlate with O/L performance
                base = (ol_row['ol_mathematics'] + ol_row['ol_science']) / 2
                mark = np.random.normal(base, 10)
                mark = max(min(mark, 100), 0)
                record[f'al_{subject.lower()}'] = round(mark, 2)
        
        # Generate Z-score
        base_zscore = np.random.normal(1.5, 0.5)
        zscore = max(min(base_zscore, 3.0), 0.0)
        record['al_zscore'] = round(zscore, 3)
        
        # Determine if passed A/L
        record['al_passed'] = zscore >= AL_STREAMS[stream]['zscore_threshold']
        
        # Generate interests and skills
        stream_interests = INTERESTS_BY_STREAM[stream]
        stream_skills = SKILLS_BY_STREAM[stream]
        
        record['interests'] = random.sample(stream_interests, k=random.randint(2, 4))
        record['skills'] = random.sample(stream_skills, k=random.randint(2, 4))
        
        data.append(record)
    
    return pd.DataFrame(data)

def determine_education_and_career(al_results: pd.DataFrame) -> pd.DataFrame:
    """Determine education path and career based on A/L results"""
    data = []
    for _, row in al_results.iterrows():
        record = row.to_dict()
        
        if not row['al_passed']:
            record.update({
                'university_entrance': False,
                'university_program': None,
                'university_completed': False,
                'university_gpa': 0.0,
                'education_path': 'Technical Training',
                'career_path': random.choice(['Technical Support', 'Sales', 'Administrative'])
            })
        else:
            stream = row['al_stream']
            zscore = row['al_zscore']
            
            # Determine university entrance and program
            possible_programs = UNIVERSITY_PROGRAMS[stream]
            if zscore >= 1.6:  # University entrance cutoff
                record['university_entrance'] = True
                record['university_program'] = random.choice(possible_programs)
                
                # Generate GPA and completion status
                completion_prob = 0.85  # 85% chance of completing
                record['university_completed'] = random.random() < completion_prob
                
                if record['university_completed']:
                    record['university_gpa'] = round(random.uniform(2.5, 4.0), 2)
                    record['education_path'] = record['university_program']
                    record['career_path'] = random.choice(AL_STREAMS[stream]['career_paths'])
                else:
                    record['university_gpa'] = round(random.uniform(1.5, 2.5), 2)
                    record['education_path'] = 'Incomplete Degree'
                    record['career_path'] = 'Technical Support'
            else:
                record.update({
                    'university_entrance': False,
                    'university_program': None,
                    'university_completed': False,
                    'university_gpa': 0.0,
                    'education_path': 'Professional Certification',
                    'career_path': random.choice(['Technical Support', 'Sales', 'Administrative'])
                })
        
        data.append(record)
    
    return pd.DataFrame(data)

def main():
    """Generate and save sample data"""
    # Create data directory if it doesn't exist
    data_dir = Path('data/raw')
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate data
    print("Generating O/L results...")
    ol_results = generate_ol_results()
    
    print("Determining A/L streams...")
    streams = determine_al_stream(ol_results)
    
    print("Generating A/L results...")
    al_results = generate_al_results(ol_results, streams)
    
    print("Determining education and career paths...")
    final_data = determine_education_and_career(al_results)
    
    # Save data
    output_path = data_dir / 'sample_data.csv'
    final_data.to_csv(output_path, index=False)
    print(f"Sample data saved to {output_path}")
    
    # Print summary statistics
    print("\nData Generation Summary:")
    print(f"Total records: {len(final_data)}")
    print(f"O/L pass rate: {(final_data['ol_passed']).mean():.1%}")
    print(f"A/L attempt rate: {(final_data['al_attempted']).mean():.1%}")
    print(f"A/L pass rate: {(final_data['al_passed']).mean():.1%}")
    print(f"University entrance rate: {(final_data['university_entrance']).mean():.1%}")
    print(f"University completion rate: {(final_data['university_completed']).mean():.1%}")
    
    print("\nStream Distribution:")
    print(final_data['al_stream'].value_counts(normalize=True))

if __name__ == '__main__':
    main()
