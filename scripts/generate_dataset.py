"""Generate and validate sample dataset."""

import os
import sys
import json
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data.generators import StudentDataGenerator
from src.data.validator import DataValidator

def main():
    # Create output directories if they don't exist
    data_dir = project_root / "data"
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"
    
    for dir_path in [data_dir, raw_dir, processed_dir]:
        dir_path.mkdir(exist_ok=True)
    
    # Initialize generator and validator
    generator = StudentDataGenerator(seed=42)  # For reproducibility
    validator = DataValidator()
    
    # Generate dataset
    print("Generating dataset...")
    dataset_size = 1000
    dataset = generator.generate_dataset(dataset_size)
    
    # Validate each profile
    print("Validating profiles...")
    valid_profiles = []
    invalid_profiles = []
    
    for profile in dataset:
        if validator.validate_student_profile(profile):
            valid_profiles.append(profile)
        else:
            print(f"Invalid profile {profile['student_id']}:")
            print("\n".join(validator.get_errors()))
            invalid_profiles.append(profile)
    
    print(f"\nValidation Results:")
    print(f"Total profiles: {len(dataset)}")
    print(f"Valid profiles: {len(valid_profiles)}")
    print(f"Invalid profiles: {len(invalid_profiles)}")
    
    if valid_profiles:
        # Save valid profiles
        print("\nSaving valid profiles...")
        
        # Save as JSON
        json_path = raw_dir / "student_profiles.json"
        with open(json_path, 'w') as f:
            json.dump(valid_profiles, f, indent=2)
        
        # Generate CSV version
        csv_path = raw_dir / "student_profiles.csv"
        generator.save_dataset(len(valid_profiles), str(raw_dir))
        
        print(f"\nDataset saved to:")
        print(f"JSON: {json_path}")
        print(f"CSV: {csv_path}")

if __name__ == "__main__":
    main()
