"""Generate and process dataset for model training."""

import sys
from pathlib import Path
import json
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data.generators import StudentDataGenerator
from src.data.preprocessor import DataPreprocessor

def main():
    """Generate, process, and save dataset."""
    # Create directories
    data_dir = project_root / "data"
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"
    models_dir = data_dir / "models"
    
    for dir_path in [data_dir, raw_dir, processed_dir, models_dir]:
        dir_path.mkdir(exist_ok=True)
    
    # Generate dataset
    print("Generating dataset...")
    generator = StudentDataGenerator(seed=42)
    dataset_size = 10000  # Adjust based on needs
    raw_data = generator.generate_dataset(dataset_size)
    
    # Save raw data
    print("Saving raw data...")
    raw_path = raw_dir / "student_profiles.json"
    with open(raw_path, 'w') as f:
        json.dump(raw_data, f, indent=2)
    
    # Initialize and fit preprocessor
    print("Preprocessing data...")
    preprocessor = DataPreprocessor()
    processed_data = preprocessor.fit_transform(raw_data)
    
    # Save processed data
    print("Saving processed data...")
    processed_path = processed_dir / "processed_profiles.csv"
    processed_data.to_csv(processed_path, index=False)
    
    # Save preprocessor
    print("Saving preprocessor...")
    preprocessor_path = models_dir / "preprocessor"
    preprocessor.save(preprocessor_path)
    
    # Print statistics
    print("\nDataset Statistics:")
    print(f"Total profiles: {len(raw_data)}")
    print(f"Features generated: {processed_data.shape[1]}")
    print("\nSample feature names:")
    print(", ".join(processed_data.columns[:10]))
    
    print("\nFiles saved:")
    print(f"Raw data: {raw_path}")
    print(f"Processed data: {processed_path}")
    print(f"Preprocessor: {preprocessor_path}")

if __name__ == "__main__":
    main()
