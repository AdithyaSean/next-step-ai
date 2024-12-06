import pandas as pd
import sys
import os
import yaml
import json

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.career_guidance_model import CareerGuidanceModel
from src.models.preprocessing import PreprocessingPipeline

# Load configuration and data stats
with open('src/config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)
    
with open('data/processed/data_stats.json', 'r') as f:
    data_stats = json.load(f)

# Create path mappings
career_paths = list(data_stats['target_distribution']['career_path'].keys())
education_paths = list(data_stats['target_distribution']['education_path'].keys())

# Load the sample data
data_path = 'data/raw/sample_data.csv'
df = pd.read_csv(data_path)

# Split into training and test sets
from sklearn.model_selection import train_test_split
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Initialize the model and preprocessing pipeline
preprocessor = PreprocessingPipeline(config)
model = CareerGuidanceModel(config)

# Preprocess the data
print("Preprocessing the data...")
X_train = preprocessor.preprocess_features(train_df)
X_test = preprocessor.preprocess_features(test_df)

y_train = preprocessor.preprocess_targets(train_df)
y_test = preprocessor.preprocess_targets(test_df)

# Train the model
print("\nTraining the model...")
model.fit(X_train, y_train)

# Make predictions on test set
print("\nMaking predictions...")
predictions = model.predict(X_test)

# Calculate accuracy
from sklearn.metrics import accuracy_score
career_predictions, education_predictions = predictions  # Unpack the tuple
y_test_career, y_test_education = y_test  # Unpack the target tuple
career_accuracy = accuracy_score(y_test_career, career_predictions)
education_accuracy = accuracy_score(y_test_education, education_predictions)

print("\nModel Performance:")
print(f"Career Path Prediction Accuracy: {career_accuracy:.2%}")
print(f"Education Path Prediction Accuracy: {education_accuracy:.2%}")

# Show some example predictions
print("\nExample Predictions:")
for i in range(5):
    student = test_df.iloc[i]
    pred_career_idx = career_predictions[i]
    pred_education_idx = education_predictions[i]
    
    print(f"\nStudent {i+1}:")
    print(f"O/L Results: {dict(filter(lambda x: 'ol_' in x[0], student.items()))}")
    if 'al_stream' in student:
        print(f"A/L Stream: {student['al_stream']}")
    print(f"Predicted Career Path: {pred_career_idx} ({career_paths[pred_career_idx]})")
    print(f"Predicted Education Path: {pred_education_idx} ({education_paths[pred_education_idx]})")

# Print path mappings for reference
print("\nPath Number Mappings:")
print("\nCareer Paths:")
for idx, path in enumerate(career_paths):
    print(f"{idx}: {path}")

print("\nEducation Paths:")
for idx, path in enumerate(education_paths):
    print(f"{idx}: {path}")
