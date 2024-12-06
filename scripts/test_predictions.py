import pandas as pd
import sys
import os
import yaml
import json
import argparse
import joblib
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.career_guidance_model import CareerGuidanceModel
from src.models.preprocessing import PreprocessingPipeline
from sklearn.metrics import accuracy_score

# Set up argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--train', type=str, help='Path to new training data CSV file')
args = parser.parse_args()

# Check if we should load existing model or create new one
model_path = Path('models/career_guidance_model.joblib')
preprocessor_path = Path('models/preprocessor.joblib')

if args.train:
    print("\nChecking model status...")
    if model_path.exists() and preprocessor_path.exists():
        print("Loading existing model and preprocessor...")
        model = joblib.load(model_path)
        preprocessor = joblib.load(preprocessor_path)
        
        # Train on new data
        print(f"\nIncremental training with new data from: {args.train}")
        new_df = pd.read_csv(args.train)
        X_new = preprocessor.preprocess_features(new_df)
        y_new = preprocessor.preprocess_targets(new_df)
        model.fit(X_new, y_new)
        
        # Save updated model
        joblib.dump(model, model_path)
        joblib.dump(preprocessor, preprocessor_path)
        print("Updated model saved successfully")
    else:
        print("No existing model found. Training new model...")

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
career_predictions, education_predictions = model.predict(X_test)
career_probas, education_probas = model.predict_proba(X_test)

# Calculate accuracy
career_accuracy = accuracy_score(y_test[0], career_predictions)
education_accuracy = accuracy_score(y_test[1], education_predictions)

print("\nModel Performance:")
print(f"Career Path Prediction Accuracy: {career_accuracy:.2%}")
print(f"Education Path Prediction Accuracy: {education_accuracy:.2%}")

# Show some example predictions
print("\nExample Predictions:")
for i in range(5):
    student = test_df.iloc[i]
    career_pred = career_predictions[i]
    education_pred = education_predictions[i]
    
    # Get prediction probabilities
    career_prob = career_probas[i][career_pred]
    education_prob = education_probas[i][education_pred]
    
    print(f"\nStudent {i+1}:")
    print(f"O/L Results: {dict(filter(lambda x: 'ol_' in x[0], student.items()))}")
    if 'al_stream' in student:
        print(f"A/L Stream: {student['al_stream']}")
    print(f"Predicted Career Path: {career_pred} ({career_paths[career_pred]}) - Confidence: {career_prob:.2%}")
    print(f"Predicted Education Path: {education_pred} ({education_paths[education_pred]}) - Confidence: {education_prob:.2%}")

# Print path mappings for reference
print("\nPath Number Mappings with Success Rates:")
print("\nCareer Paths:")
for idx, path in enumerate(career_paths):
    mask = career_predictions == idx
    success_rate = accuracy_score(y_test[0][mask], career_predictions[mask]) if any(mask) else 0
    print(f"{idx}: {path} - Success Rate: {success_rate:.2%}")

print("\nEducation Paths:")
for idx, path in enumerate(education_paths):
    mask = education_predictions == idx
    success_rate = accuracy_score(y_test[1][mask], education_predictions[mask]) if any(mask) else 0
    print(f"{idx}: {path} - Success Rate: {success_rate:.2%}")
