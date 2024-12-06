import pandas as pd
import sys
import os
import yaml

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.career_guidance_model import CareerGuidanceModel
from src.models.preprocessing import PreprocessingPipeline

# Load configuration
with open('src/config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

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
    pred = {
        'career_path': career_predictions[i],
        'education_path': education_predictions[i]
    }
    print(f"\nStudent {i+1}:")
    print(f"O/L Results: {dict(filter(lambda x: 'ol_' in x[0], student.items()))}")
    if 'al_stream' in student:
        print(f"A/L Stream: {student['al_stream']}")
    print(f"Predicted Career Path: {pred['career_path']}")
    print(f"Predicted Education Path: {pred['education_path']}")
