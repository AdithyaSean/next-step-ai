"""Predictor module for career prediction."""

from typing import Dict, Optional

import joblib
import pandas as pd


def predict(
    education_level: int,
    ol_results: Dict[str, float],
    al_stream: Optional[int] = None,
    al_results: Optional[Dict[str, float]] = None,
    gpa: Optional[float] = None,
) -> Dict[str, float]:
    """Predict career probabilities based on educational profile."""
    # Load model, scaler and metadata
    model = joblib.load("./models/career_predictor.joblib")
    metadata = joblib.load("./models/model_metadata.joblib")
    scaler = joblib.load("./models/scaler.joblib")

    # Initialize features with zeros
    feature_dict = {name: 0.0 for name in metadata["feature_names"]}

    # Set education level
    feature_dict["education_level"] = education_level

    # Map OL results
    for subject_id, score in ol_results.items():
        feature_name = f"OL_subject_{subject_id}_score"
        if feature_name in feature_dict:
            feature_dict[feature_name] = score

    # Map AL results if applicable
    if education_level >= 1 and al_stream is not None:
        feature_dict["AL_stream"] = al_stream
        if al_results:
            for subject_id, score in al_results.items():
                feature_name = f"AL_subject_{subject_id}_score"
                if feature_name in feature_dict:
                    feature_dict[feature_name] = score

    # Add GPA for university students
    if education_level == 2 and gpa is not None:
        feature_dict["gpa"] = gpa

    # Create features DataFrame
    features = pd.DataFrame([feature_dict])

    # Normalize features
    features_scaled = pd.DataFrame(scaler.transform(features), columns=features.columns)

    # Ensure correct feature order
    features_scaled = features_scaled[metadata["feature_names"]]

    # Make prediction
    predictions = model.predict(features_scaled)[0]

    # Map predictions to careers
    return {
        career: float(prob * 100)
        for career, prob in zip(metadata["career_names"], predictions)
    }


def predictor():
    """Run example predictions."""
    # Example OL student
    ol_results = {
        "0": 85,  # Math
        "1": 78,  # Science
        "2": 72,  # English
        "3": 65,  # Sinhala
        "4": 70,  # History
        "5": 75,  # Religion
    }

    print("\nOL Student Profile:")
    print("==================")
    print("\nPredicted Career Probabilities:")
    print(predict(education_level=0, ol_results=ol_results))

    # Example AL Science student
    al_results = {
        "0": 88,  # Physics
        "1": 82,  # Chemistry
        "2": 90,  # Combined Maths
    }

    print("\nAL Science Student Profile:")
    print("=========================")
    print("\nPredicted Career Probabilities:")
    print(
        predict(
            education_level=1, ol_results=ol_results, al_stream=0, al_results=al_results
        )
    )

    # Example University student
    print("\nUniversity Student Profile:")
    print("=========================")
    print("\nPredicted Career Probabilities:")
    print(
        predict(
            education_level=2,
            ol_results=ol_results,
            al_stream=0,
            al_results=al_results,
            gpa=3.75,
        )
    )


if __name__ == "__main__":
    predictor()
