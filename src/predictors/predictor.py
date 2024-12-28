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
    # Load model and metadata
    model = joblib.load("./models/saved/career_prediction_model.pkl")
    metadata = joblib.load("./models/saved/model_metadata.pkl")

    # Initialize features with zeros and correct data types
    feature_dict = {name: 0.0 for name in metadata["feature_names"]}

    # Fill in features with validated data
    feature_dict["education_level"] = float(education_level)

    # Handle OL results
    for subject, score in ol_results.items():
        col_name = f"ol_subject_{subject}"
        if col_name in feature_dict:
            feature_dict[col_name] = float(score)

    # Handle AL/University features
    if education_level >= 1 and al_stream is not None:
        feature_dict["AL_stream"] = float(al_stream)
        if al_results:
            for subject, score in al_results.items():
                col_name = f"al_subject_{subject}"
                if col_name in feature_dict:
                    feature_dict[col_name] = float(score)

    if education_level == 2 and gpa is not None:
        feature_dict["gpa"] = float(gpa)

    # Create DataFrame with pre-filled values
    features = pd.DataFrame([feature_dict])

    # Make prediction
    probs = model.predict(features)

    # Format results as dictionary with Python float values
    return {
        career: float(prob * 100)
        for career, prob in zip(metadata["career_names"], probs[0])
    }


def predictor():
    """Run example predictions."""
    # Test OL student
    ol_results = {
        "OL_subject_0_score": 85,  # Math
        "OL_subject_1_score": 78,  # Science
        "OL_subject_2_score": 72,  # English
        "OL_subject_3_score": 65,  # Sinhala
        "OL_subject_4_score": 70,  # History
        "OL_subject_5_score": 75,  # Religion
    }
    print("\nOL Student Profile:")
    print("==================")
    print("\nPredicted Career Probabilities:")
    print(predict(education_level=0, ol_results=ol_results))

    # Test AL Science student
    al_results = {
        "AL_subject_0_score": 88,  # Physics
        "AL_subject_1_score": 82,  # Chemistry
        "AL_subject_2_score": 90,  # Combined Maths
    }
    print("\nAL Science Student Profile:")
    print("=========================")
    print("\nPredicted Career Probabilities:")
    print(
        predict(
            education_level=1,
            ol_results=ol_results,
            al_stream=0,  # Physical Science
            al_results=al_results,
        )
    )

    # Test University student
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
