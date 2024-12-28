"""Predictor module for career prediction."""

import joblib
import pandas as pd


def predict(
    education_level, ol_results, al_stream=None, al_results=None, uni_score=None
):
    """Predict career probabilities."""
    # Load model and metadata
    model = joblib.load("./models/saved/career_prediction_model.pkl")
    metadata = joblib.load("./models/saved/model_metadata.pkl")

    # Create feature vector with correct column names
    features = pd.DataFrame(columns=metadata["feature_names"])

    # Fill in basic features
    features.loc[0, "education_level"] = education_level

    # Fill OL results
    for subject, score in ol_results.items():
        col_name = f"ol_subject_{subject}"
        if col_name in features.columns:
            features.loc[0, col_name] = score

    # Fill AL/University features if applicable
    if education_level >= 1 and al_stream is not None:
        features.loc[0, "al_stream"] = al_stream
        for subject, score in al_results.items():
            col_name = f"al_subject_{subject}"
            if col_name in features.columns:
                features.loc[0, col_name] = score

    if education_level == 2 and uni_score is not None:
        features.loc[0, "uni_score"] = uni_score

    # Fill missing values with 0
    features = features.fillna(0)

    # Make prediction
    probs = model.predict(features)

    # Format results as dictionary
    return {
        career: prob * 100 for career, prob in zip(metadata["career_names"], probs[0])
    }


# Example usage:
def predictor():
    """Edit profiles to test."""
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
    print("\nAL Science Student Profile:")
    print("=========================")
    al_results = {
        "AL_subject_0_score": 88,  # Physics
        "AL_subject_1_score": 82,  # Chemistry
        "AL_subject_2_score": 90,  # Combined Maths
    }
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
            uni_score=85,
        )
    )


if __name__ == "__main__":
    predictor()
