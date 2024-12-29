"""Predictor module for career prediction."""

from typing import Dict, Optional

import joblib
import numpy as np
import pandas as pd

from ..config.config import config


def get_base_features() -> pd.DataFrame:
    """Create DataFrame with training features structure."""
    try:
        # Load feature order from training
        feature_order = joblib.load(f"{config['model_dir']}/feature_order.joblib")
        feature_names = feature_order["feature_names"]

        # Create DataFrame with correct columns
        features = pd.DataFrame(columns=feature_names)
        features.loc[0] = -1.0  # Initialize with -1
        return features
    except Exception as e:
        raise RuntimeError(f"Failed to initialize features: {e}")


def predict(
    education_level: int,
    ol_results: Dict[str, float],
    al_stream: Optional[int] = None,
    al_results: Optional[Dict[str, float]] = None,
    gpa: Optional[float] = None,
) -> Dict[str, float]:
    """Predict career probabilities based on educational profile."""
    try:
        # Load model artifacts
        model = joblib.load(f"{config['model_dir']}/career_predictor.joblib")
        metadata = joblib.load(f"{config['model_dir']}/model_metadata.joblib")

        # Initialize features
        features = get_base_features()

        # Set basic features
        features.loc[0, "education_level"] = education_level
        features.loc[0, "AL_stream"] = al_stream if al_stream is not None else -1
        features.loc[0, "gpa"] = gpa if gpa is not None else -1

        # Set OL subject scores
        for subject_id, score in ol_results.items():
            col_name = f"OL_subject_{subject_id}_score"
            if col_name in features.columns:
                features.loc[0, col_name] = float(score)

        # Set AL subject scores
        if al_results:
            for subject_id, score in al_results.items():
                col_name = f"AL_subject_{subject_id}_score"
                if col_name in features.columns:
                    features.loc[0, col_name] = float(score)

        # Select features used by model
        selected_features = metadata["feature_names"]
        features_selected = features[selected_features]

        # Make prediction
        predictions = model.predict(features_selected)[0]

        # Return career probabilities
        return {
            career: float(np.clip(prob * 100, 0, 100))
            for career, prob in zip(metadata["career_names"], predictions)
        }

    except Exception as e:
        print(f"Prediction error: {e}")
        return {}


def predictor():
    """Run sample predictions."""
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
