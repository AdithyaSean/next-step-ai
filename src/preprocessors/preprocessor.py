"""Enhanced preprocessor for scikit-learn compatibility."""

import os

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

from ..config.config import CAREERS, config


def preprocessor():
    """Preprocess data for model training."""
    df = pd.read_csv(f"{config['data_dir']}/student_profiles.csv")

    # Separate features and target probabilities
    career_columns = [f"career_{career_id}" for career_id in CAREERS.values()]
    career_probs = df[career_columns]
    features = df.drop(columns=career_columns + ["profile_id"])

    # Scale numerical features
    scaler = StandardScaler()
    features_scaled = pd.DataFrame(
        scaler.fit_transform(features), columns=features.columns
    )

    # Save scaler for predictions
    os.makedirs(config["model_dir"], exist_ok=True)
    joblib.dump(scaler, f"{config['model_dir']}/scaler.joblib")

    os.makedirs(config["processed_dir"], exist_ok=True)
    features_scaled.to_csv(f"{config['processed_dir']}/features.csv", index=False)
    career_probs.to_csv(f"{config['processed_dir']}/career_probs.csv", index=False)

    print("\nDataset Summary:")
    print(f"Total profiles: {len(df)}")
    print(f"Feature columns: {len(features_scaled.columns)}")
    print(f"Target careers: {len(career_probs.columns)}")

    print("\nEducation Level Distribution:")
    for level_name, level_id in config["education_levels"].items():
        count = (df["education_level"] == level_id).sum()
        print(f"{level_name}: {count}")

    return features_scaled, career_probs


if __name__ == "__main__":
    preprocessor()
