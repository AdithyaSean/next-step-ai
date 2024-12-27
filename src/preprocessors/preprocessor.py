"""preprocessor package."""

import os

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from ..config.config import config


def preprocess_data():
    """Preprocess student data for model training."""
    data_path = os.path.join(config["data_dir"], "synthetic_student_data.csv")
    try:
        data = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}")
        return None

    # Encode categorical features
    categorical_cols = data.select_dtypes(include=["object"]).columns
    encoders = {}
    for col in categorical_cols:
        encoders[col] = LabelEncoder()
        data[col] = encoders[col].fit_transform(data[col])

    # Save processed data
    processed_data_dir = os.path.join("./data", "processed")
    os.makedirs(processed_data_dir, exist_ok=True)
    processed_data_path = os.path.join(processed_data_dir, "processed_data.csv")
    data.to_csv(processed_data_path, index=False)

    return data, encoders


if __name__ == "__main__":
    preprocess_data()
