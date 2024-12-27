"""trainer package."""

import os

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, train_test_split

from ..preprocessors.preprocessor import preprocess_data


def train_model():
    """Train and evaluate the career prediction model."""
    # Get preprocessed data
    data, encoders = preprocess_data()
    if data is None:
        return

    # Split features and target
    X = data.drop("Recommended_Career", axis=1)
    y = data["Recommended_Career"]

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Define model and parameters
    model = RandomForestClassifier(random_state=42)
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [10, 20, None],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
    }

    # Grid search
    grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Evaluate
    y_pred = grid_search.predict(X_test)
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Model Accuracy: {accuracy_score(y_test, y_pred)}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Save model
    model_dir = "./models/saved"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "career_prediction_model.pkl")
    joblib.dump(grid_search.best_estimator_, model_path)

    # Save encoders
    encoder_path = os.path.join(model_dir, "encoders.pkl")
    joblib.dump(encoders, encoder_path)


if __name__ == "__main__":
    train_model()
