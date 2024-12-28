"""trainer package."""

import os

import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split

from ..config.config import CAREERS
from ..preprocessors.preprocessor import preprocessor


def trainer():
    """Train and evaluate the career prediction model."""
    # Get preprocessed data
    features, targets = preprocessor()
    if features is None:
        return

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        features, targets, test_size=0.2, random_state=42
    )

    # Define model and parameters
    model = RandomForestRegressor(random_state=42)
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [10, 20, None],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
    }

    # Grid search
    grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1, scoring="r2")
    grid_search.fit(X_train, y_train)

    # Evaluate
    y_pred = grid_search.predict(X_test)

    print(f"\nBest parameters: {grid_search.best_params_}")
    print("\nModel Performance:")

    # Calculate metrics for each career
    for idx, career_name in enumerate(CAREERS.keys()):
        mse = mean_squared_error(y_test.iloc[:, idx], y_pred[:, idx])
        r2 = r2_score(y_test.iloc[:, idx], y_pred[:, idx])
        print(f"\n{career_name}:")
        print(f"MSE: {mse:.4f}")
        print(f"R2 Score: {r2:.4f}")

    # Save model
    model_dir = "./models/saved"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "career_prediction_model.pkl")
    joblib.dump(grid_search.best_estimator_, model_path)

    # Save feature names for later use
    feature_names = {
        "feature_names": list(features.columns),
        "career_names": list(CAREERS.keys()),
    }
    metadata_path = os.path.join(model_dir, "model_metadata.pkl")
    joblib.dump(feature_names, metadata_path)


if __name__ == "__main__":
    trainer()
