"""Train and evaluate the career prediction model."""

import os

import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split

from ..config.config import CAREERS, config
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

    # Save best model and metadata
    best_model = grid_search.best_estimator_
    os.makedirs(config["model_dir"], exist_ok=True)

    # Save model
    model_path = f"{config['model_dir']}/career_predictor.joblib"
    joblib.dump(best_model, model_path)

    # Save metadata
    metadata = {
        "feature_names": features.columns.tolist(),
        "career_names": list(CAREERS.keys()),
    }
    metadata_path = f"{config['model_dir']}/model_metadata.joblib"
    joblib.dump(metadata, metadata_path)

    # Evaluate on test set
    y_pred = best_model.predict(X_test)

    print("\nModel Performance:\n")
    for i, career in enumerate(CAREERS.keys()):
        mse = mean_squared_error(y_test.iloc[:, i], y_pred[:, i])
        r2 = r2_score(y_test.iloc[:, i], y_pred[:, i])
        print(f"{career}:")
        print(f"MSE: {mse:.4f}")
        print(f"R2 Score: {r2:.4f}\n")

    return best_model


if __name__ == "__main__":
    trainer()
