"""Train and evaluate the career prediction model."""

import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import (
    GridSearchCV,
    cross_val_score,
    learning_curve,
    train_test_split,
)

from ..config.config import CAREERS, config
from ..preprocessors.preprocessor import preprocessor


def plot_learning_curves(model, X, y):
    """Plot learning curves to detect overfitting."""
    try:
        train_sizes, train_scores, val_scores = learning_curve(
            model,
            X,
            y,
            cv=5,
            n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10),
            scoring="r2",
        )

        plt.figure(figsize=(10, 6))
        plt.plot(
            train_sizes, np.mean(train_scores, axis=1), "o-", label="Training score"
        )
        plt.plot(
            train_sizes,
            np.mean(val_scores, axis=1),
            "o-",
            label="Cross-validation score",
        )
        plt.fill_between(
            train_sizes,
            np.mean(train_scores, axis=1) - np.std(train_scores, axis=1),
            np.mean(train_scores, axis=1) + np.std(train_scores, axis=1),
            alpha=0.1,
        )
        plt.fill_between(
            train_sizes,
            np.mean(val_scores, axis=1) - np.std(val_scores, axis=1),
            np.mean(val_scores, axis=1) + np.std(val_scores, axis=1),
            alpha=0.1,
        )
        plt.xlabel("Training Examples")
        plt.ylabel("R² Score")
        plt.title("Learning Curves")
        plt.legend(loc="best")
        plt.grid(True)

        os.makedirs(config["model_dir"], exist_ok=True)
        plt.savefig(
            f"{config['model_dir']}/learning_curves.png", dpi=300, bbox_inches="tight"
        )
        plt.close()
    except Exception as e:
        print(f"Error plotting learning curves: {e}")


def trainer():
    """Train and evaluate the career prediction model."""
    # Get preprocessed data
    features, targets = preprocessor()
    if features is None:
        return

    # Split train/test first
    X_train, X_test, y_train, y_test = train_test_split(
        features, targets, test_size=0.2, random_state=42
    )

    # Initialize feature selector with proper feature names
    selector = RandomForestRegressor(n_estimators=50, max_depth=8, random_state=42)
    feature_selector = SelectFromModel(selector)

    # Fit and transform while maintaining feature names
    X_train_selected = pd.DataFrame(
        feature_selector.fit_transform(X_train, y_train),
        columns=X_train.columns[feature_selector.get_support()],
    )

    X_test_selected = pd.DataFrame(
        feature_selector.transform(X_test),
        columns=X_train.columns[feature_selector.get_support()],
    )

    # Store selected feature names
    selected_features = X_train_selected.columns.tolist()

    # Define model and parameters
    model = RandomForestRegressor(random_state=42)
    param_grid = {
        "n_estimators": [50, 100],
        "max_depth": [5, 8, 10],
        "min_samples_split": [5, 10],
        "min_samples_leaf": [2, 4],
        "max_features": ["sqrt"],
    }

    # Grid search
    grid_search = GridSearchCV(
        model, param_grid, cv=5, n_jobs=-1, scoring="r2", verbose=2
    )
    grid_search.fit(X_train_selected, y_train)

    # Get best model
    best_model = grid_search.best_estimator_

    # Now do cross-validation
    cv_scores = cross_val_score(best_model, X_train_selected, y_train, cv=5)
    print(f"\nCross-validation scores: {cv_scores}")
    print(f"Mean CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

    # Print best parameters
    print("\nBest parameters:", grid_search.best_params_)

    # Plot learning curves
    plot_learning_curves(best_model, X_train_selected, y_train)

    # Fix feature importance analysis
    feature_importance = pd.DataFrame(
        {"feature": selected_features, "importance": best_model.feature_importances_}
    ).sort_values("importance", ascending=False)
    print("\nTop 10 most important features:")
    print(feature_importance.head(10))

    os.makedirs(config["model_dir"], exist_ok=True)

    # Save metadata with proper feature names
    metadata = {
        "feature_names": selected_features,
        "career_names": list(CAREERS.keys()),
        "best_params": grid_search.best_params_,
        "cv_scores": {"mean": float(cv_scores.mean()), "std": float(cv_scores.std())},
    }

    # Save model and metadata
    joblib.dump(best_model, f"{config['model_dir']}/career_predictor.joblib")
    joblib.dump(metadata, f"{config['model_dir']}/model_metadata.joblib")
    joblib.dump(feature_selector, f"{config['model_dir']}/feature_selector.joblib")

    # Save feature importance
    feature_importance.to_csv(
        f"{config['model_dir']}/feature_importance.csv", index=False
    )

    # Evaluate on test set
    y_pred = best_model.predict(X_test_selected)

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
