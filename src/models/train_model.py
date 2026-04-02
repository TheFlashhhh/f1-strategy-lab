"""Model training module."""

import logging
import pickle
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import cross_val_score

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Train and evaluate machine learning models."""

    def __init__(self, model: Any):
        """
        Initialize trainer with a model.

        Args:
            model: Scikit-learn compatible model instance.
        """
        self.model = model

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Train the model.

        Args:
            X: Feature matrix.
            y: Target variable.
        """
        self.model.fit(X, y)
        logger.info(f"Model trained on {len(X)} samples")

    def evaluate(self, X: pd.DataFrame, y: pd.Series, cv: int = 5) -> float:
        """
        Evaluate model using cross-validation.

        Args:
            X: Feature matrix.
            y: Target variable.
            cv: Number of cross-validation folds.

        Returns:
            Mean cross-validation score.
        """
        scores = cross_val_score(self.model, X, y, cv=cv)
        mean_score = scores.mean()
        logger.info(f"Cross-validation score: {mean_score:.4f} (+/- {scores.std():.4f})")

        return mean_score

    def save_model(self, filepath: str) -> None:
        """
        Save trained model to disk.

        Args:
            filepath: Path where model will be saved.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "wb") as f:
            pickle.dump(self.model, f)

        logger.info(f"Model saved to {filepath}")

    @staticmethod
    def load_model(filepath: str) -> Any:
        """
        Load trained model from disk.

        Args:
            filepath: Path to saved model.

        Returns:
            Loaded model.
        """
        with open(filepath, "rb") as f:
            model = pickle.load(f)

        logger.info(f"Model loaded from {filepath}")
        return model
