"""Prediction module for making inferences with trained models."""

import logging
from typing import Any

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class Predictor:
    """Make predictions using trained models."""

    def __init__(self, model: Any):
        """
        Initialize predictor with a trained model.

        Args:
            model: Trained model instance.
        """
        self.model = model

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on input data.

        Args:
            X: Feature matrix.

        Returns:
            Predictions.
        """
        predictions = self.model.predict(X)
        logger.info(f"Made predictions for {len(X)} samples")

        return predictions

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get prediction probabilities for classification models.

        Args:
            X: Feature matrix.

        Returns:
            Prediction probabilities.
        """
        if not hasattr(self.model, "predict_proba"):
            raise AttributeError("Model does not support predict_proba")

        probabilities = self.model.predict_proba(X)
        logger.info(f"Computed prediction probabilities for {len(X)} samples")

        return probabilities
