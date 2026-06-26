"""
===========================================================
Project: Predictive Modeling Using Machine Learning
Dataset: Titanic - Machine Learning from Disaster

Description:
    Load the trained model and make predictions on new data.

Author: Chandradeep
===========================================================
"""

import joblib
import pandas as pd

from config import MODEL_FILE


def load_model():
    """
    Load the saved machine learning model.
    """

    return joblib.load(MODEL_FILE)


def predict(model, dataframe: pd.DataFrame):
    """
    Predict survival for new passengers.

    Parameters
    ----------
    model : sklearn estimator
    dataframe : pd.DataFrame

    Returns
    -------
    predictions
    """

    return model.predict(dataframe)