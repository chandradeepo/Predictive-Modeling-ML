"""
===========================================================
Project: Predictive Modeling Using Machine Learning
Dataset: Titanic - Machine Learning from Disaster

Description:
    Train multiple machine learning models, compare their
    performance, and save the best-performing model.

Author: Chandradeep
===========================================================
"""

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from config import (
    MODEL_FILE,
    RANDOM_STATE,
    TEST_SIZE
)


def split_dataset(X, y):
    """
    Split dataset into training and testing sets.
    """

    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )


def train_models(X_train, y_train):
    """
    Train multiple ML models.
    """

    models = {

        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE
        ),

        "Decision Tree": DecisionTreeClassifier(
            random_state=RANDOM_STATE
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE
        )
    }

    trained_models = {}

    for name, model in models.items():

        model.fit(X_train, y_train)

        trained_models[name] = model

    return trained_models


def evaluate_models(models, X_test, y_test):
    """
    Evaluate every trained model.
    """

    scores = {}

    print("\n" + "=" * 60)
    print("MODEL PERFORMANCE")
    print("=" * 60)

    for name, model in models.items():

        prediction = model.predict(X_test)

        accuracy = accuracy_score(y_test, prediction)

        scores[name] = accuracy

        print(f"{name:<25} {accuracy:.4f}")

    return scores


def save_best_model(models, scores):
    """
    Save the best-performing model.
    """

    best_model_name = max(scores, key=scores.get)

    best_model = models[best_model_name]

    joblib.dump(best_model, MODEL_FILE)

    print("\nBest Model :", best_model_name)

    print(f"Accuracy   : {scores[best_model_name]:.4f}")

    print(f"Saved To   : {MODEL_FILE}")

    return best_model_name