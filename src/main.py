"""
===========================================================
Project: Predictive Modeling Using Machine Learning

Description:
    Main entry point for the project.

Author: Chandradeep
===========================================================
"""

from data_loader import load_training_data
from preprocessing import preprocess_training_data
from eda import perform_eda
from train import (
    split_dataset,
    train_models,
    evaluate_models,
    save_best_model,
)
from evaluate import evaluate_model


def main():

    # ==========================================
    # Load Dataset
    # ==========================================

    dataframe = load_training_data()

    # ==========================================
    # Perform EDA
    # ==========================================

    perform_eda(dataframe)

    # ==========================================
    # Preprocess Data
    # ==========================================

    X, y = preprocess_training_data(dataframe)

    # ==========================================
    # Train-Test Split
    # ==========================================

    X_train, X_test, y_train, y_test = split_dataset(X, y)

    # ==========================================
    # Train Models
    # ==========================================

    models = train_models(X_train, y_train)

    # ==========================================
    # Compare Models
    # ==========================================

    scores = evaluate_models(models, X_test, y_test)

    # ==========================================
    # Save Best Model
    # ==========================================

    best_model_name = save_best_model(models, scores)

    # ==========================================
    # Detailed Evaluation
    # ==========================================

    best_model = models[best_model_name]

    evaluate_model(best_model, X_test, y_test)

    print("\nProject completed successfully.")


if __name__ == "__main__":
    main()