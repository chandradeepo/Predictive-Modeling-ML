"""
===========================================================
Project: Predictive Modeling Using Machine Learning
Dataset: Titanic - Machine Learning from Disaster

Description:
    Evaluate the trained machine learning model using
    various classification metrics and save evaluation
    visualizations.

Author: Chandradeep
===========================================================
"""

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)

from config import (
    CONFUSION_MATRIX_IMAGE,
    ROC_CURVE_IMAGE
)


def evaluate_model(model, X_test, y_test):
    """
    Evaluate a trained model and save evaluation plots.

    Parameters
    ----------
    model : sklearn estimator
        Trained machine learning model.

    X_test : pd.DataFrame
        Testing features.

    y_test : pd.Series
        True labels.
    """

    # ==========================================
    # Predictions
    # ==========================================

    y_pred = model.predict(X_test)

    # ==========================================
    # Metrics
    # ==========================================

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    print("\nClassification Report")
    print("-" * 60)
    print(classification_report(y_test, y_pred))

    # ==========================================
    # Confusion Matrix
    # ==========================================

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False
    )

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.tight_layout()

    plt.savefig(CONFUSION_MATRIX_IMAGE)

    plt.close()

    # ==========================================
    # ROC Curve
    # ==========================================

    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(X_test)[:, 1]

        fpr, tpr, _ = roc_curve(y_test, probabilities)

        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(7, 5))

        plt.plot(
            fpr,
            tpr,
            label=f"AUC = {roc_auc:.3f}"
        )

        plt.plot(
            [0, 1],
            [0, 1],
            linestyle="--"
        )

        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")

        plt.title("ROC Curve")

        plt.legend()

        plt.tight_layout()

        plt.savefig(ROC_CURVE_IMAGE)

        plt.close()

    print("\nEvaluation completed successfully.")