"""
===========================================================
Project: Predictive Modeling Using Machine Learning
Dataset: Titanic - Machine Learning from Disaster

Description:
    This module performs Exploratory Data Analysis (EDA)
    and saves visualizations to the images folder.

Author: Chandradeep
===========================================================
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from config import (
    IMAGES_DIR,
    CORRELATION_HEATMAP_IMAGE
)


sns.set_style("whitegrid")


def plot_survival_distribution(dataframe: pd.DataFrame):
    """
    Plot survival count.
    """

    plt.figure(figsize=(6, 5))

    sns.countplot(
        x="Survived",
        data=dataframe,
        palette="viridis"
    )

    plt.title("Survival Distribution")
    plt.xlabel("Survived")
    plt.ylabel("Count")

    plt.tight_layout()

    plt.savefig(IMAGES_DIR / "survival_distribution.png")

    plt.close()


def plot_age_distribution(dataframe: pd.DataFrame):
    """
    Plot age histogram.
    """

    plt.figure(figsize=(8, 5))

    sns.histplot(
        dataframe["Age"],
        bins=30,
        kde=True,
        color="steelblue"
    )

    plt.title("Age Distribution")

    plt.tight_layout()

    plt.savefig(IMAGES_DIR / "age_distribution.png")

    plt.close()


def plot_fare_distribution(dataframe: pd.DataFrame):
    """
    Plot fare distribution.
    """

    plt.figure(figsize=(8, 5))

    sns.histplot(
        dataframe["Fare"],
        bins=30,
        kde=True,
        color="orange"
    )

    plt.title("Fare Distribution")

    plt.tight_layout()

    plt.savefig(IMAGES_DIR / "fare_distribution.png")

    plt.close()


def plot_boxplots(dataframe: pd.DataFrame):
    """
    Plot boxplots for numerical columns.
    """

    numerical_columns = [
        "Age",
        "Fare",
        "SibSp",
        "Parch"
    ]

    for column in numerical_columns:

        plt.figure(figsize=(6, 4))

        sns.boxplot(
            y=dataframe[column],
            color="skyblue"
        )

        plt.title(f"{column} Boxplot")

        plt.tight_layout()

        plt.savefig(IMAGES_DIR / f"{column.lower()}_boxplot.png")

        plt.close()


def plot_correlation_heatmap(dataframe: pd.DataFrame):
    """
    Plot correlation heatmap.
    """

    plt.figure(figsize=(10, 8))

    correlation = dataframe.corr(numeric_only=True)

    sns.heatmap(
        correlation,
        annot=True,
        cmap="coolwarm",
        linewidths=0.5
    )

    plt.title("Correlation Heatmap")

    plt.tight_layout()

    plt.savefig(CORRELATION_HEATMAP_IMAGE)

    plt.close()


def perform_eda(dataframe: pd.DataFrame):
    """
    Execute complete EDA.
    """

    print("=" * 60)
    print("Performing Exploratory Data Analysis...")
    print("=" * 60)

    plot_survival_distribution(dataframe)

    plot_age_distribution(dataframe)

    plot_fare_distribution(dataframe)

    plot_boxplots(dataframe)

    plot_correlation_heatmap(dataframe)

    print("EDA completed successfully.")
    print(f"Graphs saved inside: {IMAGES_DIR}")