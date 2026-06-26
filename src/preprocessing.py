"""
===========================================================
Project: Predictive Modeling Using Machine Learning
Dataset: Titanic - Machine Learning from Disaster

Description:
    This module performs data preprocessing, including:
    - Handling missing values
    - Dropping unnecessary columns
    - Encoding categorical variables
    - Splitting features and target

Author: Chandradeep
===========================================================
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from config import TARGET_COLUMN


def clean_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess the dataset.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Original dataset.

    Returns
    -------
    pd.DataFrame
        Cleaned dataset.
    """

    df = dataframe.copy()

    # ==========================================
    # Drop unnecessary columns
    # ==========================================
    columns_to_drop = [
        "PassengerId",
        "Name",
        "Ticket",
        "Cabin"
    ]

    df.drop(columns=columns_to_drop, inplace=True)

    # ==========================================
    # Handle Missing Values
    # ==========================================

    # Age → Median
    df["Age"] = df["Age"].fillna(df["Age"].median())

    # Embarked → Mode
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

    # Fare (only useful for test dataset)
    if "Fare" in df.columns:
        df["Fare"] = df["Fare"].fillna(df["Fare"].median())

    # ==========================================
    # Encode Categorical Variables
    # ==========================================

    encoder = LabelEncoder()

    categorical_columns = [
        "Sex",
        "Embarked"
    ]

    for column in categorical_columns:
        df[column] = encoder.fit_transform(df[column])

    return df


def split_features_target(dataframe: pd.DataFrame):
    """
    Split dataset into features (X) and target (y).

    Returns
    -------
    X : pd.DataFrame
    y : pd.Series
    """

    X = dataframe.drop(columns=[TARGET_COLUMN])
    y = dataframe[TARGET_COLUMN]

    return X, y


def preprocess_training_data(dataframe: pd.DataFrame):
    """
    Complete preprocessing pipeline
    for training dataset.
    """

    cleaned_df = clean_data(dataframe)

    X, y = split_features_target(cleaned_df)

    return X, y


def preprocess_test_data(dataframe: pd.DataFrame):
    """
    Complete preprocessing pipeline
    for testing dataset.
    """

    cleaned_df = clean_data(dataframe)

    return cleaned_df