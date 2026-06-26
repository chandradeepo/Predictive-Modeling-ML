"""
===========================================================
Project: Predictive Modeling Using Machine Learning
Dataset: Titanic - Machine Learning from Disaster

Description:
    This module is responsible for loading the Titanic
    dataset and displaying basic information about it.

Author: Chandradeep
===========================================================
"""

from pathlib import Path

import pandas as pd

from config import TRAIN_DATA, TEST_DATA


def load_dataset(file_path: Path) -> pd.DataFrame:
    """
    Load a CSV dataset.

    Parameters
    ----------
    file_path : Path
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Loaded dataset.
    """

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    try:
        dataframe = pd.read_csv(file_path)
        print(f"Successfully loaded: {file_path.name}")
        return dataframe

    except Exception as error:
        raise RuntimeError(f"Error while reading dataset: {error}")


def dataset_summary(dataframe: pd.DataFrame) -> None:
    """
    Display basic information about the dataset.
    """

    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)

    print(f"Rows                : {dataframe.shape[0]}")
    print(f"Columns             : {dataframe.shape[1]}")
    print(f"Duplicate Rows      : {dataframe.duplicated().sum()}")

    print("\nMissing Values")
    print("-" * 60)
    print(dataframe.isnull().sum())

    print("\nData Types")
    print("-" * 60)
    print(dataframe.dtypes)

    print("\nFirst Five Records")
    print("-" * 60)
    print(dataframe.head())

    print("\nStatistical Summary")
    print("-" * 60)
    print(dataframe.describe(include="all"))


def load_training_data() -> pd.DataFrame:
    """
    Load the training dataset.
    """

    return load_dataset(TRAIN_DATA)


def load_testing_data() -> pd.DataFrame:
    """
    Load the testing dataset.
    """

    return load_dataset(TEST_DATA)


if __name__ == "__main__":

    train_df = load_training_data()

    dataset_summary(train_df)