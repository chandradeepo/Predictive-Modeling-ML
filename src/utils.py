"""
===========================================================
Project: Predictive Modeling Using Machine Learning

Description:
    Utility functions used throughout the project.

Author: Chandradeep
===========================================================
"""

from datetime import datetime


def print_header(title: str):
    """
    Print a formatted section header.
    """

    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def current_timestamp():
    """
    Return current timestamp.
    """

    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def print_success(message: str):
    """
    Print success message.
    """

    print(f"[SUCCESS] {message}")


def print_info(message: str):
    """
    Print informational message.
    """

    print(f"[INFO] {message}")