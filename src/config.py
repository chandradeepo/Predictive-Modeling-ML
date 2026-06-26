"""
===========================================================
Project: Predictive Modeling Using Machine Learning
Dataset: Titanic - Machine Learning from Disaster

Description:
    This file stores all project configuration variables
    such as file paths, random seed, train-test split ratio,
    and model save location.

Author: Chandradeep
===========================================================
"""

from pathlib import Path

# ==========================================================
# Project Root Directory
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ==========================================================
# Folder Paths
# ==========================================================

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
IMAGES_DIR = PROJECT_ROOT / "images"
REPORTS_DIR = PROJECT_ROOT / "reports"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# ==========================================================
# Dataset Files
# ==========================================================

TRAIN_DATA = DATA_DIR / "train.csv"
TEST_DATA = DATA_DIR / "test.csv"

# ==========================================================
# Model Save Path
# ==========================================================

MODEL_FILE = MODELS_DIR / "titanic_model.pkl"

# ==========================================================
# Random State
# ==========================================================

RANDOM_STATE = 42

# ==========================================================
# Train-Test Split
# ==========================================================

TEST_SIZE = 0.20

# ==========================================================
# Target Column
# ==========================================================

TARGET_COLUMN = "Survived"

# ==========================================================
# Image Save Paths
# ==========================================================

CONFUSION_MATRIX_IMAGE = IMAGES_DIR / "confusion_matrix.png"
ROC_CURVE_IMAGE = IMAGES_DIR / "roc_curve.png"
FEATURE_IMPORTANCE_IMAGE = IMAGES_DIR / "feature_importance.png"
CORRELATION_HEATMAP_IMAGE = IMAGES_DIR / "correlation_heatmap.png"

# ==========================================================
# Create Required Directories (if they don't exist)
# ==========================================================

for directory in [
    DATA_DIR,
    MODELS_DIR,
    IMAGES_DIR,
    REPORTS_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)