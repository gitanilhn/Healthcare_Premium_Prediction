from pathlib import Path


# ============================================================
# Project Root
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ============================================================
# Data Paths
# ============================================================

DATA_DIR = PROJECT_ROOT / "data"

DATA_FILE = DATA_DIR / "premiums_with_life_style.xlsx"


# ============================================================
# Artifact Paths
# ============================================================

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

LATEST_ARTIFACT_DIR = ARTIFACTS_DIR / "latest"


# ============================================================
# Model Artifacts
# ============================================================

MODEL_PATH = LATEST_ARTIFACT_DIR / "model.pkl"

PREPROCESSOR_PATH = LATEST_ARTIFACT_DIR / "preprocessor.pkl"

METADATA_PATH = LATEST_ARTIFACT_DIR / "metadata.json"

FEATURE_SCHEMA_PATH = LATEST_ARTIFACT_DIR / "feature_schema.json"


# ============================================================
# MLflow
# ============================================================

MLFLOW_TRACKING_URI = f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"

MLFLOW_EXPERIMENT_NAME = "Healthcare_Premium_Prediction"


# ============================================================
# Project Configuration
# ============================================================

PROJECT_NAME = "healthcare_premium_prediction"

TARGET_COLUMN = "annual_premium_amount"
