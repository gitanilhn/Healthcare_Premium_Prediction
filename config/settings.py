from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# -------------------------------------------------------
# Environment
# -------------------------------------------------------

ENVIRONMENT = os.getenv("ENVIRONMENT", "development")

# local | s3
MODEL_SOURCE = os.getenv("MODEL_SOURCE", "local").lower()

AWS_REGION = os.getenv("AWS_REGION", "ap-south-1")

MODEL_BUCKET = os.getenv("MODEL_BUCKET", "")

MODEL_VERSION = os.getenv("MODEL_VERSION", "latest")

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5000",
)

# -------------------------------------------------------
# Local Artifacts
# -------------------------------------------------------

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

LATEST_ARTIFACT_DIR = ARTIFACTS_DIR / "latest"

MODEL_FILE = LATEST_ARTIFACT_DIR / "model.pkl"

PREPROCESSOR_FILE = LATEST_ARTIFACT_DIR / "preprocessor.pkl"

METADATA_FILE = LATEST_ARTIFACT_DIR / "metadata.json"

FEATURE_SCHEMA_FILE = LATEST_ARTIFACT_DIR / "feature_schema.json"