from pathlib import Path
import os

# ============================================================
# Project Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ARTIFACT_ROOT = PROJECT_ROOT / "artifacts"
LATEST_ARTIFACT_DIR = ARTIFACT_ROOT / "latest"

LATEST_ARTIFACT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

# ============================================================
# Model Configuration
# ============================================================

MODEL_SOURCE = os.getenv(
    "MODEL_SOURCE",
    "local",          # change to "s3" in production
).lower()

MODEL_VERSION = os.getenv(
    "MODEL_VERSION",
    "v2",
)

MODEL_BUCKET = os.getenv(
    "MODEL_BUCKET",
    "healthcare-premium-mlops-anil",
)

AWS_REGION = os.getenv(
    "AWS_REGION",
    "ap-south-1",
)

MODEL_PREFIX = os.getenv(
    "MODEL_PATH",
    "models/healthcare-premium-prediction/v1",
)

# ============================================================
# Local Artifact Paths
# ============================================================

MODEL_FILE = LATEST_ARTIFACT_DIR / "model.pkl"

PREPROCESSOR_FILE = LATEST_ARTIFACT_DIR / "preprocessor.pkl"

METADATA_FILE = LATEST_ARTIFACT_DIR / "metadata.json"

FEATURE_SCHEMA_FILE = LATEST_ARTIFACT_DIR / "feature_schema.json"

# ============================================================
# Required Artifacts
# ============================================================

REQUIRED_ARTIFACTS = {
    "model.pkl": MODEL_FILE,
    "preprocessor.pkl": PREPROCESSOR_FILE,
    "metadata.json": METADATA_FILE,
    "feature_schema.json": FEATURE_SCHEMA_FILE,
}