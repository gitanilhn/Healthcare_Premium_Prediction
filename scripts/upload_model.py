from pathlib import Path

import boto3

# -----------------------------
# Configuration
# -----------------------------

BUCKET = "healthcare-premium-mlops-anil"

PREFIX = "models/healthcare-premium-prediction/v1"

ARTIFACT_DIR = Path("artifacts/latest")

FILES = [
    "model.pkl",
    "preprocessor.pkl",
    "metadata.json",
    "feature_schema.json",
]

# -----------------------------
# Upload
# -----------------------------

s3 = boto3.client("s3")

print("=" * 60)
print("Uploading Model Artifacts")
print("=" * 60)

for file in FILES:

    local_file = ARTIFACT_DIR / file

    if not local_file.exists():
        raise FileNotFoundError(local_file)

    s3_key = f"{PREFIX}/{file}"

    print(f"Uploading {file}")

    s3.upload_file(
        str(local_file),
        BUCKET,
        s3_key,
    )

print()
print("Upload Complete")
print(f"s3://{BUCKET}/{PREFIX}/")