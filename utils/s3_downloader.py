import shutil

import boto3

from botocore.exceptions import ClientError

from config.settings import (
    MODEL_BUCKET,
    MODEL_PREFIX,
    AWS_REGION,
    LATEST_ARTIFACT_DIR,
)

FILES = [
    "model.pkl",
    "preprocessor.pkl",
    "metadata.json",
    "feature_schema.json",
]


class S3ModelDownloader:

    def __init__(self):

        self.client = boto3.client(
            "s3",
            region_name=AWS_REGION,
        )

    def artifacts_exist(self):

        return all(
            (LATEST_ARTIFACT_DIR / file).exists()
            for file in FILES
        )

    def download(self):

        if self.artifacts_exist():

            print("Artifacts already exist.")
            print("Skipping S3 download.")
            return

        print("=" * 70)
        print("Downloading model artifacts from S3")
        print("=" * 70)

        if LATEST_ARTIFACT_DIR.exists():

            shutil.rmtree(
                LATEST_ARTIFACT_DIR
            )

        LATEST_ARTIFACT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        try:

            for file in FILES:

                key = f"{MODEL_PREFIX}/{file}"

                print(f"Downloading {file}")

                self.client.download_file(
                    MODEL_BUCKET,
                    key,
                    str(LATEST_ARTIFACT_DIR / file),
                )

            print()
            print("Download completed.")

        except ClientError as e:

            shutil.rmtree(
                LATEST_ARTIFACT_DIR,
                ignore_errors=True,
            )

            raise RuntimeError(
                f"Unable to download artifacts from S3: {e}"
            )