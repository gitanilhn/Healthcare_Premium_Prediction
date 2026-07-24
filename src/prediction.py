import json
import joblib
import numpy as np
import pandas as pd

from pathlib import Path


# ============================================================
# Configuration
# ============================================================

import os
import shutil
import tempfile

import boto3
from botocore.exceptions import BotoCoreError, ClientError

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# S3 configuration. These values should be supplied through environment
# variables in Docker/Kubernetes and should not be hard-coded.
MODEL_BUCKET = os.getenv("MODEL_BUCKET")
MODEL_PATH = os.getenv("MODEL_PATH")
AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION", "ap-south-1")

# Local runtime cache. The container downloads the selected model version
# here at startup. This directory is ephemeral and does not require a
# Docker image rebuild when a new model is uploaded to S3.
ARTIFACT_DIR = Path(os.getenv("MODEL_LOCAL_DIR", "/tmp/model")).resolve()

LOCAL_MODEL_PATH = ARTIFACT_DIR / "model.pkl"
LOCAL_PREPROCESSOR_PATH = ARTIFACT_DIR / "preprocessor.pkl"
LOCAL_METADATA_PATH = ARTIFACT_DIR / "metadata.json"
LOCAL_FEATURE_SCHEMA_PATH = ARTIFACT_DIR / "feature_schema.json"

S3_ARTIFACT_FILES = {
    "model.pkl": LOCAL_MODEL_PATH,
    "preprocessor.pkl": LOCAL_PREPROCESSOR_PATH,
    "metadata.json": LOCAL_METADATA_PATH,
    "feature_schema.json": LOCAL_FEATURE_SCHEMA_PATH,
}


def _normalise_s3_prefix(prefix):
    """Return an S3 prefix without leading/trailing slashes."""
    return prefix.strip("/")


def download_model_artifacts():
    """Download the selected model artifacts from S3 to /tmp/model."""

    if not MODEL_BUCKET:
        raise RuntimeError(
            "MODEL_BUCKET environment variable is not configured. "
            "Set MODEL_BUCKET to the S3 bucket containing the model artifacts."
        )

    if not MODEL_PATH:
        raise RuntimeError(
            "MODEL_PATH environment variable is not configured. "
            "Set MODEL_PATH to the S3 model/version prefix."
        )

    s3_prefix = _normalise_s3_prefix(MODEL_PATH)

    print("Model artifact source:")
    print(f"  S3 Bucket : {MODEL_BUCKET}")
    print(f"  S3 Path   : s3://{MODEL_BUCKET}/{s3_prefix}/")
    print(f"  Local Dir : {ARTIFACT_DIR}")

    # Start from a clean runtime directory so the container cannot
    # accidentally use an artifact left by an older model version.
    if ARTIFACT_DIR.exists():
        shutil.rmtree(ARTIFACT_DIR)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    s3_client = boto3.client(
        "s3",
        region_name=AWS_DEFAULT_REGION,
    )

    downloaded_files = []

    try:
        for filename, local_path in S3_ARTIFACT_FILES.items():
            s3_key = f"{s3_prefix}/{filename}" if s3_prefix else filename

            print(f"Downloading: s3://{MODEL_BUCKET}/{s3_key}")

            # Download to a temporary file first. This prevents a partially
            # downloaded artifact from being treated as a valid model.
            temp_file = tempfile.NamedTemporaryFile(
                dir=ARTIFACT_DIR,
                prefix=f".{filename}.",
                suffix=".tmp",
                delete=False,
            )
            temp_file.close()

            try:
                s3_client.download_file(
                    MODEL_BUCKET,
                    s3_key,
                    temp_file.name,
                )
                os.replace(temp_file.name, local_path)
            finally:
                if os.path.exists(temp_file.name):
                    os.remove(temp_file.name)

            downloaded_files.append(filename)

    except (ClientError, BotoCoreError) as exc:
        shutil.rmtree(ARTIFACT_DIR, ignore_errors=True)
        raise RuntimeError(
            "Failed to download model artifacts from S3. "
            f"Bucket='{MODEL_BUCKET}', Path='{MODEL_PATH}'. "
            f"AWS error: {exc}"
        ) from exc
    except Exception:
        shutil.rmtree(ARTIFACT_DIR, ignore_errors=True)
        raise

    missing_files = [
        filename
        for filename, local_path in S3_ARTIFACT_FILES.items()
        if not local_path.is_file()
    ]

    if missing_files:
        shutil.rmtree(ARTIFACT_DIR, ignore_errors=True)
        raise FileNotFoundError(
            "Model download completed but required artifacts are missing: "
            f"{missing_files}"
        )

    print(f"Downloaded artifacts: {downloaded_files}")
    print("S3 model artifacts downloaded successfully")


# ============================================================
# Prediction Service
# ============================================================


class PredictionService:

    def __init__(self):

        print("=" * 70)
        print("Initializing Healthcare Premium Prediction Model")
        print("=" * 70)

        # --------------------------------------------------------
        # Download and validate production artifacts from S3
        # --------------------------------------------------------

        download_model_artifacts()

        required_files = [
            LOCAL_MODEL_PATH,
            LOCAL_PREPROCESSOR_PATH,
            LOCAL_METADATA_PATH,
            LOCAL_FEATURE_SCHEMA_PATH,
        ]

        for file_path in required_files:
            if not file_path.is_file():
                raise FileNotFoundError(
                    f"Required artifact missing after S3 download: {file_path}"
                )

        # --------------------------------------------------------
        # Load metadata
        # --------------------------------------------------------

        with open(
            LOCAL_METADATA_PATH,
            "r",
            encoding="utf-8",
        ) as f:

            self.metadata = json.load(f)

        # --------------------------------------------------------
        # Load feature schema
        # --------------------------------------------------------

        with open(
            LOCAL_FEATURE_SCHEMA_PATH,
            "r",
            encoding="utf-8",
        ) as f:

            self.feature_schema = json.load(f)

        # --------------------------------------------------------
        # Load model
        # --------------------------------------------------------

        self.model = joblib.load(LOCAL_MODEL_PATH)

        # --------------------------------------------------------
        # Load preprocessor
        # --------------------------------------------------------

        self.preprocessor = joblib.load(LOCAL_PREPROCESSOR_PATH)

        # --------------------------------------------------------
        # Validate preprocessor
        # --------------------------------------------------------

        if not isinstance(
            self.preprocessor,
            dict,
        ):

            raise TypeError("preprocessor.pkl must contain a dictionary")

        required_preprocessor_keys = [
            "scaler",
            "feature_columns",
            "scaling_columns",
        ]

        for key in required_preprocessor_keys:

            if key not in self.preprocessor:

                raise ValueError(f"Missing preprocessor key: {key}")

        # --------------------------------------------------------
        # Extract preprocessing information
        # --------------------------------------------------------

        self.scaler = self.preprocessor["scaler"]

        self.feature_columns = self.preprocessor["feature_columns"]

        self.scaling_columns = self.preprocessor["scaling_columns"]

        # --------------------------------------------------------
        # Validate metadata feature schema
        # --------------------------------------------------------

        metadata_features = self.metadata.get(
            "feature_columns",
            [],
        )

        if metadata_features:

            if metadata_features != self.feature_columns:

                raise ValueError(
                    "Feature mismatch detected between "
                    "metadata.json and preprocessor.pkl"
                )

        # --------------------------------------------------------
        # Validate scaler
        # --------------------------------------------------------

        scaler_feature_count = getattr(
            self.scaler,
            "n_features_in_",
            None,
        )

        if scaler_feature_count != len(self.scaling_columns):

            raise ValueError(
                "Scaler feature count does not match " "preprocessor scaling columns"
            )

        # --------------------------------------------------------
        # Print model information
        # --------------------------------------------------------

        print(f"Model Version      : " f"{self.metadata.get('model_version')}")

        print(f"Algorithm          : " f"{self.metadata.get('algorithm')}")

        print(f"Expected Features  : " f"{len(self.feature_columns)}")

        print(f"Scaling Features   : " f"{len(self.scaling_columns)}")

        print(f"R2 Score           : " f"{self.metadata['metrics']['R2']}")

        print(f"MAE                : " f"{self.metadata['metrics']['MAE']}")

        print(f"RMSE               : " f"{self.metadata['metrics']['RMSE']}")

        print(
            f"Preprocessor Version : "
            f"{self.preprocessor.get('preprocessor_version', 'N/A')}"
        )

        print("=" * 70)

        print("Model loaded successfully")

        print("=" * 70)

    # ============================================================
    # Feature Preparation
    # ============================================================

    def prepare_features(
        self,
        input_data,
    ):

        # --------------------------------------------------------
        # Convert input to DataFrame
        # --------------------------------------------------------

        df = pd.DataFrame([input_data])

        # --------------------------------------------------------
        # Validate raw input fields
        # --------------------------------------------------------

        required_raw_features = [
            "age",
            "number_of_dependants",
            "income_level",
            "income_lakhs",
            "insurance_plan",
            "medical_history",
            "physical_activity",
            "stress_level",
            "gender",
            "region",
            "marital_status",
            "bmi_category",
            "smoking_status",
            "employment_status",
        ]

        missing_features = [
            feature for feature in required_raw_features if feature not in df.columns
        ]

        if missing_features:

            raise ValueError(f"Missing required input fields: " f"{missing_features}")

        # ========================================================
        # Feature Engineering
        # ========================================================

        # --------------------------------------------------------
        # Medical history
        # --------------------------------------------------------

        risk_scores = {
            "diabetes": 6,
            "heart disease": 8,
            "high blood pressure": 6,
            "thyroid": 5,
            "no disease": 0,
            "none": 0,
        }

        split_history = (
            df["medical_history"]
            .fillna("none")
            .astype(str)
            .str.split(
                " & ",
                expand=True,
            )
        )

        df["disease1"] = split_history[0].fillna("none").str.lower()

        if 1 in split_history.columns:

            df["disease2"] = split_history[1].fillna("none").str.lower()

        else:

            df["disease2"] = "none"

        df["total_risk_score"] = df["disease1"].map(risk_scores).fillna(0) + df[
            "disease2"
        ].map(risk_scores).fillna(0)

        # --------------------------------------------------------
        # Normalized risk score
        #
        # Training risk scores are based on the dataset range.
        # For the current dataset:
        # minimum = 0
        # maximum = 16
        # --------------------------------------------------------

        risk_min = 0

        risk_max = 16

        if risk_max == risk_min:

            df["normalized_risk_score"] = 0.0

        else:

            df["normalized_risk_score"] = (df["total_risk_score"] - risk_min) / (
                risk_max - risk_min
            )

        # --------------------------------------------------------
        # Lifestyle risk score
        # --------------------------------------------------------

        physical = {
            "High": 0,
            "Medium": 1,
            "Low": 4,
        }

        stress = {
            "High": 4,
            "Medium": 1,
            "Low": 0,
        }

        df["lifestyle_risk_score"] = df["physical_activity"].map(physical).fillna(
            0
        ) + df["stress_level"].map(stress).fillna(0)

        # --------------------------------------------------------
        # Label encoding
        # --------------------------------------------------------

        df["insurance_plan"] = df["insurance_plan"].map(
            {
                "Bronze": 1,
                "Silver": 2,
                "Gold": 3,
            }
        )

        df["income_level"] = df["income_level"].map(
            {
                "<10L": 1,
                "10L - 25L": 2,
                "25L - 40L": 3,
                "> 40L": 4,
            }
        )

        # --------------------------------------------------------
        # One-hot encoding
        # --------------------------------------------------------

        categorical_columns = [
            "gender",
            "region",
            "marital_status",
            "bmi_category",
            "smoking_status",
            "employment_status",
        ]

        df = pd.get_dummies(
            df,
            columns=categorical_columns,
            drop_first=True,
            dtype=int,
        )

        # --------------------------------------------------------
        # Drop raw columns
        # --------------------------------------------------------

        columns_to_drop = [
            "medical_history",
            "disease1",
            "disease2",
            "total_risk_score",
            "physical_activity",
            "stress_level",
        ]

        df.drop(
            columns=columns_to_drop,
            axis=1,
            inplace=True,
            errors="ignore",
        )

        # ========================================================
        # Ensure expected feature columns
        # ========================================================

        for feature in self.feature_columns:

            if feature not in df.columns:

                df[feature] = 0

        # --------------------------------------------------------
        # Remove unexpected columns
        # --------------------------------------------------------

        df = df[self.feature_columns]

        # --------------------------------------------------------
        # Convert to numeric
        # --------------------------------------------------------

        df = df.apply(
            pd.to_numeric,
            errors="coerce",
        )

        # --------------------------------------------------------
        # Validate missing values
        # --------------------------------------------------------

        if df.isnull().any().any():

            invalid_columns = df.columns[df.isnull().any()].tolist()

            raise ValueError(
                "Invalid or missing values " "found in features: " f"{invalid_columns}"
            )

        # ========================================================
        # Apply trained scaler
        # ========================================================

        scaling_columns_available = [
            column for column in self.scaling_columns if column in df.columns
        ]

        # --------------------------------------------------------
        # Important:
        # The scaler was trained on 7 columns.
        # Therefore transform exactly those 7 columns
        # in exactly the same order.
        # --------------------------------------------------------

        if len(scaling_columns_available) != len(self.scaling_columns):

            raise ValueError(
                "Not all scaling columns are available. "
                f"Expected: {self.scaling_columns}, "
                f"Available: "
                f"{scaling_columns_available}"
            )

        df[self.scaling_columns] = self.scaler.transform(df[self.scaling_columns])

        # --------------------------------------------------------
        # Final feature validation
        # --------------------------------------------------------

        if list(df.columns) != self.feature_columns:

            raise ValueError(
                "Final feature order does not match " "training feature order"
            )

        return df

    # ============================================================
    # Prediction
    # ============================================================

    def predict(
        self,
        input_data,
    ):

        try:

            # ----------------------------------------------------
            # Prepare features
            # ----------------------------------------------------

            X = self.prepare_features(input_data)

            # ----------------------------------------------------
            # Generate prediction
            # ----------------------------------------------------

            prediction = self.model.predict(X)

            prediction_value = float(prediction[0])

            # ----------------------------------------------------
            # Validate prediction
            # ----------------------------------------------------

            if not np.isfinite(prediction_value):

                raise ValueError("Model returned an invalid prediction")

            if prediction_value < 0:

                raise ValueError("Model returned a negative premium")

            # ----------------------------------------------------
            # Return response
            # ----------------------------------------------------

            return {
                "prediction": round(
                    prediction_value,
                    2,
                ),
                "model_version": (self.metadata["model_version"]),
                "algorithm": (self.metadata["algorithm"]),
                "model_metrics": (self.metadata["metrics"]),
            }

        except Exception as e:

            raise RuntimeError(f"Prediction failed: {str(e)}") from e


# ============================================================
# Global Predictor
# ============================================================

predictor = PredictionService()
