"""
Healthcare Premium Prediction Service

Production Architecture

Supports

- Local Development
- Docker
- Kubernetes
- AWS S3
- GitHub Actions
- ArgoCD
- Amazon EKS

Author:
Anil MLOps Project
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile

from pathlib import Path
from typing import Dict

import boto3
import joblib

from botocore.exceptions import (
    BotoCoreError,
    ClientError,
)

import numpy as np
import pandas as pd

from config.settings import (
    MODEL_SOURCE,
    MODEL_BUCKET,
    MODEL_VERSION,
    AWS_REGION,
    LATEST_ARTIFACT_DIR,
    MODEL_FILE,
    PREPROCESSOR_FILE,
    METADATA_FILE,
    FEATURE_SCHEMA_FILE,
)

# ==========================================================
# Logger
# ==========================================================

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

# ==========================================================
# Local Artifact Paths
# ==========================================================

ARTIFACT_DIR = LATEST_ARTIFACT_DIR

LOCAL_MODEL_PATH = MODEL_FILE

LOCAL_PREPROCESSOR_PATH = PREPROCESSOR_FILE

LOCAL_METADATA_PATH = METADATA_FILE

LOCAL_FEATURE_SCHEMA_PATH = FEATURE_SCHEMA_FILE

# ==========================================================
# S3 Folder Structure
#
# Example
#
# healthcare-models/
#
#     healthcare_premium_prediction/
#
#         v1/
#             model.pkl
#             preprocessor.pkl
#             metadata.json
#             feature_schema.json
#
# ==========================================================

S3_MODEL_PREFIX = (
    f"healthcare_premium_prediction/{MODEL_VERSION}"
)

# ==========================================================
# Required Artifact Files
# ==========================================================

S3_ARTIFACT_FILES = {

    "model.pkl": LOCAL_MODEL_PATH,

    "preprocessor.pkl": LOCAL_PREPROCESSOR_PATH,

    "metadata.json": LOCAL_METADATA_PATH,

    "feature_schema.json": LOCAL_FEATURE_SCHEMA_PATH,

}

# ==========================================================
# Utility Functions
# ==========================================================


def _clean_artifact_directory() -> None:
    """
    Remove local artifacts.

    Used before downloading
    a new model from S3.
    """

    if ARTIFACT_DIR.exists():

        shutil.rmtree(
            ARTIFACT_DIR,
            ignore_errors=True,
        )

    ARTIFACT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


# ==========================================================
# Validate Local Artifacts
# ==========================================================


def validate_local_artifacts() -> None:
    """
    Ensure all required
    artifact files exist.
    """

    required_files = [

        LOCAL_MODEL_PATH,

        LOCAL_PREPROCESSOR_PATH,

        LOCAL_METADATA_PATH,

        LOCAL_FEATURE_SCHEMA_PATH,

    ]

    missing_files = [

        str(file)

        for file in required_files

        if not file.exists()

    ]

    if missing_files:

        raise FileNotFoundError(

            "Missing model artifacts:\n"

            + "\n".join(missing_files)

        )

    logger.info("Local artifacts validated successfully.")


# ==========================================================
# Download Model From S3
# ==========================================================


def download_model_artifacts() -> None:
    """
    Download model artifacts
    from S3.

    The destination is

    artifacts/latest/

    Existing artifacts
    are removed first.
    """

    if not MODEL_BUCKET:

        raise RuntimeError(

            "MODEL_BUCKET is not configured."

        )

    logger.info("=" * 70)

    logger.info("Downloading model artifacts from S3")

    logger.info("=" * 70)

    logger.info(f"Bucket  : {MODEL_BUCKET}")

    logger.info(f"Version : {MODEL_VERSION}")

    logger.info(f"Prefix  : {S3_MODEL_PREFIX}")

    _clean_artifact_directory()

    s3_client = boto3.client(

        "s3",

        region_name=AWS_REGION,

    )

    downloaded_files = []

    try:

        for filename, local_path in S3_ARTIFACT_FILES.items():

            s3_key = f"{S3_MODEL_PREFIX}/{filename}"

            logger.info(

                f"Downloading {s3_key}"

            )

            temp_file = tempfile.NamedTemporaryFile(

                delete=False,

                dir=ARTIFACT_DIR,

                suffix=".tmp",

            )

            temp_file.close()

            try:

                s3_client.download_file(

                    MODEL_BUCKET,

                    s3_key,

                    temp_file.name,

                )

                os.replace(

                    temp_file.name,

                    local_path,

                )

            finally:

                if os.path.exists(temp_file.name):

                    os.remove(temp_file.name)

            downloaded_files.append(filename)

    except (

        ClientError,

        BotoCoreError,

    ) as exc:

        shutil.rmtree(

            ARTIFACT_DIR,

            ignore_errors=True,

        )

        raise RuntimeError(

            "Unable to download model "

            f"version {MODEL_VERSION}"

        ) from exc

    validate_local_artifacts()

    logger.info("Downloaded Files")

    for file in downloaded_files:

        logger.info(f"   {file}")

    logger.info("=" * 70)

    logger.info("Model download completed.")

    logger.info("=" * 70)

# ==========================================================
# Prediction Service
# ==========================================================


class PredictionService:
    """
    Production Prediction Service

    Responsible for

    - Downloading model (if required)
    - Loading artifacts
    - Validating artifacts
    - Loading metadata
    - Loading preprocessor
    - Loading trained model
    """

    def __init__(self) -> None:

        self.model = None

        self.preprocessor = None

        self.metadata: Dict = {}

        self.feature_schema: Dict = {}

        self.scaler = None

        self.feature_columns = []

        self.scaling_columns = []

        self.is_loaded = False

        logger.info("=" * 70)
        logger.info("Healthcare Premium Prediction Service")
        logger.info("=" * 70)

        self.load_model()

    # ======================================================
    # Load Model
    # ======================================================

    def load_model(
        self,
        mode: str | None = None,
    ) -> None:
        """
        Load model artifacts.

        Supported modes

        local
        s3
        """

        if self.is_loaded:

            logger.info("Model already loaded.")

            return

        selected_mode = (mode or MODEL_SOURCE).lower().strip()

        logger.info(f"Model Source : {selected_mode}")

        # ---------------------------------------------
        # Download From S3
        # ---------------------------------------------

        if selected_mode == "s3":

            download_model_artifacts()

        # ---------------------------------------------
        # Local Artifacts
        # ---------------------------------------------

        elif selected_mode == "local":

            validate_local_artifacts()

        else:

            raise ValueError(

                f"Unsupported MODEL_SOURCE : {selected_mode}"

            )

        # ---------------------------------------------
        # Final Validation
        # ---------------------------------------------

        validate_local_artifacts()

        # ---------------------------------------------
        # Load Metadata
        # ---------------------------------------------

        with open(

            LOCAL_METADATA_PATH,

            "r",

            encoding="utf-8",

        ) as f:

            self.metadata = json.load(f)

        # ---------------------------------------------
        # Load Feature Schema
        # ---------------------------------------------

        with open(

            LOCAL_FEATURE_SCHEMA_PATH,

            "r",

            encoding="utf-8",

        ) as f:

            self.feature_schema = json.load(f)

        # ---------------------------------------------
        # Load Model
        # ---------------------------------------------

        self.model = joblib.load(

            LOCAL_MODEL_PATH

        )

        # ---------------------------------------------
        # Load Preprocessor
        # ---------------------------------------------

        self.preprocessor = joblib.load(

            LOCAL_PREPROCESSOR_PATH

        )

        # ---------------------------------------------
        # Validate Preprocessor
        # ---------------------------------------------

        if not isinstance(

            self.preprocessor,

            dict,

        ):

            raise TypeError(

                "preprocessor.pkl must contain a dictionary."

            )

        required_keys = [

            "scaler",

            "feature_columns",

            "scaling_columns",

        ]

        for key in required_keys:

            if key not in self.preprocessor:

                raise ValueError(

                    f"Missing preprocessor key : {key}"

                )

        # ---------------------------------------------
        # Extract Objects
        # ---------------------------------------------

        self.scaler = self.preprocessor["scaler"]

        self.feature_columns = self.preprocessor["feature_columns"]

        self.scaling_columns = self.preprocessor["scaling_columns"]

        # ---------------------------------------------
        # Metadata Validation
        # ---------------------------------------------

        metadata_features = self.metadata.get(

            "feature_columns",

            [],

        )

        if metadata_features:

            if metadata_features != self.feature_columns:

                raise ValueError(

                    "Feature mismatch between "

                    "metadata.json and preprocessor.pkl"

                )

        # ---------------------------------------------
        # Validate Scaler
        # ---------------------------------------------

        scaler_features = getattr(

            self.scaler,

            "n_features_in_",

            None,

        )

        if scaler_features is not None:

            if scaler_features != len(

                self.scaling_columns

            ):

                raise ValueError(

                    "Scaler feature mismatch."

                )

        self.is_loaded = True

        self._print_model_summary()

    # ======================================================
    # Print Model Summary
    # ======================================================

    def _print_model_summary(self) -> None:

        metrics = self.metadata.get(

            "metrics",

            {},

        )

        logger.info("")

        logger.info("=" * 70)

        logger.info("Current Loaded Model")

        logger.info("=" * 70)

        logger.info(

            f"Source               : {MODEL_SOURCE}"

        )

        logger.info(

            f"Version              : {MODEL_VERSION}"

        )

        logger.info(

            f"Algorithm            : {self.metadata.get('algorithm')}"

        )

        logger.info(

            f"Expected Features    : {len(self.feature_columns)}"

        )

        logger.info(

            f"Scaling Features     : {len(self.scaling_columns)}"

        )

        logger.info(

            f"R2 Score             : {metrics.get('R2')}"

        )

        logger.info(

            f"MAE                  : {metrics.get('MAE')}"

        )

        logger.info(

            f"RMSE                 : {metrics.get('RMSE')}"

        )

        logger.info(

            f"Artifacts            : {ARTIFACT_DIR}"

        )

        logger.info("=" * 70)

        logger.info("Prediction Service Ready")

        logger.info("=" * 70)


    # ========================================================
    # Feature Preparation
    # ========================================================

    def prepare_features(
        self,
        input_data,
    ):

        if not self.is_loaded:
            raise RuntimeError(
                "Model is not loaded. Call predictor.load_model() before making predictions."
            )

        df = pd.DataFrame([input_data])

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

        missing = [
            feature
            for feature in required_raw_features
            if feature not in df.columns
        ]

        if missing:
            raise ValueError(
                f"Missing required input fields: {missing}"
            )

        # ----------------------------------------------------
        # Medical history
        # ----------------------------------------------------

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
            .str.split(" & ", expand=True)
        )

        df["disease1"] = split_history[0].fillna("none").str.lower()

        if 1 in split_history.columns:
            df["disease2"] = (
                split_history[1]
                .fillna("none")
                .str.lower()
            )
        else:
            df["disease2"] = "none"

        df["total_risk_score"] = (
            df["disease1"].map(risk_scores).fillna(0)
            + df["disease2"].map(risk_scores).fillna(0)
        )

        # ----------------------------------------------------
        # Normalized risk score
        # ----------------------------------------------------

        df["normalized_risk_score"] = (
            df["total_risk_score"] / 16
        )

        # ----------------------------------------------------
        # Lifestyle score
        # ----------------------------------------------------

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

        df["lifestyle_risk_score"] = (
            df["physical_activity"]
            .map(physical)
            .fillna(0)
            +
            df["stress_level"]
            .map(stress)
            .fillna(0)
        )

        # ----------------------------------------------------
        # Label Encoding
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # One Hot Encoding
        # ----------------------------------------------------

        df = pd.get_dummies(
            df,
            columns=[
                "gender",
                "region",
                "marital_status",
                "bmi_category",
                "smoking_status",
                "employment_status",
            ],
            drop_first=True,
            dtype=int,
        )

        # ----------------------------------------------------
        # Remove training helper columns
        # ----------------------------------------------------

        df.drop(
            columns=[
                "medical_history",
                "physical_activity",
                "stress_level",
                "disease1",
                "disease2",
                "total_risk_score",
            ],
            errors="ignore",
            inplace=True,
        )

        # ----------------------------------------------------
        # Missing Columns
        # ----------------------------------------------------

        for column in self.feature_columns:

            if column not in df.columns:
                df[column] = 0

        df = df[self.feature_columns]

        df = df.apply(
            pd.to_numeric,
            errors="coerce",
        )

        if df.isnull().any().any():

            invalid_columns = (
                df.columns[df.isnull().any()]
                .tolist()
            )

            raise ValueError(
                f"Invalid values found in {invalid_columns}"
            )

        # ----------------------------------------------------
        # Scaling
        # ----------------------------------------------------

        df[self.scaling_columns] = (
            self.scaler.transform(
                df[self.scaling_columns]
            )
        )

        return df

    # ========================================================
    # Prediction
    # ========================================================

    def predict(
        self,
        input_data,
    ):

        if not self.is_loaded:
            raise RuntimeError(
                "Model is not loaded. Call predictor.load_model() before making predictions."
            )

        try:

            X = self.prepare_features(input_data)

            prediction = self.model.predict(X)

            prediction = float(prediction[0])

            if not np.isfinite(prediction):
                raise ValueError(
                    "Invalid prediction generated."
                )

            if prediction < 0:
                raise ValueError(
                    "Negative premium generated."
                )

            return {
                "prediction": round(prediction, 2),
                "model_version": self.metadata.get(
                    "model_version"
                ),
                "algorithm": self.metadata.get(
                    "algorithm"
                ),
                "metrics": self.metadata.get(
                    "metrics",
                    {},
                ),
            }

        except Exception as e:
            raise RuntimeError(
                f"Prediction failed: {e}"
            ) from e


# ============================================================
# Global Predictor
# ============================================================

predictor = PredictionService()