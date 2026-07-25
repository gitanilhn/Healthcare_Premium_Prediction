import json
import joblib
import numpy as np
import pandas as pd

import os
import shutil
import tempfile

from pathlib import Path

import boto3
from botocore.exceptions import BotoCoreError, ClientError


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ------------------------------------------------------------
# Model loading mode
#
# Supported values:
#
#   s3       -> Download artifacts from S3
#   local    -> Load artifacts from local MODEL_LOCAL_DIR
#   disabled -> Do not load model
#
# Default:
#   s3 when MODEL_BUCKET and MODEL_PATH are configured
#   disabled otherwise
#
# This prevents pytest/CI from trying to access S3 during
# module import.
# ------------------------------------------------------------

MODEL_LOAD_MODE = os.getenv("MODEL_LOAD_MODE")

MODEL_BUCKET = os.getenv("MODEL_BUCKET")
MODEL_PATH = os.getenv("MODEL_PATH")

AWS_DEFAULT_REGION = os.getenv(
    "AWS_DEFAULT_REGION",
    "ap-south-1",
)

# ------------------------------------------------------------
# Local runtime cache
#
# Production:
#   /tmp/model
#
# Local development:
#   Can be overridden with MODEL_LOCAL_DIR
#
# CI:
#   Can use MODEL_LOAD_MODE=disabled
# ------------------------------------------------------------

ARTIFACT_DIR = Path(
    os.getenv(
        "MODEL_LOCAL_DIR",
        "/tmp/model",
    )
).resolve()

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


# ============================================================
# Determine Model Loading Mode
# ============================================================


def get_model_load_mode():
    """
    Determine how model artifacts should be loaded.

    Priority:

    1. Explicit MODEL_LOAD_MODE environment variable
    2. S3 if MODEL_BUCKET and MODEL_PATH are configured
    3. Disabled otherwise

    Examples:

    Production:
        MODEL_LOAD_MODE=s3

    Local:
        MODEL_LOAD_MODE=local

    CI:
        MODEL_LOAD_MODE=disabled
    """

    if MODEL_LOAD_MODE:
        return MODEL_LOAD_MODE.lower().strip()

    if MODEL_BUCKET and MODEL_PATH:
        return "s3"

    return "disabled"


# ============================================================
# S3 Utilities
# ============================================================


def _normalise_s3_prefix(prefix):
    """
    Return an S3 prefix without leading/trailing slashes.
    """

    if prefix is None:
        return ""

    return prefix.strip("/")


def download_model_artifacts():
    """
    Download the selected model artifacts from S3
    into the local runtime artifact directory.

    This function is explicitly called only when
    S3 model loading is required.

    It is NOT called automatically when this module
    is imported.
    """

    if not MODEL_BUCKET:
        raise RuntimeError(
            "MODEL_BUCKET environment variable is not configured. "
            "Set MODEL_BUCKET to the S3 bucket containing "
            "the model artifacts."
        )

    if not MODEL_PATH:
        raise RuntimeError(
            "MODEL_PATH environment variable is not configured. "
            "Set MODEL_PATH to the S3 model/version prefix."
        )

    s3_prefix = _normalise_s3_prefix(MODEL_PATH)

    print("Model artifact source:")
    print(f"  S3 Bucket : {MODEL_BUCKET}")
    print(f"  S3 Path   : " f"s3://{MODEL_BUCKET}/{s3_prefix}/")
    print(f"  Local Dir : {ARTIFACT_DIR}")

    # --------------------------------------------------------
    # Start from a clean runtime directory.
    #
    # This prevents the application from accidentally
    # loading artifacts belonging to an older model version.
    # --------------------------------------------------------

    if ARTIFACT_DIR.exists():

        shutil.rmtree(
            ARTIFACT_DIR,
            ignore_errors=True,
        )

    ARTIFACT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Create S3 client
    #
    # boto3 will automatically use:
    #
    # - IAM Role
    # - Environment credentials
    # - AWS CLI credentials
    # - EKS IRSA / Pod Identity
    #
    # depending on the runtime environment.
    # --------------------------------------------------------

    s3_client = boto3.client(
        "s3",
        region_name=AWS_DEFAULT_REGION,
    )

    downloaded_files = []

    try:

        for (
            filename,
            local_path,
        ) in S3_ARTIFACT_FILES.items():

            if s3_prefix:

                s3_key = f"{s3_prefix}/{filename}"

            else:

                s3_key = filename

            print(f"Downloading: " f"s3://{MODEL_BUCKET}/{s3_key}")

            # ------------------------------------------------
            # Download to temporary file first.
            #
            # This prevents partially downloaded artifacts
            # from being treated as valid model files.
            # ------------------------------------------------

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
            "Failed to download model artifacts "
            "from S3. "
            f"Bucket='{MODEL_BUCKET}', "
            f"Path='{MODEL_PATH}'. "
            f"AWS error: {exc}"
        ) from exc

    except Exception:

        shutil.rmtree(
            ARTIFACT_DIR,
            ignore_errors=True,
        )

        raise

    # --------------------------------------------------------
    # Validate downloaded artifacts
    # --------------------------------------------------------

    missing_files = [
        filename
        for (
            filename,
            local_path,
        ) in S3_ARTIFACT_FILES.items()
        if not local_path.is_file()
    ]

    if missing_files:

        shutil.rmtree(
            ARTIFACT_DIR,
            ignore_errors=True,
        )

        raise FileNotFoundError(
            "Model download completed but "
            "required artifacts are missing: "
            f"{missing_files}"
        )

    print(f"Downloaded artifacts: " f"{downloaded_files}")

    print("S3 model artifacts downloaded successfully")


# ============================================================
# Local Artifact Validation
# ============================================================


def validate_local_artifacts():
    """
    Validate that all required local model artifacts exist.
    """

    required_files = [
        LOCAL_MODEL_PATH,
        LOCAL_PREPROCESSOR_PATH,
        LOCAL_METADATA_PATH,
        LOCAL_FEATURE_SCHEMA_PATH,
    ]

    missing_files = [
        str(file_path) for file_path in required_files if not file_path.is_file()
    ]

    if missing_files:

        raise FileNotFoundError(
            "Required model artifacts are missing: " f"{missing_files}"
        )


# ============================================================
# Prediction Service
# ============================================================


class PredictionService:

    def __init__(self):

        print("=" * 70)
        print("Initializing Healthcare Premium Prediction Service")
        print("=" * 70)

        self.model = None
        self.preprocessor = None
        self.metadata = None
        self.feature_schema = None

        self.scaler = None
        self.feature_columns = None
        self.scaling_columns = None

        self.is_loaded = False

        self.load_model()

    # ========================================================
    # Model Loading
    # ========================================================

    def load_model(
        self,
        mode=None,
    ):
        """
        Load model artifacts.

        mode options:

            s3
            local
            disabled

        If mode is not supplied, it is automatically
        determined from environment variables.
        """

        if self.is_loaded:

            print("Model is already loaded. " "Skipping reload.")

            return

        selected_mode = mode if mode else get_model_load_mode()

        selected_mode = selected_mode.lower().strip()

        print(f"Model loading mode: " f"{selected_mode}")

        # ----------------------------------------------------
        # Disabled mode
        #
        # Used by CI during import/test collection.
        # ----------------------------------------------------

        if selected_mode == "disabled":

            print("Model loading is disabled.")

            print("PredictionService initialized " "without loading model artifacts.")

            return

        # ----------------------------------------------------
        # S3 mode
        # ----------------------------------------------------

        if selected_mode == "s3":

            download_model_artifacts()

        # ----------------------------------------------------
        # Local mode
        # ----------------------------------------------------

        elif selected_mode == "local":

            print("Loading model artifacts " "from local directory:")

            print(f"  {ARTIFACT_DIR}")

            validate_local_artifacts()

        else:

            raise ValueError(
                "Invalid MODEL_LOAD_MODE. "
                f"Received: '{selected_mode}'. "
                "Supported values are: "
                "'s3', 'local', 'disabled'."
            )

        # ----------------------------------------------------
        # Validate artifacts
        # ----------------------------------------------------

        validate_local_artifacts()

        # ----------------------------------------------------
        # Load metadata
        # ----------------------------------------------------

        with open(
            LOCAL_METADATA_PATH,
            "r",
            encoding="utf-8",
        ) as f:

            self.metadata = json.load(f)

        # ----------------------------------------------------
        # Load feature schema
        # ----------------------------------------------------

        with open(
            LOCAL_FEATURE_SCHEMA_PATH,
            "r",
            encoding="utf-8",
        ) as f:

            self.feature_schema = json.load(f)

        # ----------------------------------------------------
        # Load model
        # ----------------------------------------------------

        self.model = joblib.load(LOCAL_MODEL_PATH)

        # ----------------------------------------------------
        # Load preprocessor
        # ----------------------------------------------------

        self.preprocessor = joblib.load(LOCAL_PREPROCESSOR_PATH)

        # ----------------------------------------------------
        # Validate preprocessor
        # ----------------------------------------------------

        if not isinstance(
            self.preprocessor,
            dict,
        ):

            raise TypeError("preprocessor.pkl must contain " "a dictionary")

        required_preprocessor_keys = [
            "scaler",
            "feature_columns",
            "scaling_columns",
        ]

        for key in required_preprocessor_keys:

            if key not in self.preprocessor:

                raise ValueError(f"Missing preprocessor key: " f"{key}")

        # ----------------------------------------------------
        # Extract preprocessing information
        # ----------------------------------------------------

        self.scaler = self.preprocessor["scaler"]

        self.feature_columns = self.preprocessor["feature_columns"]

        self.scaling_columns = self.preprocessor["scaling_columns"]

        # ----------------------------------------------------
        # Validate metadata feature schema
        # ----------------------------------------------------

        metadata_features = self.metadata.get(
            "feature_columns",
            [],
        )

        if metadata_features:

            if metadata_features != self.feature_columns:

                raise ValueError(
                    "Feature mismatch detected "
                    "between metadata.json and "
                    "preprocessor.pkl"
                )

        # ----------------------------------------------------
        # Validate scaler
        # ----------------------------------------------------

        scaler_feature_count = getattr(
            self.scaler,
            "n_features_in_",
            None,
        )

        if scaler_feature_count is not None and scaler_feature_count != len(
            self.scaling_columns
        ):

            raise ValueError(
                "Scaler feature count does not "
                "match preprocessor scaling "
                "columns. "
                f"Scaler expects "
                f"{scaler_feature_count}, "
                f"but scaling_columns contains "
                f"{len(self.scaling_columns)}."
            )

        # ----------------------------------------------------
        # Print model information
        # ----------------------------------------------------

        print(f"Model Version      : " f"{self.metadata.get('model_version')}")

        print(f"Algorithm          : " f"{self.metadata.get('algorithm')}")

        print(f"Expected Features  : " f"{len(self.feature_columns)}")

        print(f"Scaling Features   : " f"{len(self.scaling_columns)}")

        metrics = self.metadata.get(
            "metrics",
            {},
        )

        print(f"R2 Score           : " f"{metrics.get('R2', 'N/A')}")

        print(f"MAE                : " f"{metrics.get('MAE', 'N/A')}")

        print(f"RMSE               : " f"{metrics.get('RMSE', 'N/A')}")

        preprocessor_version = self.preprocessor.get(
            "preprocessor_version",
            "N/A",
        )
        print(f"Preprocessor Version : " f"{preprocessor_version}")

        print("=" * 70)

        print("Model loaded successfully")

        print("=" * 70)

        self.is_loaded = True

    # ========================================================
    # Feature Preparation
    # ========================================================

    def prepare_features(
        self,
        input_data,
    ):

        # ----------------------------------------------------
        # Make sure model is loaded
        # ----------------------------------------------------

        if not self.is_loaded:

            raise RuntimeError(
                "Model is not loaded. "
                "Call predictor.load_model() "
                "before making predictions."
            )

        # ----------------------------------------------------
        # Convert input to DataFrame
        # ----------------------------------------------------

        df = pd.DataFrame([input_data])

        # ----------------------------------------------------
        # Validate raw input fields
        # ----------------------------------------------------

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

            raise ValueError("Missing required input fields: " f"{missing_features}")

        # ====================================================
        # Feature Engineering
        # ====================================================

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

        # ----------------------------------------------------
        # Normalized risk score
        # ----------------------------------------------------

        risk_min = 0

        risk_max = 16

        if risk_max == risk_min:

            df["normalized_risk_score"] = 0.0

        else:

            df["normalized_risk_score"] = (df["total_risk_score"] - risk_min) / (
                risk_max - risk_min
            )

        # ----------------------------------------------------
        # Lifestyle risk score
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

        df["lifestyle_risk_score"] = df["physical_activity"].map(physical).fillna(
            0
        ) + df["stress_level"].map(stress).fillna(0)

        # ----------------------------------------------------
        # Label encoding
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
        # One-hot encoding
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Drop raw columns
        # ----------------------------------------------------

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

        # ====================================================
        # Ensure expected feature columns
        # ====================================================

        for feature in self.feature_columns:

            if feature not in df.columns:

                df[feature] = 0

        # ----------------------------------------------------
        # Remove unexpected columns
        # ----------------------------------------------------

        df = df[self.feature_columns]

        # ----------------------------------------------------
        # Convert to numeric
        # ----------------------------------------------------

        df = df.apply(
            pd.to_numeric,
            errors="coerce",
        )

        # ----------------------------------------------------
        # Validate missing values
        # ----------------------------------------------------

        if df.isnull().any().any():

            invalid_columns = df.columns[df.isnull().any()].tolist()

            raise ValueError(
                "Invalid or missing values " "found in features: " f"{invalid_columns}"
            )

        # ====================================================
        # Apply trained scaler
        # ====================================================

        scaling_columns_available = [
            column for column in self.scaling_columns if column in df.columns
        ]

        # ----------------------------------------------------
        # Ensure all scaling columns exist
        # ----------------------------------------------------

        if len(scaling_columns_available) != len(self.scaling_columns):

            raise ValueError(
                "Not all scaling columns "
                "are available. "
                f"Expected: "
                f"{self.scaling_columns}, "
                f"Available: "
                f"{scaling_columns_available}"
            )

        # ----------------------------------------------------
        # Transform exactly the columns used during training
        # ----------------------------------------------------

        df[self.scaling_columns] = self.scaler.transform(df[self.scaling_columns])

        # ----------------------------------------------------
        # Final feature validation
        # ----------------------------------------------------

        if list(df.columns) != self.feature_columns:

            raise ValueError(
                "Final feature order does not " "match training feature order"
            )

        return df

    # ========================================================
    # Prediction
    # ========================================================

    def predict(
        self,
        input_data,
    ):

        try:

            # ------------------------------------------------
            # Ensure model is loaded
            # ------------------------------------------------

            if not self.is_loaded:

                raise RuntimeError(
                    "Model is not loaded. "
                    "Call predictor.load_model() "
                    "before making predictions."
                )

            # ------------------------------------------------
            # Prepare features
            # ------------------------------------------------

            X = self.prepare_features(input_data)

            # ------------------------------------------------
            # Generate prediction
            # ------------------------------------------------

            prediction = self.model.predict(X)

            prediction_value = float(prediction[0])

            # ------------------------------------------------
            # Validate prediction
            # ------------------------------------------------

            if not np.isfinite(prediction_value):

                raise ValueError("Model returned an invalid " "prediction")

            if prediction_value < 0:

                raise ValueError("Model returned a negative " "premium")

            # ------------------------------------------------
            # Return response
            # ------------------------------------------------

            return {
                "prediction": round(
                    prediction_value,
                    2,
                ),
                "model_version": (self.metadata.get("model_version")),
                "algorithm": (self.metadata.get("algorithm")),
                "model_metrics": (
                    self.metadata.get(
                        "metrics",
                        {},
                    )
                ),
            }

        except Exception as e:

            raise RuntimeError(f"Prediction failed: {str(e)}") from e


# ============================================================
# Global Predictor
#
# IMPORTANT:
#
# Do NOT load S3 here.
#
# This object can safely be imported by pytest:
#
#     from src.prediction import predictor
#
# No AWS credentials are required during import.
#
# Production startup should explicitly call:
#
#     predictor.load_model()
#
# ============================================================

predictor = PredictionService()
