import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from config.settings import (
    MODEL_SOURCE,
    MODEL_VERSION,
    LATEST_ARTIFACT_DIR,
    MODEL_FILE,
    PREPROCESSOR_FILE,
    METADATA_FILE,
    FEATURE_SCHEMA_FILE,
)

from utils.s3_downloader import S3ModelDownloader


# ==========================================================
# Logging
# ==========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ==========================================================
# Artifact Directory
# ==========================================================

ARTIFACT_DIR = Path(
    LATEST_ARTIFACT_DIR
)


# ==========================================================
# Local Artifact Paths
# ==========================================================

LOCAL_MODEL_PATH = (
    ARTIFACT_DIR
    / Path(MODEL_FILE).name
)

LOCAL_PREPROCESSOR_PATH = (
    ARTIFACT_DIR
    / Path(PREPROCESSOR_FILE).name
)

LOCAL_METADATA_PATH = (
    ARTIFACT_DIR
    / Path(METADATA_FILE).name
)

LOCAL_FEATURE_SCHEMA_PATH = (
    ARTIFACT_DIR
    / Path(FEATURE_SCHEMA_FILE).name
)


# ==========================================================
# Validate Local Artifacts
# ==========================================================

def validate_local_artifacts():

    required_files = [

        LOCAL_MODEL_PATH,

        LOCAL_PREPROCESSOR_PATH,

        LOCAL_METADATA_PATH,

        LOCAL_FEATURE_SCHEMA_PATH,

    ]

    missing_files = [

        str(file_path)

        for file_path in required_files

        if not file_path.exists()

    ]

    if missing_files:

        raise FileNotFoundError(

            "Required model artifacts are missing:\n"

            + "\n".join(missing_files)

        )

    logger.info(

        "All required model artifacts validated successfully."

    )

    logger.info(

        f"Artifact directory: {ARTIFACT_DIR}"

    )


# ==========================================================
# Prediction Service
# ==========================================================

class PredictionService:

    def __init__(self):

        logger.info("=" * 70)

        logger.info(
            "Healthcare Premium Prediction Service"
        )

        logger.info("=" * 70)


        # --------------------------------------------------
        # Model Objects
        # --------------------------------------------------

        self.model = None

        self.preprocessor = None

        self.metadata = None

        self.feature_schema = None


        # --------------------------------------------------
        # Preprocessor Components
        # --------------------------------------------------

        self.scaler = None

        self.feature_columns = None

        self.scaling_columns = None


        # --------------------------------------------------
        # Service State
        # --------------------------------------------------

        self.is_loaded = False


        # --------------------------------------------------
        # Load Model
        # --------------------------------------------------

        self.load_model()


    # ======================================================
    # Model Version
    # ======================================================

    @property
    def model_version(self):

        if self.metadata:

            return self.metadata.get(

                "model_version",

                MODEL_VERSION,

            )

        return MODEL_VERSION


    # ======================================================
    # Load Model
    # ======================================================

    def load_model(self):

        if self.is_loaded:

            logger.info(

                "Model already loaded. "
                "Skipping reload."

            )

            return


        logger.info(

            f"Model Source : {MODEL_SOURCE}"

        )

        logger.info(

            f"Configured Model Version : "
            f"{MODEL_VERSION}"

        )

        logger.info(

            f"Artifact Directory : "
            f"{ARTIFACT_DIR}"

        )


        # ==================================================
        # 1. Download Artifacts
        # ==================================================

        if MODEL_SOURCE.lower() == "s3":

            logger.info(

                "Model source is S3."

            )

            downloader = (
                S3ModelDownloader()
            )


            if downloader.artifacts_exist():

                logger.info(

                    "Required artifacts "
                    "already exist locally."

                )

            else:

                logger.info(

                    "Required artifacts "
                    "not found locally."

                )

                logger.info(

                    "Downloading model artifacts "
                    "from S3..."

                )

                downloader.download()


        elif MODEL_SOURCE.lower() == "local":

            logger.info(

                "Model source is local."

            )

            logger.info(

                "Loading artifacts from "
                "local artifact directory."

            )


        else:

            raise ValueError(

                f"Unsupported MODEL_SOURCE: "
                f"{MODEL_SOURCE}. "

                "Expected 'local' or 's3'."

            )


        # ==================================================
        # 2. Validate Artifacts
        # ==================================================

        validate_local_artifacts()


        # ==================================================
        # 3. Load Metadata
        # ==================================================

        logger.info(

            f"Loading metadata from: "
            f"{LOCAL_METADATA_PATH}"

        )


        with open(

            LOCAL_METADATA_PATH,

            "r",

            encoding="utf-8",

        ) as file:

            self.metadata = json.load(
                file
            )


        # ==================================================
        # 4. Load Feature Schema
        # ==================================================

        logger.info(

            f"Loading feature schema from: "
            f"{LOCAL_FEATURE_SCHEMA_PATH}"

        )


        with open(

            LOCAL_FEATURE_SCHEMA_PATH,

            "r",

            encoding="utf-8",

        ) as file:

            self.feature_schema = json.load(
                file
            )


        # ==================================================
        # 5. Load Model
        # ==================================================

        logger.info(

            f"Loading model from: "
            f"{LOCAL_MODEL_PATH}"

        )


        self.model = joblib.load(

            LOCAL_MODEL_PATH

        )


        # ==================================================
        # 6. Load Preprocessor
        # ==================================================

        logger.info(

            f"Loading preprocessor from: "
            f"{LOCAL_PREPROCESSOR_PATH}"

        )


        self.preprocessor = joblib.load(

            LOCAL_PREPROCESSOR_PATH

        )


        # ==================================================
        # 7. Validate Preprocessor
        # ==================================================

        if not isinstance(

            self.preprocessor,

            dict,

        ):

            raise TypeError(

                "preprocessor.pkl must contain "
                "a dictionary."

            )


        required_preprocessor_keys = [

            "scaler",

            "feature_columns",

            "scaling_columns",

            "insurance_plan_mapping",

            "income_level_mapping",

            "categorical_columns",

            "drop_first",

            "risk_scores",

            "risk_min",

            "risk_max",

            "physical_activity_score",

            "stress_score",

        ]


        missing_preprocessor_keys = [

            key

            for key in required_preprocessor_keys

            if key not in self.preprocessor

        ]


        if missing_preprocessor_keys:

            raise ValueError(

                "Missing required keys in "
                "preprocessor.pkl: "

                f"{missing_preprocessor_keys}"

            )


        # ==================================================
        # 8. Extract Preprocessor Components
        # ==================================================

        self.scaler = (

            self.preprocessor[
                "scaler"
            ]

        )


        self.feature_columns = (

            self.preprocessor[
                "feature_columns"
            ]

        )


        self.scaling_columns = (

            self.preprocessor[
                "scaling_columns"
            ]

        )


        # ==================================================
        # 9. Validate Metadata Features
        # ==================================================

        metadata_features = (

            self.metadata.get(

                "feature_columns",

                [],

            )

        )


        if (

            metadata_features

            and metadata_features

            != self.feature_columns

        ):

            raise ValueError(

                "Feature mismatch detected between "
                "metadata.json and preprocessor.pkl."

            )


        # ==================================================
        # 10. Validate Feature Schema
        # ==================================================

        schema_features = (

            self.feature_schema.get(

                "feature_columns",

                [],

            )

        )


        if (

            schema_features

            and schema_features

            != self.feature_columns

        ):

            raise ValueError(

                "Feature mismatch detected between "
                "feature_schema.json and "
                "preprocessor.pkl."

            )


        # ==================================================
        # 11. Validate Scaler
        # ==================================================

        scaler_features = getattr(

            self.scaler,

            "n_features_in_",

            None,

        )


        if (

            scaler_features is not None

            and scaler_features

            != len(
                self.scaling_columns
            )

        ):

            raise ValueError(

                "Scaler feature mismatch. "

                f"Scaler expects "
                f"{scaler_features} features, "

                f"but scaling_columns contains "

                f"{len(self.scaling_columns)} "
                "features."

            )


        # ==================================================
        # 12. Validate Model
        # ==================================================

        if self.model is None:

            raise ValueError(

                "Model could not be loaded."

            )


        if not hasattr(

            self.model,

            "predict",

        ):

            raise TypeError(

                "Loaded model does not have "
                "a 'predict' method."

            )


        # ==================================================
        # 13. Extract Metrics
        # ==================================================

        metrics = (

            self.metadata.get(

                "metrics",

                {},

            )

        )


        # ==================================================
        # 14. Log Model Information
        # ==================================================

        logger.info("")

        logger.info("=" * 70)

        logger.info(
            "Current Loaded Model"
        )

        logger.info("=" * 70)


        logger.info(

            f"Source               : "
            f"{MODEL_SOURCE}"

        )


        logger.info(

            f"Configured Version   : "
            f"{MODEL_VERSION}"

        )


        logger.info(

            f"Artifact Version     : "
            f"{self.model_version}"

        )


        logger.info(

            f"Algorithm            : "
            f"{self.metadata.get('algorithm')}"

        )


        logger.info(

            f"Expected Features    : "
            f"{len(self.feature_columns)}"

        )


        logger.info(

            f"Scaling Features     : "
            f"{len(self.scaling_columns)}"

        )


        logger.info(

            f"R2 Score             : "
            f"{metrics.get('R2')}"

        )


        logger.info(

            f"MAE                  : "
            f"{metrics.get('MAE')}"

        )


        logger.info(

            f"RMSE                 : "
            f"{metrics.get('RMSE')}"

        )


        logger.info(

            f"Model Artifact       : "
            f"{LOCAL_MODEL_PATH}"

        )


        logger.info(

            f"Preprocessor Artifact: "
            f"{LOCAL_PREPROCESSOR_PATH}"

        )


        logger.info(

            f"Metadata Artifact    : "
            f"{LOCAL_METADATA_PATH}"

        )


        logger.info(

            f"Feature Schema       : "
            f"{LOCAL_FEATURE_SCHEMA_PATH}"

        )


        logger.info("=" * 70)

        logger.info(

            "Prediction Service Ready"

        )

        logger.info("=" * 70)


        # ==================================================
        # 15. Mark Service as Loaded
        # ==================================================

        self.is_loaded = True


    # ======================================================
    # Feature Preparation
    # ======================================================

    def prepare_features(

        self,

        input_data,

    ):

        if not self.is_loaded:

            self.load_model()


        # ==================================================
        # 1. Convert Input to DataFrame
        # ==================================================

        if isinstance(

            input_data,

            dict,

        ):

            df = pd.DataFrame(

                [input_data]

            )


        elif isinstance(

            input_data,

            pd.DataFrame,

        ):

            df = input_data.copy()


        else:

            raise TypeError(

                "input_data must be a dictionary "
                "or pandas DataFrame."

            )


        # ==================================================
        # 2. Standardize Column Names
        # ==================================================

        df.columns = (

            df.columns

            .str.strip()

            .str.lower()

            .str.replace(

                " ",

                "_",

                regex=False,

            )

        )


        # ==================================================
        # 3. Handle Missing Value Representations
        # ==================================================

        missing_values = [

            "",

            " ",

            "-",

            "none",

            "null",

            "nan",

        ]


        df = df.replace(

            missing_values,

            np.nan,

        )


        # ==================================================
        # 4. Required Raw Input Features
        # ==================================================

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


        missing_raw_features = [

            column

            for column in required_raw_features

            if column not in df.columns

        ]


        if missing_raw_features:

            raise ValueError(

                "Missing required input features: "

                f"{missing_raw_features}"

            )


        # ==================================================
        # 5. Convert Numeric Input Columns
        # ==================================================

        numeric_input_columns = [

            "age",

            "number_of_dependants",

            "income_lakhs",

        ]


        for column in numeric_input_columns:

            df[column] = pd.to_numeric(

                df[column],

                errors="coerce",

            )


        # ==================================================
        # 6. Validate Required Values
        # ==================================================

        missing_value_columns = (

            df[
                required_raw_features
            ]

            .isnull()

            .any()

        )


        missing_value_columns = (

            missing_value_columns[

                missing_value_columns

            ]

            .index

            .tolist()

        )


        if missing_value_columns:

            raise ValueError(

                "Missing or invalid values found "
                "in required input features: "

                f"{missing_value_columns}"

            )


        # ==================================================
        # 7. Business Validation
        # ==================================================

        if (

            (df["age"] < 0)

            | (df["age"] > 100)

        ).any():

            raise ValueError(

                "Age must be between 0 and 100."

            )


        if (

            df[
                "income_lakhs"
            ] < 0

        ).any():

            raise ValueError(

                "income_lakhs cannot be negative."

            )


        if (

            df[
                "number_of_dependants"
            ] < 0

        ).any():

            raise ValueError(

                "number_of_dependants cannot be negative."

            )


        # ==================================================
        # 8. Normalize Text Values
        # ==================================================

        text_columns = [

            "income_level",

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


        for column in text_columns:

            df[column] = (

                df[column]

                .astype(str)

                .str.strip()

            )


        # ==================================================
        # 9. Medical Risk Score
        # ==================================================

        risk_scores = (

            self.preprocessor[

                "risk_scores"

            ]

        )


        medical_split = (

            df[
                "medical_history"
            ]

            .astype(str)

            .str.lower()

            .str.split(

                " & ",

                expand=True,

            )

        )


        df[
            "disease1"
        ] = (

            medical_split[0]

            .fillna(

                "none"

            )

        )


        if 1 in medical_split.columns:

            df[
                "disease2"
            ] = (

                medical_split[1]

                .fillna(

                    "none"

                )

            )

        else:

            df[
                "disease2"
            ] = "none"


        df[
            "total_risk_score"
        ] = (

            df[
                "disease1"
            ]

            .map(

                risk_scores

            )

            .fillna(0)

            +

            df[
                "disease2"
            ]

            .map(

                risk_scores

            )

            .fillna(0)

        )


        # ==================================================
        # 10. Normalize Risk Using Training Parameters
        # ==================================================

        risk_min = float(

            self.preprocessor[

                "risk_min"

            ]

        )


        risk_max = float(

            self.preprocessor[

                "risk_max"

            ]

        )


        if risk_max == risk_min:

            df[
                "normalized_risk_score"
            ] = 0.0

        else:

            df[
                "normalized_risk_score"
            ] = (

                df[
                    "total_risk_score"
                ]

                - risk_min

            ) / (

                risk_max

                - risk_min

            )


        # ==================================================
        # 11. Lifestyle Risk Score
        # ==================================================

        physical_activity_score = (

            self.preprocessor[

                "physical_activity_score"

            ]

        )


        stress_score = (

            self.preprocessor[

                "stress_score"

            ]

        )


        df[
            "lifestyle_risk_score"
        ] = (

            df[
                "physical_activity"
            ]

            .map(

                physical_activity_score

            )

            .fillna(0)

            +

            df[
                "stress_level"
            ]

            .map(

                stress_score

            )

            .fillna(0)

        )


        # ==================================================
        # 12. Apply Saved Insurance Mapping
        # ==================================================

        insurance_plan_mapping = (

            self.preprocessor[

                "insurance_plan_mapping"

            ]

        )


        df[
            "insurance_plan"
        ] = (

            df[
                "insurance_plan"
            ]

            .map(

                insurance_plan_mapping

            )

        )


        if (

            df[
                "insurance_plan"
            ]

            .isnull()

            .any()

        ):

            raise ValueError(

                "Unknown insurance_plan value detected."

            )


        # ==================================================
        # 13. Apply Saved Income Mapping
        # ==================================================

        income_level_mapping = (

            self.preprocessor[

                "income_level_mapping"

            ]

        )


        df[
            "income_level"
        ] = (

            df[
                "income_level"
            ]

            .map(

                income_level_mapping

            )

        )


        if (

            df[
                "income_level"
            ]

            .isnull()

            .any()

        ):

            raise ValueError(

                "Unknown income_level value detected."

            )


        # ==================================================
        # 14. One-Hot Encoding
        # ==================================================

        categorical_columns = (

            self.preprocessor[

                "categorical_columns"

            ]

        )


        drop_first = (

            self.preprocessor[

                "drop_first"

            ]

        )


        missing_categorical = [

            column

            for column in categorical_columns

            if column not in df.columns

        ]


        if missing_categorical:

            raise ValueError(

                "Missing categorical input features: "

                f"{missing_categorical}"

            )


        df = pd.get_dummies(

            df,

            columns=categorical_columns,

            drop_first=drop_first,

            dtype=int,

        )


        # ==================================================
        # 15. Remove Training-Only Raw Columns
        # ==================================================

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

            inplace=True,

            errors="ignore",

        )


        # ==================================================
        # 16. Align With Training Feature Schema
        # ==================================================

        expected_features = list(

            self.feature_columns

        )


        # --------------------------------------------------
        # Detect Unexpected Features
        # --------------------------------------------------

        extra_features = [

            column

            for column in df.columns

            if column not in expected_features

        ]


        if extra_features:

            logger.warning(

                "Ignoring unexpected features generated "
                "during inference: "

                f"{extra_features}"

            )


        # --------------------------------------------------
        # Add Missing Features
        # --------------------------------------------------

        for column in expected_features:

            if column not in df.columns:

                df[column] = 0


        # --------------------------------------------------
        # Keep Only Expected Features
        # --------------------------------------------------

        df = df[

            expected_features

        ].copy()


        # ==================================================
        # 17. Convert Final Features to Numeric
        # ==================================================

        for column in expected_features:

            df[column] = pd.to_numeric(

                df[column],

                errors="coerce",

            )


        # ==================================================
        # 18. Validate Missing Values
        # ==================================================

        if df.isnull().any().any():

            missing_columns = (

                df.columns[

                    df.isnull().any()

                ]

                .tolist()

            )


            raise ValueError(

                "Missing or invalid values found "
                "after feature engineering: "

                f"{missing_columns}"

            )


        # ==================================================
        # 19. Apply Training-Fitted Scaler
        # ==================================================

        if self.scaling_columns:

            missing_scaling_columns = [

                column

                for column in self.scaling_columns

                if column not in df.columns

            ]


            if missing_scaling_columns:

                raise ValueError(

                    "Missing scaling columns: "

                    f"{missing_scaling_columns}"

                )


            df[

                self.scaling_columns

            ] = self.scaler.transform(

                df[

                    self.scaling_columns

                ]

            )


        # ==================================================
        # 20. Final Feature Order Validation
        # ==================================================

        if (

            list(df.columns)

            != list(self.feature_columns)

        ):

            raise ValueError(

                "Final feature order does not match "
                "training feature order."

            )


        # ==================================================
        # 21. Final Feature Count Validation
        # ==================================================

        if (

            df.shape[1]

            != len(

                self.feature_columns

            )

        ):

            raise ValueError(

                "Final feature count mismatch. "

                f"Expected: "
                f"{len(self.feature_columns)}, "

                f"Received: "
                f"{df.shape[1]}"

            )


        logger.info(

            "Feature preparation completed successfully."

        )


        logger.info(

            f"Final feature shape: "
            f"{df.shape}"

        )


        logger.info(

            f"Final features: "
            f"{list(df.columns)}"

        )


        return df


    # ======================================================
    # Predict
    # ======================================================

    def predict(

        self,

        input_data,

    ):

        if not self.is_loaded:

            self.load_model()


        # --------------------------------------------------
        # Prepare Features
        # --------------------------------------------------

        features = (

            self.prepare_features(

                input_data

            )

        )


        logger.info(

            f"Prediction input shape: "
            f"{features.shape}"

        )


        # --------------------------------------------------
        # Generate Prediction
        # --------------------------------------------------

        prediction = (

            self.model.predict(

                features

            )

        )


        # --------------------------------------------------
        # Convert Prediction to NumPy Array
        # --------------------------------------------------

        prediction = np.asarray(

            prediction

        ).reshape(-1)


        # --------------------------------------------------
        # Convert Results to Python Float
        # --------------------------------------------------

        results = [

            float(value)

            for value in prediction

        ]


        # --------------------------------------------------
        # Single Prediction
        # --------------------------------------------------

        if len(results) == 1:

            return results[0]


        # --------------------------------------------------
        # Multiple Predictions
        # --------------------------------------------------

        return results


# ==========================================================
# Singleton Prediction Service
# ==========================================================

predictor = PredictionService()


# ==========================================================
# Public Prediction Function
# ==========================================================

def predict_premium(

    input_data,

):

    return predictor.predict(

        input_data

    )