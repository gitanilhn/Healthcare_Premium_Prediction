from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


class DataTransformation:

    def __init__(self):

        self.project_root = Path(__file__).resolve().parents[1]

        self.artifact_dir = (
            self.project_root
            / "artifacts"
            / "latest"
        )

    # ============================================================
    # MAIN TRANSFORMATION
    # ============================================================

    def transform(self, df):

        print("=" * 70)
        print("DATA TRANSFORMATION")
        print("=" * 70)

        df = df.copy()

        # --------------------------------------------------------
        # Create artifact directory
        # --------------------------------------------------------

        self.artifact_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # --------------------------------------------------------
        # Standardize column names
        # --------------------------------------------------------

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

        print(
            "Original dataset shape:",
            df.shape,
        )

        # ========================================================
        # 1. DATA CLEANING
        # ========================================================

        # Remove duplicate rows
        df.drop_duplicates(
            inplace=True
        )

        # Remove rows with missing values
        # Training data must not contain missing values
        df.dropna(
            inplace=True
        )

        print(
            "After cleaning shape:",
            df.shape,
        )

        # ========================================================
        # 2. BUSINESS VALIDATION
        # ========================================================

        # Age should be between 0 and 100
        df = df[
            (df["age"] >= 0)
            & (df["age"] <= 100)
        ]

        # Income cannot be negative
        df = df[
            df["income_lakhs"] >= 0
        ]

        # Dependants cannot be negative
        df = df[
            df["number_of_dependants"] >= 0
        ]

        # ========================================================
        # 3. FEATURE ENGINEERING
        # ========================================================

        # --------------------------------------------------------
        # Medical Risk Score
        # --------------------------------------------------------

        risk_scores = {
            "diabetes": 6,
            "heart disease": 8,
            "high blood pressure": 6,
            "thyroid": 5,
            "no disease": 0,
            "none": 0,
        }

        # --------------------------------------------------------
        # Split medical history
        # --------------------------------------------------------

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
        ] = medical_split[
            0
        ].fillna(
            "none"
        )

        if 1 in medical_split.columns:

            df[
                "disease2"
            ] = medical_split[
                1
            ].fillna(
                "none"
            )

        else:

            df[
                "disease2"
            ] = "none"

        # --------------------------------------------------------
        # Calculate total risk score
        # --------------------------------------------------------

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

        # --------------------------------------------------------
        # Learn risk normalization parameters
        #
        # IMPORTANT:
        # These values are learned during training and saved.
        # Prediction MUST reuse these exact values.
        # --------------------------------------------------------

        risk_min = float(
            df[
                "total_risk_score"
            ].min()
        )

        risk_max = float(
            df[
                "total_risk_score"
            ].max()
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

        # ========================================================
        # Lifestyle Risk Score
        # ========================================================

        physical_activity_score = {
            "High": 0,
            "Medium": 1,
            "Low": 4,
        }

        stress_score = {
            "High": 4,
            "Medium": 1,
            "Low": 0,
        }

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

        # ========================================================
        # 4. ENCODE ORDINAL CATEGORICAL VARIABLES
        # ========================================================

        insurance_plan_mapping = {
            "Bronze": 1,
            "Silver": 2,
            "Gold": 3,
        }

        income_level_mapping = {
            "<10L": 1,
            "10L - 25L": 2,
            "25L - 40L": 3,
            "> 40L": 4,
        }

        df[
            "insurance_plan"
        ] = df[
            "insurance_plan"
        ].map(
            insurance_plan_mapping
        )

        df[
            "income_level"
        ] = df[
            "income_level"
        ].map(
            income_level_mapping
        )

        # --------------------------------------------------------
        # Validate categorical mappings
        # --------------------------------------------------------

        if df[
            "insurance_plan"
        ].isnull().any():

            raise ValueError(
                "Unknown insurance_plan value detected."
            )

        if df[
            "income_level"
        ].isnull().any():

            raise ValueError(
                "Unknown income_level value detected."
            )

        # ========================================================
        # 5. ONE-HOT ENCODING
        # ========================================================

        categorical_columns = [

            "gender",

            "region",

            "marital_status",

            "bmi_category",

            "smoking_status",

            "employment_status",

        ]

        # IMPORTANT:
        # drop_first=True must be identical between
        # training and inference.
        #
        # The exact resulting feature_columns are saved
        # and reused during prediction.

        df = pd.get_dummies(
            df,
            columns=categorical_columns,
            drop_first=True,
            dtype=int,
        )

        # ========================================================
        # 6. REMOVE UNUSED RAW COLUMNS
        # ========================================================

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

        # ========================================================
        # 7. SEPARATE FEATURES AND TARGET
        # ========================================================

        X = df.drop(
            columns=[
                "annual_premium_amount"
            ]
        )

        y = df[
            "annual_premium_amount"
        ]

        # ========================================================
        # 8. FINAL FEATURE SCHEMA
        # ========================================================

        feature_columns = list(
            X.columns
        )

        # ========================================================
        # 9. TRAIN / TEST SPLIT
        # ========================================================

        X_train, X_test, y_train, y_test = (
            train_test_split(
                X,
                y,
                test_size=0.30,
                random_state=10,
            )
        )

        # ========================================================
        # 10. SCALING
        # ========================================================

        scaling_columns = [

            "age",

            "number_of_dependants",

            "income_level",

            "income_lakhs",

            "insurance_plan",

            "lifestyle_risk_score",

            "normalized_risk_score",

        ]

        # Validate scaling columns
        missing_scaling_columns = [

            column

            for column in scaling_columns

            if column not in X_train.columns

        ]

        if missing_scaling_columns:

            raise ValueError(
                "Scaling columns missing from "
                f"training data: "
                f"{missing_scaling_columns}"
            )

        scaler = MinMaxScaler()

        X_train = X_train.copy()

        X_test = X_test.copy()

        # Fit ONLY on training data
        X_train[
            scaling_columns
        ] = scaler.fit_transform(
            X_train[
                scaling_columns
            ]
        )

        # Transform test data using
        # the training-fitted scaler
        X_test[
            scaling_columns
        ] = scaler.transform(
            X_test[
                scaling_columns
            ]
        )

        # ========================================================
        # 11. SAVE PREPROCESSOR
        # ========================================================

        preprocessor = {

            "preprocessor_version":
                "preprocessor_v4",

            "scaler":
                scaler,

            "feature_columns":
                feature_columns,

            "scaling_columns":
                scaling_columns,

            "insurance_plan_mapping":
                insurance_plan_mapping,

            "income_level_mapping":
                income_level_mapping,

            "categorical_columns":
                categorical_columns,

            "drop_first":
                True,

            "risk_scores":
                risk_scores,

            "risk_min":
                risk_min,

            "risk_max":
                risk_max,

            "physical_activity_score":
                physical_activity_score,

            "stress_score":
                stress_score,

        }

        joblib.dump(
            preprocessor,
            self.artifact_dir
            / "preprocessor.pkl",
        )

        # ========================================================
        # 12. SAVE FEATURE SCHEMA
        # ========================================================

        feature_schema = {

            "schema_version":
                "feature_schema_v4",

            "feature_count":
                len(feature_columns),

            "feature_columns":
                feature_columns,

            "scaling_columns":
                scaling_columns,

            "categorical_columns":
                categorical_columns,

            "drop_first":
                True,

            "derived_features": [

                "normalized_risk_score",

                "lifestyle_risk_score",

            ],

        }

        with open(

            self.artifact_dir
            / "feature_schema.json",

            "w",

            encoding="utf-8",

        ) as f:

            json.dump(

                feature_schema,

                f,

                indent=4,

            )

        # ========================================================
        # 15. PRINT RESULTS
        # ========================================================

        print(
            "Transformation completed successfully."
        )

        print(
            "Training samples:",
            len(X_train),
        )

        print(
            "Testing samples:",
            len(X_test),
        )

        print(
            "Final feature count:",
            len(feature_columns),
        )

        print(
            "Risk minimum:",
            risk_min,
        )

        print(
            "Risk maximum:",
            risk_max,
        )

        print(
            "Feature columns:"
        )

        for feature in feature_columns:

            print(
                "  -",
                feature,
            )

        print(
            "Scaling columns:"
        )

        for column in scaling_columns:

            print(
                "  -",
                column,
            )

        print(
            "Preprocessor saved:",
            self.artifact_dir
            / "preprocessor.pkl",
        )

        print(
            "Feature schema saved:",
            self.artifact_dir
            / "feature_schema.json",
        )

        print("=" * 70)

        return (

            X_train,

            X_test,

            y_train,

            y_test,

            preprocessor,

        )