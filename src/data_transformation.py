from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


class DataTransformation:

    def __init__(self):

        self.project_root = Path(__file__).resolve().parents[1]

        self.artifact_dir = self.project_root / "artifacts" / "latest"

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

        self.artifact_dir.mkdir(parents=True, exist_ok=True)

        # --------------------------------------------------------
        # Data Cleaning
        # --------------------------------------------------------

        print("Original dataset shape:", df.shape)

        # Remove duplicate rows
        df.drop_duplicates(inplace=True)

        # Remove rows with missing values
        # Training data must not contain missing values
        df.dropna(inplace=True)

        print("After cleaning shape:", df.shape)

        # --------------------------------------------------------
        # Business validation
        # --------------------------------------------------------

        # Age should be between 0 and 100
        df = df[(df["age"] >= 0) & (df["age"] <= 100)]

        # Income cannot be negative
        df = df[df["income_lakhs"] >= 0]

        # Dependants cannot be negative
        df = df[df["number_of_dependants"] >= 0]

        # --------------------------------------------------------
        # Feature Engineering
        # --------------------------------------------------------

        # ========================================================
        # Medical Risk Score
        # ========================================================

        risk_scores = {
            "diabetes": 6,
            "heart disease": 8,
            "high blood pressure": 6,
            "thyroid": 5,
            "no disease": 0,
            "none": 0,
        }

        # Split medical history
        medical_split = (
            df["medical_history"].astype(str).str.lower().str.split(" & ", expand=True)
        )

        df["disease1"] = medical_split[0].fillna("none")

        if 1 in medical_split.columns:

            df["disease2"] = medical_split[1].fillna("none")

        else:

            df["disease2"] = "none"

        # Calculate total risk
        df["total_risk_score"] = df["disease1"].map(risk_scores).fillna(0) + df[
            "disease2"
        ].map(risk_scores).fillna(0)

        # Normalize risk score
        risk_min = df["total_risk_score"].min()

        risk_max = df["total_risk_score"].max()

        if risk_max == risk_min:

            df["normalized_risk_score"] = 0.0

        else:

            df["normalized_risk_score"] = (df["total_risk_score"] - risk_min) / (
                risk_max - risk_min
            )

        # ========================================================
        # Lifestyle Risk Score
        # ========================================================

        physical_activity_score = {"High": 0, "Medium": 1, "Low": 4}

        stress_score = {"High": 4, "Medium": 1, "Low": 0}

        df["lifestyle_risk_score"] = df["physical_activity"].map(
            physical_activity_score
        ).fillna(0) + df["stress_level"].map(stress_score).fillna(0)

        # ========================================================
        # Encode Insurance Plan
        # ========================================================

        insurance_plan_mapping = {"Bronze": 1, "Silver": 2, "Gold": 3}

        df["insurance_plan"] = df["insurance_plan"].map(insurance_plan_mapping)

        # ========================================================
        # Encode Income Level
        # ========================================================

        income_level_mapping = {"<10L": 1, "10L - 25L": 2, "25L - 40L": 3, "> 40L": 4}

        df["income_level"] = df["income_level"].map(income_level_mapping)

        # --------------------------------------------------------
        # Validate categorical encoding
        # --------------------------------------------------------

        if df["insurance_plan"].isnull().any():

            raise ValueError("Unknown insurance_plan value detected")

        if df["income_level"].isnull().any():

            raise ValueError("Unknown income_level value detected")

        # ========================================================
        # One-Hot Encoding
        # ========================================================

        categorical_columns = [
            "gender",
            "region",
            "marital_status",
            "bmi_category",
            "smoking_status",
            "employment_status",
        ]

        df = pd.get_dummies(df, columns=categorical_columns, drop_first=True, dtype=int)

        # ========================================================
        # Remove unused columns
        # ========================================================

        columns_to_drop = [
            "medical_history",
            "disease1",
            "disease2",
            "total_risk_score",
            "physical_activity",
            "stress_level",
        ]

        df.drop(columns=columns_to_drop, inplace=True, errors="ignore")

        # ========================================================
        # Separate Features and Target
        # ========================================================

        X = df.drop(columns=["annual_premium_amount"])

        y = df["annual_premium_amount"]

        # ========================================================
        # Save Final Feature Schema
        # ========================================================

        feature_columns = list(X.columns)

        # ========================================================
        # Train/Test Split
        # ========================================================

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.30, random_state=10
        )

        # ========================================================
        # Scaling
        # IMPORTANT:
        # Fit scaler ONLY on training data
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

        scaler = MinMaxScaler()

        # Fit ONLY on training data

        X_train = X_train.copy()

        X_test = X_test.copy()

        X_train[scaling_columns] = scaler.fit_transform(X_train[scaling_columns])

        # Transform test data

        X_test[scaling_columns] = scaler.transform(X_test[scaling_columns])

        # ========================================================
        # Save Preprocessor
        # ========================================================

        preprocessor = {
            "preprocessor_version": "preprocessor_v3",
            "scaler": scaler,
            "feature_columns": feature_columns,
            "scaling_columns": scaling_columns,
            "insurance_plan_mapping": insurance_plan_mapping,
            "income_level_mapping": income_level_mapping,
            "categorical_columns": categorical_columns,
        }

        joblib.dump(preprocessor, self.artifact_dir / "preprocessor.pkl")

        # ========================================================
        # Save Feature Schema
        # ========================================================

        feature_schema = {
            "feature_count": len(feature_columns),
            "feature_columns": feature_columns,
            "scaling_columns": scaling_columns,
            "categorical_columns": categorical_columns,
        }

        import json

        with open(
            self.artifact_dir / "feature_schema.json", "w", encoding="utf-8"
        ) as f:

            json.dump(feature_schema, f, indent=4)

        # ========================================================
        # Save Train/Test Data
        # ========================================================

        train_df = X_train.copy()

        train_df["annual_premium_amount"] = y_train.values

        train_df.to_csv(self.artifact_dir / "train.csv", index=False)

        test_df = X_test.copy()

        test_df["annual_premium_amount"] = y_test.values

        test_df.to_csv(self.artifact_dir / "test.csv", index=False)

        # ========================================================
        # Print Results
        # ========================================================

        print("Transformation completed successfully")

        print("Training samples:", len(X_train))

        print("Testing samples:", len(X_test))

        print("Final feature count:", len(feature_columns))

        print("Feature columns:")

        for feature in feature_columns:

            print("  -", feature)

        print("Scaling columns:")

        for column in scaling_columns:

            print("  -", column)

        print("Preprocessor saved:", self.artifact_dir / "preprocessor.pkl")

        print("=" * 70)

        return (X_train, X_test, y_train, y_test, preprocessor)
