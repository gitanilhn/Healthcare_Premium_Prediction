import pandas as pd


class DataValidation:

    REQUIRED_COLUMNS = [
        "age",
        "number_of_dependants",
        "income_lakhs",
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
        "annual_premium_amount",
    ]

    def validate(self, df: pd.DataFrame) -> bool:

        print("=" * 70)
        print("DATA VALIDATION")
        print("=" * 70)

        # --------------------------------------------------------
        # Validate DataFrame
        # --------------------------------------------------------

        if df is None:

            raise ValueError("Input DataFrame is None")

        if df.empty:

            raise ValueError("Input DataFrame is empty")

        # --------------------------------------------------------
        # Check Required Columns
        # --------------------------------------------------------

        missing_columns = [
            column for column in self.REQUIRED_COLUMNS if column not in df.columns
        ]

        if missing_columns:

            raise ValueError("Missing required columns: " f"{missing_columns}")

        # --------------------------------------------------------
        # Check Duplicate Rows
        # --------------------------------------------------------

        duplicate_count = df.duplicated().sum()

        print(f"Duplicate rows : {duplicate_count}")

        # --------------------------------------------------------
        # Check Missing Values
        # --------------------------------------------------------

        missing_values = df.isnull().sum().sum()

        print(f"Missing values  : {missing_values}")

        # --------------------------------------------------------
        # Validate Age
        # --------------------------------------------------------

        if df["age"].dropna().lt(0).any():

            raise ValueError("Age contains negative values")

        # --------------------------------------------------------
        # Validate Income
        # --------------------------------------------------------

        if df["income_lakhs"].dropna().lt(0).any():

            raise ValueError("income_lakhs contains negative values")

        # --------------------------------------------------------
        # Validate Target
        # --------------------------------------------------------

        if df["annual_premium_amount"].dropna().lt(0).any():

            raise ValueError("annual_premium_amount contains " "negative values")

        # --------------------------------------------------------
        # Validation Successful
        # --------------------------------------------------------

        print("Required columns : PASS")

        print("Data validation  : PASS")

        print("=" * 70)

        return True
