from pathlib import Path
import pandas as pd


class DataIngestion:

    def __init__(self):

        # Project root
        self.project_root = Path(__file__).resolve().parents[1]

        # Dataset path
        self.data_path = self.project_root / "data" / "premiums_with_life_style.xlsx"

    def load_data(self):

        print("=" * 70)
        print("DATA INGESTION")
        print("=" * 70)

        # ---------------------------------------------------------
        # Check dataset exists
        # ---------------------------------------------------------

        if not self.data_path.exists():

            raise FileNotFoundError(f"Dataset not found: {self.data_path}")

        print(f"Loading dataset: {self.data_path}")

        # ---------------------------------------------------------
        # Read Excel
        # ---------------------------------------------------------

        df = pd.read_excel(self.data_path)

        # ---------------------------------------------------------
        # Normalize column names
        # ---------------------------------------------------------

        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

        # ---------------------------------------------------------
        # Display information
        # ---------------------------------------------------------

        print(f"Rows loaded    : {len(df)}")

        print(f"Columns loaded : {len(df.columns)}")

        print(f"Dataset shape  : {df.shape}")

        print("Normalized columns:")

        for column in df.columns:

            print(f"  - {column}")

        print("=" * 70)

        return df
