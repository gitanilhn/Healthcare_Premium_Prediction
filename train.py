from src.data_ingestion import DataIngestion
from src.data_validation import DataValidation
from src.data_transformation import DataTransformation
from src.model_trainer import ModelTrainer


def main():

    print("\n")
    print("=" * 70)
    print("HEALTHCARE PREMIUM PREDICTION - TRAINING PIPELINE")
    print("=" * 70)

    # ==========================================================
    # STEP 1 - Data Ingestion
    # ==========================================================

    ingestion = DataIngestion()

    df = ingestion.load_data()

    # ==========================================================
    # STEP 2 - Data Validation
    # ==========================================================

    validation = DataValidation()

    validation.validate(
        df
    )

    # ==========================================================
    # STEP 3 - Data Transformation
    # ==========================================================

    transformation = DataTransformation()

    (
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor,
    ) = transformation.transform(
        df
    )

    # ==========================================================
    # STEP 4 - Model Training
    # ==========================================================

    trainer = ModelTrainer()

    results = trainer.train(

        X_train,

        X_test,

        y_train,

        y_test,

        preprocessor,
    )

    # ==========================================================
    # Final Summary
    # ==========================================================

    print("\n")
    print("=" * 70)
    print("TRAINING PIPELINE COMPLETED")
    print("=" * 70)

    print(
        f"Model Version : "
        f"{results['model_version']}"
    )

    print(
        f"Algorithm     : "
        f"{results['algorithm']}"
    )

    print(
        f"R2 Score      : "
        f"{results['r2_score']}"
    )

    print(
        f"MAE           : "
        f"{results['mae']}"
    )

    print(
        f"RMSE          : "
        f"{results['rmse']}"
    )

    print("=" * 70)


if __name__ == "__main__":

    main()