from pathlib import Path
from datetime import datetime
import json

import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import xgboost as xgb

from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    root_mean_squared_error,
)


class ModelTrainer:

    def train(
        self,
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor,
    ):

        print("=" * 70)
        print("MODEL TRAINING")
        print("=" * 70)

        # ==========================================================
        # MLflow Configuration
        # ==========================================================

        mlflow.set_tracking_uri("http://127.0.0.1:5000")

        mlflow.set_experiment("Healthcare_Premium_Prediction")

        # ==========================================================
        # 1. Linear Regression
        # ==========================================================

        print("\nTraining Linear Regression...")

        lr_model = LinearRegression()

        lr_model.fit(X_train, y_train)

        lr_predictions = lr_model.predict(X_test)

        lr_r2 = r2_score(y_test, lr_predictions)

        lr_mae = mean_absolute_error(y_test, lr_predictions)

        lr_rmse = root_mean_squared_error(y_test, lr_predictions)

        print(f"Linear Regression R2   : {lr_r2:.4f}")

        print(f"Linear Regression MAE  : {lr_mae:.4f}")

        print(f"Linear Regression RMSE : {lr_rmse:.4f}")

        # ==========================================================
        # Log Linear Regression to MLflow
        # ==========================================================

        with mlflow.start_run(run_name="Linear_Regression"):

            mlflow.log_param("algorithm", "Linear Regression")

            mlflow.log_metric("R2", lr_r2)

            mlflow.log_metric("MAE", lr_mae)

            mlflow.log_metric("RMSE", lr_rmse)

            mlflow.sklearn.log_model(lr_model, name="linear_regression_model")

        # ==========================================================
        # 2. XGBoost
        # ==========================================================

        print("\nTraining XGBoost...")

        xgb_model = xgb.XGBRegressor(
            objective="reg:squarederror",
            n_estimators=100,
            learning_rate=0.05,
            max_depth=6,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=10,
            n_jobs=-1,
        )

        xgb_model.fit(X_train, y_train)

        xgb_predictions = xgb_model.predict(X_test)

        xgb_r2 = r2_score(y_test, xgb_predictions)

        xgb_mae = mean_absolute_error(y_test, xgb_predictions)

        xgb_rmse = root_mean_squared_error(y_test, xgb_predictions)

        print(f"XGBoost R2   : {xgb_r2:.4f}")

        print(f"XGBoost MAE  : {xgb_mae:.4f}")

        print(f"XGBoost RMSE : {xgb_rmse:.4f}")

        # ==========================================================
        # Log XGBoost to MLflow
        # ==========================================================

        with mlflow.start_run(run_name="XGBoost"):

            mlflow.log_param("algorithm", "XGBoost")

            mlflow.log_param("n_estimators", 100)

            mlflow.log_param("learning_rate", 0.05)

            mlflow.log_param("max_depth", 6)

            mlflow.log_metric("R2", xgb_r2)

            mlflow.log_metric("MAE", xgb_mae)

            mlflow.log_metric("RMSE", xgb_rmse)

            mlflow.xgboost.log_model(xgb_model, name="xgboost_model")

        # ==========================================================
        # Model Comparison
        # ==========================================================

        if xgb_r2 >= lr_r2:

            best_model = xgb_model

            best_algorithm = "XGBoost"

            best_r2 = xgb_r2

            best_mae = xgb_mae

            best_rmse = xgb_rmse

        else:

            best_model = lr_model

            best_algorithm = "Linear Regression"

            best_r2 = lr_r2

            best_mae = lr_mae

            best_rmse = lr_rmse

        # ==========================================================
        # Model Version
        # ==========================================================

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        project_name = "healthcare_premium_prediction"

        algorithm_name = best_algorithm.lower().replace(" ", "_")

        model_version = f"{project_name}_" f"{algorithm_name}_" f"v1_" f"{timestamp}"

        # ==========================================================
        # Artifact Paths
        # ==========================================================

        project_root = Path(__file__).resolve().parents[1]

        artifact_root = project_root / "artifacts"

        versioned_dir = artifact_root / model_version

        latest_dir = artifact_root / "latest"

        versioned_dir.mkdir(parents=True, exist_ok=True)

        latest_dir.mkdir(parents=True, exist_ok=True)

        # ==========================================================
        # Save Best Model
        # ==========================================================

        joblib.dump(best_model, versioned_dir / "model.pkl")

        joblib.dump(best_model, latest_dir / "model.pkl")

        # ==========================================================
        # Copy Preprocessor
        # ==========================================================

        joblib.dump(preprocessor, versioned_dir / "preprocessor.pkl")

        joblib.dump(preprocessor, latest_dir / "preprocessor.pkl")

        # ==========================================================
        # Metadata
        # ==========================================================

        metadata = {
            "project_name": project_name,
            "model_name": "Healthcare Premium Prediction",
            "model_version": model_version,
            "algorithm": best_algorithm,
            "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "dataset": "premiums_with_life_style.xlsx",
            "training_samples": len(X_train),
            "testing_samples": len(X_test),
            "total_features": len(preprocessor["feature_columns"]),
            "feature_columns": preprocessor["feature_columns"],
            "scaling_columns": preprocessor["scaling_columns"],
            "metrics": {
                "R2": round(best_r2, 4),
                "MAE": round(best_mae, 4),
                "RMSE": round(best_rmse, 4),
            },
            "model_comparison": {
                "Linear Regression": {
                    "R2": round(lr_r2, 4),
                    "MAE": round(lr_mae, 4),
                    "RMSE": round(lr_rmse, 4),
                },
                "XGBoost": {
                    "R2": round(xgb_r2, 4),
                    "MAE": round(xgb_mae, 4),
                    "RMSE": round(xgb_rmse, 4),
                },
            },
        }

        # ==========================================================
        # Save Metadata
        # ==========================================================

        with open(versioned_dir / "metadata.json", "w", encoding="utf-8") as f:

            json.dump(metadata, f, indent=4)

        with open(latest_dir / "metadata.json", "w", encoding="utf-8") as f:

            json.dump(metadata, f, indent=4)

        # ==========================================================
        # Final Output
        # ==========================================================

        print("\n" + "=" * 70)

        print("MODEL TRAINING COMPLETED")

        print("=" * 70)

        print(f"Best Model      : " f"{best_algorithm}")

        print(f"Model Version   : " f"{model_version}")

        print(f"R2 Score        : " f"{best_r2:.4f}")

        print(f"MAE             : " f"{best_mae:.4f}")

        print(f"RMSE            : " f"{best_rmse:.4f}")

        print(f"Latest Artifact : " f"{latest_dir}")

        print("=" * 70)

        return {
            "model_version": model_version,
            "algorithm": best_algorithm,
            "r2_score": round(best_r2, 4),
            "mae": round(best_mae, 4),
            "rmse": round(best_rmse, 4),
        }
