from pathlib import Path
from datetime import datetime
import json
import shutil

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

    # ==========================================================
    # INITIALIZATION
    # ==========================================================

    def __init__(self):

        # ------------------------------------------------------
        # Project directories
        # ------------------------------------------------------

        self.project_root = Path(__file__).resolve().parents[1]

        self.artifact_root = (
            self.project_root
            / "artifacts"
        )

        self.latest_dir = (
            self.artifact_root
            / "latest"
        )

        # ------------------------------------------------------
        # MLflow configuration
        # ------------------------------------------------------

        self.mlflow_tracking_uri = (
            "http://127.0.0.1:5000"
        )

        self.mlflow_experiment_name = (
            "Healthcare_Premium_Prediction"
        )

        self.mlflow_registered_model_name = (
            "Healthcare_Premium_Prediction_Model"
        )

    # ==========================================================
    # MAIN TRAINING METHOD
    # ==========================================================

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

        # ======================================================
        # 1. Validate Inputs
        # ======================================================

        self._validate_inputs(
            X_train,
            X_test,
            y_train,
            y_test,
            preprocessor,
        )

        # ======================================================
        # 2. Configure MLflow
        # ======================================================

        print(
            "\nConfiguring MLflow..."
        )

        mlflow.set_tracking_uri(
            self.mlflow_tracking_uri
        )

        mlflow.set_experiment(
            self.mlflow_experiment_name
        )

        print(
            f"MLflow Tracking URI : "
            f"{self.mlflow_tracking_uri}"
        )

        print(
            f"MLflow Experiment   : "
            f"{self.mlflow_experiment_name}"
        )

        # ======================================================
        # 3. Train Linear Regression
        # ======================================================

        lr_results = (
            self._train_linear_regression(
                X_train,
                X_test,
                y_train,
                y_test,
            )
        )

        # ======================================================
        # 4. Train XGBoost
        # ======================================================

        xgb_results = (
            self._train_xgboost(
                X_train,
                X_test,
                y_train,
                y_test,
            )
        )

        # ======================================================
        # 5. Compare Models
        # ======================================================

        best_result = (
            self._select_best_model(
                lr_results,
                xgb_results,
            )
        )

        best_model = (
            best_result["model"]
        )

        best_algorithm = (
            best_result["algorithm"]
        )

        best_r2 = (
            best_result["r2"]
        )

        best_mae = (
            best_result["mae"]
        )

        best_rmse = (
            best_result["rmse"]
        )

        print("\n" + "=" * 70)

        print(
            "BEST MODEL SELECTION"
        )

        print("=" * 70)

        print(
            f"Best Model : "
            f"{best_algorithm}"
        )

        print(
            f"R2         : "
            f"{best_r2:.4f}"
        )

        print(
            f"MAE        : "
            f"{best_mae:.4f}"
        )

        print(
            f"RMSE       : "
            f"{best_rmse:.4f}"
        )

        # ======================================================
        # 6. Generate Artifact Version
        # ======================================================

        timestamp = (
            datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )
        )

        project_name = (
            "healthcare_premium_prediction"
        )

        algorithm_name = (
            best_algorithm
            .lower()
            .replace(" ", "_")
        )

        model_version = (
            f"{project_name}_"
            f"{algorithm_name}_"
            f"{timestamp}"
        )

        # ======================================================
        # 7. Create Versioned Artifact Directory
        # ======================================================

        versioned_dir = (
            self.artifact_root
            / model_version
        )

        versioned_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.latest_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # ======================================================
        # 8. Register Best Model in MLflow
        # ======================================================

        mlflow_result = (
            self._register_best_model(
                best_model=best_model,
                best_algorithm=best_algorithm,
                best_r2=best_r2,
                best_mae=best_mae,
                best_rmse=best_rmse,
                X_train=X_train,
                X_test=X_test,
                preprocessor=preprocessor,
            )
        )

        # ======================================================
        # 9. Save Best Model
        # ======================================================

        versioned_model_path = (
            versioned_dir
            / "model.pkl"
        )

        latest_model_path = (
            self.latest_dir
            / "model.pkl"
        )

        joblib.dump(
            best_model,
            versioned_model_path,
        )

        joblib.dump(
            best_model,
            latest_model_path,
        )

        # ======================================================
        # 10. Save Preprocessor
        # ======================================================

        versioned_preprocessor_path = (
            versioned_dir
            / "preprocessor.pkl"
        )

        latest_preprocessor_path = (
            self.latest_dir
            / "preprocessor.pkl"
        )

        joblib.dump(
            preprocessor,
            versioned_preprocessor_path,
        )

        joblib.dump(
            preprocessor,
            latest_preprocessor_path,
        )

        # ======================================================
        # 11. Create Metadata
        # ======================================================

        metadata = {

            "project_name":
                project_name,

            "model_name":
                "Healthcare Premium Prediction",

            "model_version":
                model_version,

            "algorithm":
                best_algorithm,

            "training_date":
                datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),

            "dataset":
                "premiums_with_life_style.xlsx",

            "training_samples":
                len(X_train),

            "testing_samples":
                len(X_test),

            "total_features":
                len(
                    preprocessor[
                        "feature_columns"
                    ]
                ),

            "feature_columns":
                preprocessor[
                    "feature_columns"
                ],

            "scaling_columns":
                preprocessor[
                    "scaling_columns"
                ],

            "metrics": {

                "R2":
                    round(
                        best_r2,
                        4,
                    ),

                "MAE":
                    round(
                        best_mae,
                        4,
                    ),

                "RMSE":
                    round(
                        best_rmse,
                        4,
                    ),
            },

            "model_comparison": {

                "Linear Regression": {

                    "R2":
                        round(
                            lr_results["r2"],
                            4,
                        ),

                    "MAE":
                        round(
                            lr_results["mae"],
                            4,
                        ),

                    "RMSE":
                        round(
                            lr_results["rmse"],
                            4,
                        ),
                },

                "XGBoost": {

                    "R2":
                        round(
                            xgb_results["r2"],
                            4,
                        ),

                    "MAE":
                        round(
                            xgb_results["mae"],
                            4,
                        ),

                    "RMSE":
                        round(
                            xgb_results["rmse"],
                            4,
                        ),
                },
            },

            # --------------------------------------------------
            # MLflow information
            # --------------------------------------------------

            "mlflow": {

                "tracking_uri":
                    self.mlflow_tracking_uri,

                "experiment_name":
                    self.mlflow_experiment_name,

                "registered_model_name":
                    self.mlflow_registered_model_name,

                "run_id":
                    mlflow_result[
                        "run_id"
                    ],

                "model_version":
                    mlflow_result[
                        "model_version"
                    ],

                "registration_status":
                    mlflow_result[
                        "registration_status"
                    ],
            },
        }

        # ======================================================
        # 12. Save Metadata
        # ======================================================

        versioned_metadata_path = (
            versioned_dir
            / "metadata.json"
        )

        latest_metadata_path = (
            self.latest_dir
            / "metadata.json"
        )

        with open(
            versioned_metadata_path,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                metadata,
                f,
                indent=4,
            )

        with open(
            latest_metadata_path,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                metadata,
                f,
                indent=4,
            )

        # ======================================================
        # 13. Save Feature Schema
        # ======================================================

        feature_schema = {

            "feature_count":
                len(
                    preprocessor[
                        "feature_columns"
                    ]
                ),

            "feature_columns":
                preprocessor[
                    "feature_columns"
                ],

            "scaling_columns":
                preprocessor[
                    "scaling_columns"
                ],

            "categorical_columns":
                preprocessor.get(
                    "categorical_columns",
                    [],
                ),

            "preprocessor_version":
                preprocessor.get(
                    "preprocessor_version",
                    "unknown",
                ),
        }

        versioned_schema_path = (
            versioned_dir
            / "feature_schema.json"
        )

        latest_schema_path = (
            self.latest_dir
            / "feature_schema.json"
        )

        with open(
            versioned_schema_path,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                feature_schema,
                f,
                indent=4,
            )

        with open(
            latest_schema_path,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                feature_schema,
                f,
                indent=4,
            )

        # ======================================================
        # 14. Final Validation
        # ======================================================

        self._validate_final_artifacts(
            versioned_dir
        )

        self._validate_final_artifacts(
            self.latest_dir
        )

        # ======================================================
        # 15. Final Output
        # ======================================================

        print("\n" + "=" * 70)

        print(
            "MODEL TRAINING COMPLETED"
        )

        print("=" * 70)

        print(
            f"Best Model          : "
            f"{best_algorithm}"
        )

        print(
            f"Artifact Version    : "
            f"{model_version}"
        )

        print(
            f"MLflow Run ID       : "
            f"{mlflow_result['run_id']}"
        )

        print(
            f"MLflow Model        : "
            f"{self.mlflow_registered_model_name}"
        )

        print(
            f"MLflow Model Version: "
            f"{mlflow_result['model_version']}"
        )

        print(
            f"R2 Score            : "
            f"{best_r2:.4f}"
        )

        print(
            f"MAE                 : "
            f"{best_mae:.4f}"
        )

        print(
            f"RMSE                : "
            f"{best_rmse:.4f}"
        )

        print(
            f"Versioned Artifact  : "
            f"{versioned_dir}"
        )

        print(
            f"Latest Artifact     : "
            f"{self.latest_dir}"
        )

        print("\nProduction Artifacts:")

        print(
            "  - model.pkl"
        )

        print(
            "  - preprocessor.pkl"
        )

        print(
            "  - metadata.json"
        )

        print(
            "  - feature_schema.json"
        )

        print("=" * 70)

        return {

            "model_version":
                model_version,

            "algorithm":
                best_algorithm,

            "r2_score":
                round(
                    best_r2,
                    4,
                ),

            "mae":
                round(
                    best_mae,
                    4,
                ),

            "rmse":
                round(
                    best_rmse,
                    4,
                ),

            "mlflow_run_id":
                mlflow_result[
                    "run_id"
                ],

            "mlflow_model_name":
                self.mlflow_registered_model_name,

            "mlflow_model_version":
                mlflow_result[
                    "model_version"
                ],
        }

    # ==========================================================
    # TRAIN LINEAR REGRESSION
    # ==========================================================

    def _train_linear_regression(
        self,
        X_train,
        X_test,
        y_train,
        y_test,
    ):

        print(
            "\nTraining Linear Regression..."
        )

        model = LinearRegression()

        model.fit(
            X_train,
            y_train,
        )

        predictions = model.predict(
            X_test
        )

        r2 = r2_score(
            y_test,
            predictions,
        )

        mae = mean_absolute_error(
            y_test,
            predictions,
        )

        rmse = root_mean_squared_error(
            y_test,
            predictions,
        )

        print(
            f"Linear Regression R2   : "
            f"{r2:.4f}"
        )

        print(
            f"Linear Regression MAE  : "
            f"{mae:.4f}"
        )

        print(
            f"Linear Regression RMSE : "
            f"{rmse:.4f}"
        )

        # ------------------------------------------------------
        # MLflow run
        # ------------------------------------------------------

        with mlflow.start_run(
            run_name="Linear_Regression"
        ):

            mlflow.log_param(
                "algorithm",
                "Linear Regression",
            )

            mlflow.log_param(
                "training_samples",
                len(X_train),
            )

            mlflow.log_param(
                "testing_samples",
                len(X_test),
            )

            mlflow.log_metric(
                "R2",
                r2,
            )

            mlflow.log_metric(
                "MAE",
                mae,
            )

            mlflow.log_metric(
                "RMSE",
                rmse,
            )

            mlflow.sklearn.log_model(
                model,
                name="linear_regression_model",
            )

        return {

            "model":
                model,

            "algorithm":
                "Linear Regression",

            "r2":
                r2,

            "mae":
                mae,

            "rmse":
                rmse,
        }

    # ==========================================================
    # TRAIN XGBOOST
    # ==========================================================

    def _train_xgboost(
        self,
        X_train,
        X_test,
        y_train,
        y_test,
    ):

        print(
            "\nTraining XGBoost..."
        )

        model = xgb.XGBRegressor(

            objective=
                "reg:squarederror",

            n_estimators=
                100,

            learning_rate=
                0.05,

            max_depth=
                6,

            min_child_weight=
                3,

            subsample=
                0.8,

            colsample_bytree=
                0.8,

            random_state=
                10,

            n_jobs=
                -1,
        )

        model.fit(
            X_train,
            y_train,
        )

        predictions = model.predict(
            X_test
        )

        r2 = r2_score(
            y_test,
            predictions,
        )

        mae = mean_absolute_error(
            y_test,
            predictions,
        )

        rmse = root_mean_squared_error(
            y_test,
            predictions,
        )

        print(
            f"XGBoost R2   : "
            f"{r2:.4f}"
        )

        print(
            f"XGBoost MAE  : "
            f"{mae:.4f}"
        )

        print(
            f"XGBoost RMSE : "
            f"{rmse:.4f}"
        )

        # ------------------------------------------------------
        # MLflow run
        # ------------------------------------------------------

        with mlflow.start_run(
            run_name="XGBoost"
        ):

            mlflow.log_param(
                "algorithm",
                "XGBoost",
            )

            mlflow.log_param(
                "n_estimators",
                100,
            )

            mlflow.log_param(
                "learning_rate",
                0.05,
            )

            mlflow.log_param(
                "max_depth",
                6,
            )

            mlflow.log_param(
                "min_child_weight",
                3,
            )

            mlflow.log_param(
                "subsample",
                0.8,
            )

            mlflow.log_param(
                "colsample_bytree",
                0.8,
            )

            mlflow.log_metric(
                "R2",
                r2,
            )

            mlflow.log_metric(
                "MAE",
                mae,
            )

            mlflow.log_metric(
                "RMSE",
                rmse,
            )

            mlflow.xgboost.log_model(
                model,
                name="xgboost_model",
            )

        return {

            "model":
                model,

            "algorithm":
                "XGBoost",

            "r2":
                r2,

            "mae":
                mae,

            "rmse":
                rmse,
        }

    # ==========================================================
    # SELECT BEST MODEL
    # ==========================================================

    def _select_best_model(
        self,
        lr_results,
        xgb_results,
    ):

        if (
            xgb_results["r2"]
            >= lr_results["r2"]
        ):

            return xgb_results

        return lr_results

    # ==========================================================
    # REGISTER BEST MODEL
    # ==========================================================

    def _register_best_model(
        self,
        best_model,
        best_algorithm,
        best_r2,
        best_mae,
        best_rmse,
        X_train,
        X_test,
        preprocessor,
    ):

        print("\n" + "=" * 70)

        print(
            "REGISTERING BEST MODEL IN MLFLOW"
        )

        print("=" * 70)

        # ------------------------------------------------------
        # Start final model run
        # ------------------------------------------------------

        with mlflow.start_run(
            run_name=(
                f"Best_Model_"
                f"{best_algorithm.replace(' ', '_')}"
            )
        ) as run:

            run_id = run.info.run_id

            # --------------------------------------------------
            # Log model parameters
            # --------------------------------------------------

            mlflow.log_param(
                "algorithm",
                best_algorithm,
            )

            mlflow.log_param(
                "model_selection_metric",
                "R2",
            )

            mlflow.log_param(
                "training_samples",
                len(X_train),
            )

            mlflow.log_param(
                "testing_samples",
                len(X_test),
            )

            mlflow.log_param(
                "feature_count",
                len(
                    preprocessor[
                        "feature_columns"
                    ]
                ),
            )

            # --------------------------------------------------
            # Log metrics
            # --------------------------------------------------

            mlflow.log_metric(
                "R2",
                best_r2,
            )

            mlflow.log_metric(
                "MAE",
                best_mae,
            )

            mlflow.log_metric(
                "RMSE",
                best_rmse,
            )

            # --------------------------------------------------
            # Log model
            # --------------------------------------------------

            if (
                best_algorithm
                == "XGBoost"
            ):

                model_info = (
                    mlflow.xgboost.log_model(
                        best_model,
                        name="production_model",
                    )
                )

            else:

                model_info = (
                    mlflow.sklearn.log_model(
                        best_model,
                        name="production_model",
                    )
                )

            # --------------------------------------------------
            # Register model
            # --------------------------------------------------

            registered_model = (
                mlflow.register_model(
                    model_uri=(
                        model_info.model_uri
                    ),
                    name=(
                        self.mlflow_registered_model_name
                    ),
                )
            )

            mlflow_model_version = (
                str(
                    registered_model.version
                )
            )

            print(
                f"MLflow Run ID        : "
                f"{run_id}"
            )

            print(
                f"Registered Model     : "
                f"{self.mlflow_registered_model_name}"
            )

            print(
                f"Registered Version   : "
                f"{mlflow_model_version}"
            )

            return {

                "run_id":
                    run_id,

                "model_version":
                    mlflow_model_version,

                "registration_status":
                    "REGISTERED",
            }

    # ==========================================================
    # INPUT VALIDATION
    # ==========================================================

    def _validate_inputs(
        self,
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor,
    ):

        if X_train is None:

            raise ValueError(
                "X_train cannot be None."
            )

        if X_test is None:

            raise ValueError(
                "X_test cannot be None."
            )

        if y_train is None:

            raise ValueError(
                "y_train cannot be None."
            )

        if y_test is None:

            raise ValueError(
                "y_test cannot be None."
            )

        if preprocessor is None:

            raise ValueError(
                "preprocessor cannot be None."
            )

        required_keys = [

            "feature_columns",

            "scaling_columns",

        ]

        for key in required_keys:

            if key not in preprocessor:

                raise ValueError(

                    f"Missing required "
                    f"preprocessor key: {key}"

                )

        if list(
            X_train.columns
        ) != list(
            preprocessor[
                "feature_columns"
            ]
        ):

            raise ValueError(

                "X_train columns do not "
                "match preprocessor "
                "feature_columns."

            )

        if list(
            X_test.columns
        ) != list(
            preprocessor[
                "feature_columns"
            ]
        ):

            raise ValueError(

                "X_test columns do not "
                "match preprocessor "
                "feature_columns."

            )

    # ==========================================================
    # FINAL ARTIFACT VALIDATION
    # ==========================================================

    def _validate_final_artifacts(
        self,
        artifact_dir,
    ):

        required_files = [

            "model.pkl",

            "preprocessor.pkl",

            "metadata.json",

            "feature_schema.json",

        ]

        missing_files = [

            file_name

            for file_name
            in required_files

            if not (
                artifact_dir
                / file_name
            ).exists()

        ]

        if missing_files:

            raise FileNotFoundError(

                "Required production "
                "artifacts are missing "
                f"from {artifact_dir}: "
                f"{missing_files}"

            )

        # ------------------------------------------------------
        # Make sure unwanted CSV files are not present
        # ------------------------------------------------------

        unwanted_files = [

            "train.csv",

            "test.csv",

        ]

        for file_name in unwanted_files:

            file_path = (
                artifact_dir
                / file_name
            )

            if file_path.exists():

                print(

                    f"Removing unwanted "
                    f"production artifact: "
                    f"{file_path}"

                )

                file_path.unlink()

        print(

            f"Artifact validation "
            f"successful: "
            f"{artifact_dir}"

        )