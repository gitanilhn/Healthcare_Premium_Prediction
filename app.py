from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.prediction import predictor


# ==========================================================
# FastAPI Application
# ==========================================================

app = FastAPI(
    title="Healthcare Premium Prediction API",
    description=(
        "ML API for predicting healthcare insurance premiums."
    ),
    version="1.0.0",
)


# ==========================================================
# Request Schema
# ==========================================================

class PredictionRequest(BaseModel):

    age: int = Field(
        ...,
        ge=0,
        description="Age of the individual",
    )

    number_of_dependants: int = Field(
        ...,
        ge=0,
        description="Number of dependants",
    )

    income_level: str = Field(
        ...,
        description="Income category",
    )

    income_lakhs: float = Field(
        ...,
        ge=0,
        description="Annual income in lakhs",
    )

    insurance_plan: str

    medical_history: str

    physical_activity: str

    stress_level: str

    gender: str

    region: str

    marital_status: str

    bmi_category: str

    smoking_status: str

    employment_status: str


# ==========================================================
# Response Schema
# ==========================================================

class PredictionResponse(BaseModel):

    predicted_premium: float

    model_version: str


# ==========================================================
# Health Check
# ==========================================================

@app.get(
    "/health",
)
def health_check():

    return {
        "status": "healthy",
        "service": "healthcare-premium-prediction",
    }


# ==========================================================
# Readiness Check
# ==========================================================

@app.get(
    "/ready",
)
def readiness_check():

    if not predictor.is_loaded:

        raise HTTPException(
            status_code=503,
            detail="Prediction model is not ready.",
        )

    return {
        "status": "ready",
        "model_loaded": True,
        "model_version": predictor.model_version,
    }


# ==========================================================
# Prediction Endpoint
# ==========================================================

@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(
    request: PredictionRequest,
):

    try:

        # Convert validated Pydantic model
        # into dictionary

        input_data = request.model_dump()


        # Run prediction

        prediction = predictor.predict(
            input_data
        )


        # Return prediction

        return PredictionResponse(

            predicted_premium=float(
                prediction
            ),

            model_version=(
                predictor.model_version
            ),
        )


    except ValueError as e:

        raise HTTPException(
            status_code=400,
            detail=str(e),
        )


    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=(
                "Prediction failed: "
                f"{str(e)}"
            ),
        )