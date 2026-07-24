from fastapi import FastAPI, HTTPException

from schemas import PredictionRequest

from src.prediction import predictor


app = FastAPI(
    title="Healthcare Premium Prediction API",
    version="1.0.0",
)


@app.get("/")
def root():

    return {
        "message": "Healthcare Premium Prediction API",
        "model_version": predictor.metadata["model_version"],
        "algorithm": predictor.metadata["algorithm"],
    }


@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": True,
        "model_version": predictor.metadata["model_version"],
    }


@app.post("/predict")
def predict(
    request: PredictionRequest,
):

    try:

        result = predictor.predict(request.model_dump())

        return result

    except Exception as e:

        raise HTTPException(status_code=500, detail=str(e))
