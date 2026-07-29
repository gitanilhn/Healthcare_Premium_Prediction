from fastapi.testclient import TestClient

from app import app


client = TestClient(app)


def get_valid_prediction_payload():
    return {
        "age": 35,
        "number_of_dependants": 2,
        "income_level": "10L - 25L",
        "income_lakhs": 15,
        "insurance_plan": "Gold",
        "medical_history": "No Disease",
        "physical_activity": "Medium",
        "stress_level": "Medium",
        "gender": "Male",
        "region": "South",
        "marital_status": "Married",
        "bmi_category": "Normal",
        "smoking_status": "No",
        "employment_status": "Salaried",
    }


def test_prediction_endpoint():
    response = client.post(
        "/predict",
        json=get_valid_prediction_payload(),
    )

    assert response.status_code == 200


def test_prediction_response_contains_data():
    response = client.post(
        "/predict",
        json=get_valid_prediction_payload(),
    )

    assert response.status_code == 200

    data = response.json()

    assert "predicted_premium" in data
    assert isinstance(data["predicted_premium"], (int, float))
    assert data["predicted_premium"] > 0

    assert "model_version" in data
    assert isinstance(data["model_version"], str)
    assert len(data["model_version"]) > 0
