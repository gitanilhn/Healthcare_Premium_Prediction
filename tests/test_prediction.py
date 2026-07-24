from src.prediction import predictor


def get_valid_prediction_input():
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
        "region": "Southwest",
        "marital_status": "Married",
        "bmi_category": "Normal",
        "smoking_status": "Never",
        "employment_status": "Salaried",
    }


def test_prediction_returns_result():
    result = predictor.predict(get_valid_prediction_input())

    assert result is not None
    assert isinstance(result, dict)


def test_prediction_is_numeric():
    result = predictor.predict(get_valid_prediction_input())

    assert "prediction" in result
    assert isinstance(result["prediction"], (int, float))


def test_prediction_is_positive():
    result = predictor.predict(get_valid_prediction_input())

    assert result["prediction"] > 0


def test_prediction_is_consistent():
    input_data = get_valid_prediction_input()

    result_1 = predictor.predict(input_data)
    result_2 = predictor.predict(input_data)

    assert result_1 == result_2
