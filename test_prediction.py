from src.prediction import predictor


input_data = {

    "age": 35,

    "number_of_dependants": 2,

    "income_level": "10L - 25L",

    "income_lakhs": 12.5,

    "insurance_plan": "Silver",

    "medical_history": "no disease",

    "physical_activity": "Medium",

    "stress_level": "Medium",

    "gender": "Male",

    "region": "Northwest",

    "marital_status": "Unmarried",

    "bmi_category": "Overweight",

    "smoking_status": "Occasional",

    "employment_status": "Salaried",
}


print("=" * 70)

print("LOCAL MODEL PREDICTION TEST")

print("=" * 70)


try:

    result = predictor.predict(
        input_data
    )

    print()

    print("Prediction successful")

    print()

    print(
        "Annual Premium Prediction :",
        result["prediction"]
    )

    print(
        "Model Version             :",
        result["model_version"]
    )

    print(
        "Algorithm                 :",
        result["algorithm"]
    )

    print(
        "Model Metrics             :",
        result["model_metrics"]
    )

    print()

    print("=" * 70)

except Exception as e:

    print()

    print(
        "Prediction failed"
    )

    print(
        "Error:",
        str(e)
    )

    print("=" * 70)

    raise