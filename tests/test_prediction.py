from src.prediction import predict_premium


input_data = {
    "age": 35,
    "gender": "Male",
    "region": "Southwest",
    "marital_status": "Married",
    "physical_activity": "Medium",
    "stress_level": "Medium",
    "number_of_dependants": 2,
    "bmi_category": "Overweight",
    "smoking_status": "Occasional",
    "employment_status": "Salaried",
    "income_level": "10L - 25L",
    "income_lakhs": 15,
    "medical_history": "No Disease",
    "insurance_plan": "Gold",
}


prediction = predict_premium(input_data)

print("=" * 60)
print("PREDICTION TEST")
print("=" * 60)
print("Input:")
print(input_data)
print()
print("Predicted Annual Premium:", prediction)
print("=" * 60)