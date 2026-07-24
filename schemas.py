from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):

    age: int = Field(
        ...,
        ge=18,
        le=100,
    )

    number_of_dependants: int = Field(
        ...,
        ge=0,
    )

    income_level: str

    income_lakhs: float = Field(
        ...,
        gt=0,
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
