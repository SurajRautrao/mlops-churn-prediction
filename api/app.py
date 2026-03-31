'''
from fastapi import FastAPI
import joblib
import pandas as pd
from catboost import Pool

app = FastAPI()

# Load CatBoost model
model = joblib.load("models/catboost_model.pkl")

# Default values for missing columns
default_values = {
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70,
    "TotalCharges": 300,
    "tenure": 12
}

@app.post("/predict")
def predict(data: dict):
    try:
        # ------------------ Merge Input ------------------
        full_data = default_values.copy()
        full_data.update(data)

        df = pd.DataFrame([full_data])

        # ------------------ Data Cleaning ------------------
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce")
        df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce")

        df = df.fillna(0)

        # ------------------ Prediction ------------------
        # Identify categorical columns
        cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

        # Convert to string (safety)
        for col in cat_cols:
            df[col] = df[col].astype(str)

        # Create Pool
        pool = Pool(df, cat_features=cat_cols)
        pred = model.predict_proba(pool)[0][1]

        return {"churn_probability": float(pred)}

    except Exception as e:
        return {"error": str(e)}
'''

from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import pandas as pd
from catboost import Pool
from typing import Optional

app = FastAPI()

# ------------------ Load Model ------------------
model = joblib.load("models/catboost_model.pkl")

# ------------------ Default Values ------------------
default_values = {
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70,
    "TotalCharges": 300,
    "tenure": 12
}

# ------------------ Pydantic Model ------------------
class CustomerInput(BaseModel):
    tenure: Optional[int] = Field(12, example=12)
    MonthlyCharges: Optional[float] = Field(70, example=70)
    TotalCharges: Optional[float] = Field(800, example=800)
    Contract: Optional[str] = Field("Month-to-month", example="Month-to-month")
    InternetService: Optional[str] = Field("DSL", example="DSL")

    # Optional additional fields (advanced UI support)
    gender: Optional[str] = "Male"
    SeniorCitizen: Optional[int] = 0
    Partner: Optional[str] = "No"
    Dependents: Optional[str] = "No"
    PhoneService: Optional[str] = "Yes"
    MultipleLines: Optional[str] = "No"
    OnlineSecurity: Optional[str] = "No"
    OnlineBackup: Optional[str] = "No"
    DeviceProtection: Optional[str] = "No"
    TechSupport: Optional[str] = "No"
    StreamingTV: Optional[str] = "No"
    StreamingMovies: Optional[str] = "No"
    PaperlessBilling: Optional[str] = "Yes"
    PaymentMethod: Optional[str] = "Electronic check"

# ------------------ API Endpoint ------------------
@app.post("/predict")
def predict(data: CustomerInput):
    try:
        # Convert Pydantic model → dict
        input_data = data.dict()

        # Merge with defaults
        full_data = default_values.copy()
        full_data.update(input_data)

        df = pd.DataFrame([full_data])

        # ------------------ Data Cleaning ------------------
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce")
        df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce")

        df = df.fillna(0)

        # ------------------ CatBoost Handling ------------------
        cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

        for col in cat_cols:
            df[col] = df[col].astype(str)

        pool = Pool(df, cat_features=cat_cols)

        # ------------------ Prediction ------------------
        pred = model.predict_proba(pool)[0][1]

        return {"churn_probability": float(pred)}

    except Exception as e:
        return {"error": str(e)}
