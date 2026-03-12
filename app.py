from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import os
import logging

app = FastAPI(title="Customer Churn Prediction API", description="An API to predict customer churn probability.")

MODEL_PATH = "model/churn_rf_model.pkl"
FEATURES_PATH = "model/model_features.pkl"

if not os.path.exists(MODEL_PATH) or not os.path.exists(FEATURES_PATH):
    raise RuntimeError("Model or feature files not found. Ensure train.py has been executed.")

# Load artifacts into memory on startup
model = joblib.load(MODEL_PATH)
model_features = joblib.load(FEATURES_PATH)

class CustomerData(BaseModel):
    """Data schema representing a single customer profile."""
    gender: str = "Female"
    SeniorCitizen: int = 0
    Partner: str = "Yes"
    Dependents: str = "No"
    tenure: int = 1
    PhoneService: str = "No"
    MultipleLines: str = "No phone service"
    InternetService: str = "DSL"
    OnlineSecurity: str = "No"
    OnlineBackup: str = "Yes"
    DeviceProtection: str = "No"
    TechSupport: str = "No"
    StreamingTV: str = "No"
    StreamingMovies: str = "No"
    Contract: str = "Month-to-month"
    PaperlessBilling: str = "Yes"
    PaymentMethod: str = "Electronic check"
    MonthlyCharges: float = 29.85
    TotalCharges: float = 29.85

@app.get("/")
def home():
    """Health check endpoint."""
    return {"message": "Customer Churn Prediction API is running!"}

@app.post("/predict")
def predict_churn(customer: CustomerData):
    """
    Accepts customer profile data, applies required preprocessing, 
    and returns a churn probability prediction.
    """
    try:
        # Convert the json payload to an encoded DataFrame
        df = pd.DataFrame([customer.dict()])
        df_encoded = pd.get_dummies(df, drop_first=True)

        # Align the inference DataFrame structure with the training feature structure
        X_predict = pd.DataFrame(columns=model_features)

        for col in X_predict.columns:
            if col in df_encoded.columns:
                X_predict[col] = df_encoded[col]
            else:
                X_predict[col] = 0

        # Model inference
        prediction_array = model.predict(X_predict)
        prediction_prob = model.predict_proba(X_predict)[0][1]

        return {
            "churn_prediction": bool(prediction_array[0]),
            "churn_probability": float(prediction_prob)
        }
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal inference error.")
