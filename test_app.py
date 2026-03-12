from fastapi.testclient import TestClient
from app import app
import logging

client = TestClient(app)

def test_home():
    """Verify the root health check endpoint responds correctly."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Customer Churn Prediction API is running!"}

def test_predict_churn_high_risk_customer():
    """
    Test the /predict endpoint using a synthetic high-risk customer profile.
    Confirms probability keys exist and valid HTTP status code is returned.
    """
    dummy_input = {
      "gender": "Female",
      "SeniorCitizen": 1,
      "Partner": "No",
      "Dependents": "No",
      "tenure": 1,
      "PhoneService": "Yes",
      "MultipleLines": "Yes",
      "InternetService": "Fiber optic",
      "OnlineSecurity": "No",
      "OnlineBackup": "No",
      "DeviceProtection": "No",
      "TechSupport": "No",
      "StreamingTV": "Yes",
      "StreamingMovies": "No",
      "Contract": "Month-to-month",
      "PaperlessBilling": "Yes",
      "PaymentMethod": "Electronic check",
      "MonthlyCharges": 99.00,
      "TotalCharges": 99.00
    }
    
    response = client.post("/predict", json=dummy_input)
    assert response.status_code == 200
    
    data = response.json()
    assert "churn_prediction" in data
    assert "churn_probability" in data
    
    logging.info(f"Test Inference Response -> Churn Predict: {data['churn_prediction']}, Probability: {data['churn_probability']:.4f}")
