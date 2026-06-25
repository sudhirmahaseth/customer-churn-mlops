import os
import pickle
import pandas as pd
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

# डमी क्लास ताकि sklearn के बिना भी predict() मेथड काम कर सके
class DummyModel:
    def predict(self, df):
        # हमेशा 1 या 0 रिटर्न करेगा (आपके टेस्ट असर्शन के मुताबिक)
        return [1]

def setup_module(module):
    # 1. artifacts फोल्डर और डमी train.csv बनाएं
    os.makedirs("artifacts", exist_ok=True)
    dummy_df = pd.DataFrame(columns=[
        "gender", "SeniorCitizen", "Partner", "Dependents", "tenure", 
        "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity", 
        "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", 
        "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod", 
        "MonthlyCharges", "TotalCharges", "charges_per_tenure", "Churn"
    ])
    dummy_df.to_csv("artifacts/train.csv", index=False)

    # 2. 🎯 models फोल्डर और डमी model.pkl बनाएं ताकि FileNotFoundError न आए
    os.makedirs("models", exist_ok=True)
    dummy_model = DummyModel()
    with open("models/model.pkl", "wb") as f:
        pickle.dump(dummy_model, f)

def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "API is running successfully!"}

def test_predict():
    payload = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "No",
        "MultipleLines": "No phone service",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 29.85,
        "TotalCharges": 29.85
    }

    response = client.post(
        "/predict",
        json=payload
    )

    assert response.status_code == 200
    
    json_response = response.json()
    assert "prediction" in json_response
    assert "status" in json_response
    assert json_response["prediction"] in [0, 1]