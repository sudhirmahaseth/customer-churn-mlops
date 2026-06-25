import os
import pickle
import shutil
import pandas as pd
from fastapi.testclient import TestClient
from api.main import app

# FastAPI TestClient इनिशियलाइज करें
client = TestClient(app)

# डमी क्लास ताकि बिना असली sklearn मॉडल के भी .predict() मेथड काम कर सके
class DummyModel:
    def predict(self, df):
        # हमेशा 1 रिटर्न करेगा (आपकी टेस्ट कंडीशंस के मुताबिक)
        return [1]

# 🛠️ टेस्ट शुरू होने से पहले डमी एनवायरनमेंट सेटअप करें
def setup_module(module):
    print("\nSetting up dynamic dummy environment for tests...")
    
    # 1. artifacts फोल्डर और डमी train.csv बनाएं ताकि FileNotFoundError न आए
    os.makedirs("artifacts", exist_ok=True)
    dummy_df = pd.DataFrame(columns=[
        "gender", "SeniorCitizen", "Partner", "Dependents", "tenure", 
        "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity", 
        "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", 
        "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod", 
        "MonthlyCharges", "TotalCharges", "charges_per_tenure", "Churn"
    ])
    dummy_df.to_csv("artifacts/train.csv", index=False)

    # 2. models फोल्डर और डमी model.pkl बनाएं ताकि पाइपलाइन इसे लोड कर सके
    os.makedirs("models", exist_ok=True)
    dummy_model = DummyModel()
    with open("models/model.pkl", "wb") as f:
        pickle.dump(dummy_model, f)

# 🧹 टेस्ट खत्म होते ही नकली फाइलों को साफ करें (ताकि Docker Build में ये न जाएं)
def teardown_module(module):
    print("\nCleaning up dummy files from environment...")
    if os.path.exists("artifacts/train.csv"):
        os.remove("artifacts/train.csv")
    if os.path.exists("models/model.pkl"):
        os.remove("models/model.pkl")
    print("Test environment cleaned up successfully!")

# 1. होम रूट का टेस्ट
def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "API is running successfully!"}

# 2. प्रेडिक्शन एंडपॉइंट का टेस्ट
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

    # असर्शन्स (Assertions) चेक करें
    assert response.status_code == 200
    
    json_response = response.json()
    assert "prediction" in json_response
    assert "status" in json_response
    assert json_response["status"] == "Success"
    assert json_response["prediction"] in [0, 1]