import uvicorn
import os
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.prediction_pipeline import PredictionPipeline
from src.logger import logger

# 1. FastAPI ऐप इनिशियलाइज करें
app = FastAPI(title="Customer Churn MLOps API", version="1.0.0")

# 2. इनपुट डेटा वैलिडेशन के लिए Pydantic Model
class ChurnInput(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

# टेस्ट स्क्रिप्ट 'test_home' को पास करने के लिए रूट
@app.get("/")
def read_root():
    return {"status": "API is running successfully!"}

# 3. अपडेटेड प्रेडिक्ट रूट (बिना CustomData के, सीधा DataFrame कन्वर्शन)
@app.post("/predict")
def predict(data: ChurnInput):
    try:
        logger.info("API received a new prediction request")
        
        # Pydantic डेटा को dict में बदलें
        data_dict = data.model_dump()
        
        # Feature Engineering: 'charges_per_tenure' कैलकुलेट करें
        if "charges_per_tenure" not in data_dict:
            data_dict["charges_per_tenure"] = data_dict["TotalCharges"] / (data_dict["tenure"] if data_dict["tenure"] > 0 else 1)

        # 🎯 डायरेक्ट और क्लीन तरीका: डिक्शनरी को सीधे pandas DataFrame में कन्वर्ट करें
        df = pd.DataFrame([data_dict])
        
        # प्रेडिक्शन पाइपलाइन रन करें
        pipeline = PredictionPipeline()
        prediction = pipeline.predict(df)
        
        result = int(prediction[0])
        
        return {
            "prediction": result,
            "status": "Success"
        }
        
    except Exception as e:
        logger.error(f"Error in API execution: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)