from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from src.prediction_pipeline import CustomData, PredictionPipeline
from src.logger import logger
import uvicorn

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

@app.get("/")
def read_root():
    return {"status": "API is running successfully!"}

# 3. आपके ओरिजिनल प्रेडिक्ट रूट को अपडेट किया गया है
@app.post("/predict")
def predict(data: ChurnInput):
    try:
        logger.info("API received a new prediction request")
        
        # Pydantic डेटा को dict में बदलकर CustomData में पास करें
        data_dict = data.model_dump()
        custom_data = CustomData(**data_dict)
        
        # डेटा को DataFrame में कन्वर्ट करें
        df = custom_data.get_data_as_data_frame()
        
        # प्रेडिक्शन पाइपलाइन रन करें
        pipeline = PredictionPipeline()
        prediction = pipeline.predict(df)
        
        result = int(prediction[0])
        status = "Customer will Churn" if result == 1 else "Customer will Stay"
        
        return {
            "prediction": result,
            "status": status
        }
        
    except Exception as e:
        logger.error(f"Error in API execution: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)