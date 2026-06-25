import uvicorn
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.prediction_pipeline import CustomData, PredictionPipeline
from src.logger import logger

# 1. FastAPI ऐप इनिशियलाइज करें (आपके टेस्ट_होम के मुताबिक टाइटल्स और रूट्स)
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

# 3. अपडेटेड प्रेडिक्ट रूट जो टेस्ट और प्रोडक्शन पाइपलाइन दोनों को संभालेगा
@app.post("/predict")
def predict(data: ChurnInput):
    try:
        logger.info("API received a new prediction request")
        
        # Pydantic डेटा को dict में बदलकर कस्टम डेटा ऑब्जेक्ट बनाना
        data_dict = data.model_dump()
        
        # 🎯 सुरक्षा के लिए: अगर टेस्ट एनवायरनमेंट से 'charges_per_tenure' नहीं आ रहा है, 
        # और आपकी CustomData क्लास इसे ढूंढ रही है, तो इसे यहीं जोड़ देते हैं।
        if "charges_per_tenure" not in data_dict:
            data_dict["charges_per_tenure"] = data_dict["TotalCharges"] / (data_dict["tenure"] if data_dict["tenure"] > 0 else 1)

        custom_data = CustomData(**data_dict)
        
        # डेटा को DataFrame में कन्वर्ट करें
        df = custom_data.get_data_as_data_frame()
        
        # प्रेडिक्शन पाइपलाइन रन करें
        pipeline = PredictionPipeline()
        prediction = pipeline.predict(df)
        
        result = int(prediction[0])
        
        # 🎯 टेस्ट स्क्रिप्ट 'assert "status" in json_response' को चेक कर रही है।
        # यहाँ हम स्टेटस में "Success" या आपका कस्टम मैसेज भेज सकते हैं।
        return {
            "prediction": result,
            "status": "Success"
        }
        
    except Exception as e:
        logger.error(f"Error in API execution: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)