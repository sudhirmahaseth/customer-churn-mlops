import os
import joblib
import dvc.api
from src.logger import logger

class PredictionPipeline:
    def __init__(self):
        # 🎯 पाथ्स डिफाइन करें
        self.model_path = os.path.join("models", "model.pkl")
        self.preprocessor_path = os.path.join("models", "preprocessor.pkl")
        
        # 1. मॉडल लोड करें (अगर लोकल नहीं है तो DagsHub से लाइव स्ट्रीम करेगा)
        self.model = self._load_artifact(self.model_path)
        
        # 2. प्रीप्रोसेसर लोड करें
        self.preprocessor = self._load_artifact(self.preprocessor_path)

    def _load_artifact(self, local_path):
        """अगर फाइल लोकल नहीं मिलती, तो उसे DagsHub DVC Repository से लाइव लोड करेगा"""
        if not os.path.exists(local_path):
            logger.info(f"Artifact {local_path} not found locally. Streaming from DagsHub via DVC API...")
            try:
                # यह बिना लोकल क्रेडेंशियल्स के डैग्सहब से सीधे बाइनरी फ़ाइल स्ट्रीम कर लेगा
                with dvc.api.open(
                    path=local_path.replace("\\", "/"), # Windows/Linux पाथ कम्पैटिबिलिटी के लिए
                    repo="https://dagshub.com/sudhirmahaseth/customer-churn-mlops.git",
                    mode="rb"
                ) as f:
                    return joblib.load(f)
            except Exception as e:
                logger.error(f"Failed to stream artifact from DagsHub: {str(e)}")
                raise e
        else:
            logger.info(f"Loading artifact from local path: {local_path}")
            return joblib.load(local_path)

    def predict(self, features):
        try:
            logger.info("Transforming features using preprocessor...")
            # अगर प्रीप्रोसेसर पाइपलाइन का हिस्सा है, तो पहले ट्रांसफॉर्म करें
            if self.preprocessor:
                # नोट: अगर आपका preprocessor.pkl स्केलर/इन्कोडर है, तो transform चलाएं
                # अगर आपकी पाइपलाइन केवल मॉडल आधारित है, तो इसे अपने पुराने लॉजिक के अनुसार रखें
                pass 
            
            logger.info("Making prediction using the model...")
            prediction = self.model.predict(features)
            return prediction
        except Exception as e:
            logger.error(f"Error in prediction pipeline: {str(e)}")
            raise e