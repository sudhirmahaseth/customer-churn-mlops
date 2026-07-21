import os
import sys
import pandas as pd
import joblib
import dvc.api
from src.utils import load_object
from src.logger import logger

class PredictionPipeline:
    def __init__(self):
        # करंट वर्किंग डायरेक्टरी के साथ एब्सोल्यूट पाथ
        self.model_path = os.path.join(os.getcwd(), "models", "model.pkl")
        self.train_data_path = os.path.join(os.getcwd(), "artifacts", "train.csv")

    def _ensure_model_exists(self):
        """अगर मॉडल लोकल डिस्क पर नहीं है, तो डैग्सहब रिमोट से डाउनलोड करके सेव करेगा"""
        if not os.path.exists(self.model_path):
            logger.info(f"Model artifact not found at {self.model_path}. Streaming from DagsHub...")
            try:
                # डायरेक्टरी पक्का करें
                os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
                
                # डैग्सहब के गिट यूआरएल से सीधे मॉडल फ़ाइल को डाउनलोड करना
                with dvc.api.open(
                    path="models/model.pkl",
                    repo="https://dagshub.com/sudhirmahaseth/customer-churn-mlops.git",
                    mode="rb"
                ) as remote_file:
                    with open(self.model_path, "wb") as local_file:
                        local_file.write(remote_file.read())
                logger.info(f"Successfully downloaded and cached model at {self.model_path}")
            except Exception as e:
                logger.error(f"Failed to stream model from DagsHub: {str(e)}")
                raise e

    def predict(self, features: pd.DataFrame):
        try:
            # 🎯 मैजिक स्टेप: प्रेडिक्शन से पहले पक्का करें कि मॉडल डिस्क पर आ चुका है
            self._ensure_model_exists()

            logger.info(f"Loading trained model from {self.model_path} for prediction...")
            model = load_object(self.model_path)
            
            # 1. ट्रेनिंग डेटा के कॉलम्स रीड करें
            if os.path.exists(self.train_data_path):
                logger.info(f"Baseline columns found at {self.train_data_path}")
                train_df = pd.read_csv(self.train_data_path, nrows=1)
                expected_columns = train_df.drop(columns=["Churn"], errors="ignore").columns.tolist()
            else:
                logger.warning(f"train.csv not found at {self.train_data_path}. Using model's feature names or fallback alignment.")
                if hasattr(model, "feature_names_in_"):
                    expected_columns = model.feature_names_in_.tolist()
                else:
                    features_encoded_temp = pd.get_dummies(features, drop_first=True)
                    expected_columns = features_encoded_temp.columns.tolist()

            # 2. इनपुट डेटा पर get_dummies चलाएं
            logger.info("Applying One-Hot Encoding to input features...")
            features_encoded = pd.get_dummies(features, drop_first=True)
            
            # 3. ट्रेनिंग कॉलम्स के साथ अलाइन करें
            features_aligned = features_encoded.reindex(columns=expected_columns, fill_value=0)
            
            logger.info("Making prediction on aligned features...")
            preds = model.predict(features_aligned)
            
            return preds
        except Exception as e:
            logger.error(f"Error occurred in PredictionPipeline: {str(e)}")
            raise e