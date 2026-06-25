import os
import sys
import pandas as pd
from src.utils import load_object
from src.logger import logger

class PredictionPipeline:
    def __init__(self):
        # 💻 बेस्ट प्रैक्टिस: करंट वर्किंग डायरेक्टरी के साथ एब्सोल्यूट पाथ बनाना
        self.model_path = os.path.join(os.getcwd(), "models", "model.pkl")
        self.train_data_path = os.path.join(os.getcwd(), "artifacts", "train.csv")

    def predict(self, features: pd.DataFrame):
        try:
            logger.info(f"Loading trained model from {self.model_path} for prediction...")
            model = load_object(self.model_path)
            
            # 1. ट्रेनिंग डेटा के कॉलम्स रीड करें
            if os.path.exists(self.train_data_path):
                logger.info(f"Baseline columns found at {self.train_data_path}")
                train_df = pd.read_csv(self.train_data_path, nrows=1)
                expected_columns = train_df.drop(columns=["Churn"], errors="ignore").columns.tolist()
            else:
                # 🛡️ फॉलबैक: अगर CI/CD पाइपलाइन या डमी रन में train.csv नहीं है, तो मॉडल की अपनी प्रॉपर्टी का इस्तेमाल करें
                logger.warning(f"train.csv not found at {self.train_data_path}. Using model's feature names or fallback alignment.")
                if hasattr(model, "feature_names_in_"):
                    expected_columns = model.feature_names_in_.tolist()
                else:
                    # अगर मॉडल में भी फ़ीचर नेम नहीं हैं, तो इनपुट इनकोडेड फ़ीचर्स को ही बेस मान लें
                    features_encoded_temp = pd.get_dummies(features, drop_first=True)
                    expected_columns = features_encoded_temp.columns.tolist()

            # 2. इनपुट डेटा पर get_dummies चलाएं
            logger.info("Applying One-Hot Encoding to input features...")
            features_encoded = pd.get_dummies(features, drop_first=True)
            
            # 3. 🎯 मैजिक स्टेप: ट्रेनिंग कॉलम्स के साथ अライン करें
            features_aligned = features_encoded.reindex(columns=expected_columns, fill_value=0)
            
            logger.info("Making prediction on aligned features...")
            preds = model.predict(features_aligned)
            
            return preds
        except Exception as e:
            logger.error(f"Error occurred in PredictionPipeline: {str(e)}")
            raise e