import os
import sys
import pandas as pd
from src.utils import load_object
from src.logger import logger

class PredictionPipeline:
    def __init__(self):
        self.model_path = os.path.join("models", "model.pkl")
        # हम ट्रेनिंग डेटा के कॉलम्स का इस्तेमाल करेंगे अलाइनमेंट के लिए
        self.train_data_path = os.path.join("artifacts", "train.csv")

    def predict(self, features: pd.DataFrame):
        try:
            logger.info(f"Loading trained model from {self.model_path} for prediction...")
            model = load_object(self.model_path)
            
            # 1. ट्रेनिंग डेटा के कॉलम्स रीड करें (बिना 'Churn' के) ताकि ऑर्डर और कॉलम्स परफेक्ट मैच हों
            if os.path.exists(self.train_data_path):
                train_df = pd.read_csv(self.train_data_path, nrows=1)
                expected_columns = train_df.drop(columns=["Churn"], errors="ignore").columns.tolist()
            else:
                raise FileNotFoundError(f"Required baseline columns not found at {self.train_data_path}")

            # 2. इनपुट डेटा पर get_dummies चलाएं
            logger.info("Applying One-Hot Encoding to input features...")
            features_encoded = pd.get_dummies(features, drop_first=True)
            
            # 3. 🎯 मैजिक स्टेप: ट्रेनिंग कॉलम्स के साथ अलाइन करें (जो मिसिंग हैं वहां 0 आ जाएगा)
            features_aligned = features_encoded.reindex(columns=expected_columns, fill_value=0)
            
            logger.info("Making prediction on aligned features...")
            preds = model.predict(features_aligned)
            
            return preds
        except Exception as e:
            logger.error(f"Error occurred in PredictionPipeline: {str(e)}")
            raise e

class CustomData:
    def __init__(self, 
                 gender: str,
                 SeniorCitizen: int,
                 Partner: str,
                 Dependents: str,
                 tenure: int,
                 PhoneService: str,
                 MultipleLines: str,
                 InternetService: str,
                 OnlineSecurity: str,
                 OnlineBackup: str,
                 DeviceProtection: str,
                 TechSupport: str,
                 StreamingTV: str,
                 StreamingMovies: str,
                 Contract: str,
                 PaperlessBilling: str,
                 PaymentMethod: str,
                 MonthlyCharges: float,
                 TotalCharges: float,
                 charges_per_tenure: float = None):

        self.gender = gender
        self.SeniorCitizen = SeniorCitizen
        self.Partner = Partner
        self.Dependents = Dependents
        self.tenure = tenure
        self.PhoneService = PhoneService
        self.MultipleLines = MultipleLines
        self.InternetService = InternetService
        self.OnlineSecurity = OnlineSecurity
        self.OnlineBackup = OnlineBackup
        self.DeviceProtection = DeviceProtection
        self.TechSupport = TechSupport
        self.StreamingTV = StreamingTV
        self.StreamingMovies = StreamingMovies
        self.Contract = Contract
        self.PaperlessBilling = PaperlessBilling
        self.PaymentMethod = PaymentMethod
        self.MonthlyCharges = MonthlyCharges
        self.TotalCharges = TotalCharges
        
        if charges_per_tenure is None:
            self.charges_per_tenure = TotalCharges / (tenure if tenure > 0 else 1)
        else:
            self.charges_per_tenure = charges_per_tenure

    def get_data_as_data_frame(self) -> pd.DataFrame:
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "SeniorCitizen": [self.SeniorCitizen],
                "Partner": [self.Partner],
                "Dependents": [self.Dependents],
                "tenure": [self.tenure],
                "PhoneService": [self.PhoneService],
                "MultipleLines": [self.MultipleLines],
                "InternetService": [self.InternetService],
                "OnlineSecurity": [self.OnlineSecurity],
                "OnlineBackup": [self.OnlineBackup],
                "DeviceProtection": [self.DeviceProtection],
                "TechSupport": [self.TechSupport],
                "StreamingTV": [self.StreamingTV],
                "StreamingMovies": [self.StreamingMovies],
                "Contract": [self.Contract],
                "PaperlessBilling": [self.PaperlessBilling],
                "PaymentMethod": [self.PaymentMethod],
                "MonthlyCharges": [self.MonthlyCharges],
                "TotalCharges": [self.TotalCharges],
                "charges_per_tenure": [self.charges_per_tenure]
            }
            
            df = pd.DataFrame(custom_data_input_dict)
            logger.info("Custom data successfully converted to DataFrame")
            return df
            
        except Exception as e:
            logger.error(f"Error in converting CustomData to DataFrame: {str(e)}")
            raise e