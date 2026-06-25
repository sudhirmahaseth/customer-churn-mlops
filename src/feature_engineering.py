import os
import pandas as pd
from sklearn.model_selection import train_test_split # Split करने के लिए
from src.logger import logger

class FeatureEngineering:

    def transform(self, df):
        logger.info("Creating new feature: charges_per_tenure")
        if "MonthlyCharges" in df.columns and "tenure" in df.columns:
            df["charges_per_tenure"] = df["MonthlyCharges"] / (df["tenure"] + 1)
        return df

if __name__ == "__main__":
    input_path = "artifacts/processed.csv"
    
    logger.info(f"Reading processed data from {input_path}")
    df = pd.read_csv(input_path)
    
    # 1. Feature Engineering अप्लाई करें
    fe = FeatureEngineering()
    df = fe.transform(df)
    
    # 2. डेटा को Train और Test सेट्स में स्प्लिट करें (MLOps Standard)
    logger.info("Splitting dataset into train and test sets")
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # 3. Artifacts फ़ोल्डर में सेव करें
    train_path = "artifacts/train.csv"
    test_path = "artifacts/test.csv"
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    logger.info(f"Feature engineering completed. Train/Test shapes: {train_df.shape} / {test_df.shape}")
    print(f"Success! Saved train.csv and test.csv in artifacts/")