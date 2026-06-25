import os
import pandas as pd
from src.logger import logger

class DataPreprocessing:

    def preprocess(self, df):
        logger.info("Dropping missing values")
        df.dropna(inplace=True)
        
        target_col = "Churn"
        
        if target_col in df.columns:
            logger.info(f"Target column '{target_col}' found. Encoding it to 0 and 1.")
            # Churn column को 1 और 0 में मैप कर रहे हैं ताकि get_dummies इसे न छुए
            df[target_col] = df[target_col].map({'Yes': 1, 'No': 0, 1: 1, 0: 0})
            
            # Target को अलग निकालें
            target = df[target_col]
            features = df.drop(columns=[target_col])
            
            # सिर्फ फीचर्स पर get_dummies चलाएं
            features = pd.get_dummies(features, drop_first=True)
            
            # वापस जोड़ें
            df = features.copy()
            df[target_col] = target
        else:
            logger.warning(f"Target column '{target_col}' NOT found in input data!")
            df = pd.get_dummies(df, drop_first=True)
            
        return df

if __name__ == "__main__":
    input_path = "artifacts/raw.csv"
    output_path = "artifacts/processed.csv"
    
    logger.info(f"Reading data from {input_path}")
    df = pd.read_csv(input_path)
    
    processor = DataPreprocessing()
    processed_df = processor.preprocess(df)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    processed_df.to_csv(output_path, index=False)
    
    logger.info(f"Preprocessing completed. Saved to {output_path}")
    print(f"Success! Saved to {output_path}")