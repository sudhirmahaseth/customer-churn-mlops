import pandas as pd
import os

from src.logger import logger

class DataIngestion:

    def initiate_data_ingestion(self):

        logger.info("Reading dataset")

        df = pd.read_csv("data/churn.csv")

        os.makedirs("artifacts", exist_ok=True)

        df.to_csv(
            "artifacts/raw.csv",
            index=False
        )

        logger.info("Dataset saved")

        return "artifacts/raw.csv"


if __name__ == "__main__":

    obj = DataIngestion()

    obj.initiate_data_ingestion()