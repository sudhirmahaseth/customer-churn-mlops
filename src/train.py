import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from src.utils import save_object
from src.config import TrainingConfig
from src.logger import logger
import mlflow
import mlflow.sklearn

class ModelTrainer:

    def train_model(self):
        config = TrainingConfig()

        # 1. Feature Engineering से बनी train.csv को रीड करें
        input_path = "artifacts/train.csv"
        logger.info(f"Loading training data from {input_path}")
        df = pd.read_csv(input_path)

        # 2. X (Features) और y (Target) को अलग करें
        X_train = df.drop(config.target_column, axis=1)
        y_train = df[config.target_column]

        logger.info("Starting MLflow experiment run...")
        with mlflow.start_run():

            # 3. हाइपरपैरामीटर्स डिक्शनरी (Overfitting रोकने के लिए max_depth=8 सेट किया है)
            rf_params = {
                "n_estimators": 100,
                "max_depth": 8,
                "random_state": 42,
                "class_weight": "balanced"  # <-- यह मैजिक लाइन जोड़ें
            }
            
            # MLflow में सारे पैरामीटर्स एक साथ लॉग करें
            mlflow.log_params(rf_params)

            # मॉडल इनिशियलाइज और फिट करें
            model = RandomForestClassifier(**rf_params)
            
            logger.info("Training Random Forest Model...")
            model.fit(X_train, y_train)

            # 4. ट्रेनिंग एक्यूरेसी कैलकुलेट करें
            pred = model.predict(X_train)
            accuracy = accuracy_score(y_train, pred)

            # 5. MLflow में मैट्रिक्स लॉक करना
            mlflow.log_metric("train_accuracy", accuracy)

            # 6. MLflow Artifacts में मॉडल को लॉग करना (इससे UI में Logged Models दिखेगा)
            logger.info("Logging model to MLflow Artifacts...")
            mlflow.sklearn.log_model(
                sk_model=model, 
                artifact_path="model", 
                registered_model_name="CustomerChurnRFModel"
            )

            # 7. लोकल डिस्क पर मॉडल को सेव करना (config में दिए गए पाथ पर)
            logger.info(f"Saving trained model locally to {config.model_path}")
            save_object(
                config.model_path,
                model
            )

            print(f"Model Training Completed! Training Accuracy: {accuracy}")


if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.train_model()