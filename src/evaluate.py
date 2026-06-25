import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.utils import load_object
from src.logger import logger
import mlflow

class ModelEvaluation:

    def evaluate(self):
        # 1. Feature Engineering से बनी test.csv को लोड करें
        test_data_path = "artifacts/test.csv"
        logger.info(f"Loading test data from {test_data_path}")
        df = pd.read_csv(test_data_path)

        # 2. X_test और y_test अलग करें
        X_test = df.drop("Churn", axis=1)
        y_test = df["Churn"]

        # 3. ट्रेन किया हुआ मॉडल लोड करें
        model_path = "models/model.pkl" 
        logger.info(f"Loading model from {model_path}")
        model = load_object(model_path)

        # 4. प्रेडिक्शन और मैट्रिक्स कैलकुलेशन
        logger.info("Making predictions on test data...")
        pred = model.predict(X_test)

        acc = accuracy_score(y_test, pred)
        prec = precision_score(y_test, pred, zero_division=0)
        rec = recall_score(y_test, pred, zero_division=0)
        f1 = f1_score(y_test, pred, zero_division=0)

        # 5. कंसोल पर प्रिंट करें
        print("\n================ MODEL EVALUATION METRICS ================")
        print(f"Accuracy  : {acc:.4f}")
        print(f"Precision : {prec:.4f}")
        print(f"Recall    : {rec:.4f}")
        print(f"F1 Score  : {f1:.4f}")
        print("==========================================================\n")

        # 6. MLflow के लास्ट रन को खोजकर उसी में टेस्ट मैट्रिक्स लॉग करना
        try:
            logger.info("Connecting to the latest MLflow run to log test metrics...")
            
            # सबसे रीसेंट एक्सपेरिमेंट का लास्ट रन ढूंढें
            latest_run = mlflow.search_runs(order_by=["start_time DESC"], max_results=1).iloc[0]
            run_id = latest_run.run_id
            
            # उस रन ID के अंदर मैट्रिक्स लॉग करें
            with mlflow.start_run(run_id=run_id):
                mlflow.log_metric("test_accuracy", acc)
                mlflow.log_metric("test_precision", prec)
                mlflow.log_metric("test_recall", rec)
                mlflow.log_metric("test_f1", f1)
            
            logger.info(f"Successfully logged test metrics to MLflow Run ID: {run_id}")
            
        except Exception as e:
            logger.warning(f"Could not log to MLflow automatically: {e}. Running without MLflow logging.")

if __name__ == "__main__":
    evaluator = ModelEvaluation()
    evaluator.evaluate()