import os
import json
import tarfile
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

MODEL_DIR = "/opt/ml/processing/model"
TEST_DIR = "/opt/ml/processing/test"
OUTPUT_DIR = "/opt/ml/processing/evaluation"


def preprocess(df):
    df = df.drop("customerID", axis=1)
    df["TotalCharge"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    
    le = LabelEncoder()
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    categorical_cols = [c for c in categorical_cols if c!="Churn"]

    for col in categorical_cols:
        df[col] = le.fit_transform(df[col].astype(str))

    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    return df

    if __name__ == "__main__":
        logger.info("Starting Evaluation")

        model_tar_path = os.path.join(MODEL_DIR, "model.tar.gz")
        if os.path.exists(model_tar_path):
            with tarfile.open(model_tar_path) as tar:
                tar.extractall(path=MODEL_DIR)
            logger.info("Extracted model.tar.gz")
        
        model_path = os.path.join(MODEL_DIR, "model.joblib")
        model = joblib.load(model_path)
        logger.info(f"Loaded model from {model_path}")
        
        test_path = os.path.join(TEST_DIR, "Customer-Churn.csv")
        df = pd.read_csv(test_path)
        df = preprocess(df)

        X = df.drop("Churn", axis=1)
        y = df["Churn"]

        preds = model.predict(X)
        f1 = f1_score(y,preds)
        logger.info(f"Evaluation F1 Score: {f1:.4f}")

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        report = {"f1_score": f1 }
        report_path = os.path.join(OUTPUT_DIR, "evaluation.json")
        with open(report_path, "w") as f:
            json.dump(report, f)

        logger.info(f"Evaluation report saved to {report_path}") 