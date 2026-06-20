import os
import argparse
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
import joblib
import json
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def preprocess(df):
    df = df.drop("customerID", axis=1)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)

    le=LabelEncoder()
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    categorical_cols = [c for c in categorical_cols if c!="Churn"]

    for col in categorical_cols:
        df[col] = le.fit_transform(df[col].astype(str))

    df["Churn"] = df["Churn"].map({"Yes": 1, "No":0})
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"))
    args = parser.parse_args()

    train_path = os.path.join(args.train, "Customer-Churn.csv")

    df = pd.read_csv(train_path)
    df = preprocess(df)

    x = df.drop("Churn", axis=1)
    y = df["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=42
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    f1 = f1_score(y_test, preds)
    logger.info(f"f1_score: {f1:.4f}")

    metrics = {"f1_score: f1"}
    metrics_path = os.path.join(args.model_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f)


    model_path = os.path.join(args.model_dir, "model.joblib")
    joblib.dump(model, model_path)
    logger.info(f"Model saved to {model_path}")