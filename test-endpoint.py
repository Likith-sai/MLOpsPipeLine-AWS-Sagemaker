import boto3
import pandas as pd
import json
from config import REGION

runtime = boto3.client("sagemaker-runtime", region_name = REGION)

sample = [[0, 1, 0, 0, 12, 1, 0, 1, 2, 1, 0, 1, 0, 1, 1, 0, 29.85, 29.85]]
payload = json.dumps(sample)

response = runtime.invoke_endpoint(
    EndpointName = "churn-classifier-endpoint",
    ContentType = "application/json",
    Body = payload
)

result = json.loads(response["Body"].read().decode())
print(f"Prediction: {result}")