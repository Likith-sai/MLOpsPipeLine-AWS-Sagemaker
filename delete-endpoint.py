import boto3
sm = boto3.client("sagemaker", region_name="ap-south-1")
sm.delete_endpoint(EndpointName="churn-classifier-endpoint")
print("Endpoint deleted")