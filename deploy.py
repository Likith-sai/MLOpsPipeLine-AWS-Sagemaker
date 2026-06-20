import boto3
import sagemaker
from sagemaker.sklearn import SKLearnModel
from config import ROLE_ARN, REGION,BUCKET, PREFIX

session = sagemaker.Session(default_bucket=BUCKET)
sm_client = boto3.client("sagemaker", region_name=REGION)

response = sm_client.list_model_packages(
    ModelPackageGroupName = "ChurnModelPackageGroup",
    ModelApprovalStatus = "Approved",
    SortBy = "CreationTime",
    SortOrder = "Descending"
)

latest_model_arn = response["ModelPackageSummaryList"][0]["ModelPackageArn"]
print(f"Deploying Model: {latest_model_arn}")

model = sagemaker.ModelPackage(
    role=ROLE_ARN,
    model_package_arn=latest_model_arn,
    sagemaker_session = session
)

predictor = model.deploy(
    initial_instance_count = 1,
    instance_type = "ml.m5.large",
    endpoint_name = "churn-classifier-endpoint"
)

print(f"Endpoint deployed: churn-classifier-endpoint")
print(f"Test with: predictor.predict(data)")