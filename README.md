# SageMaker-AWS

A simple AWS SageMaker helper repository for training, deploying, evaluating, and managing a model using S3 storage.

## Project structure

- `pip install -r requirements.txt` – Install the necessary packages.
- `config.py` – central configuration for AWS region, account, role, bucket, and S3 prefixes.
- `train.py` – training script.
- `deploy.py` – deploys a SageMaker endpoint.
- `evaluate.py` – evaluates the deployed model.
- `pipeline.py` – pipeline orchestration that runs training, evaluation, and model registration.
- `test-endpoint.py` – tests the deployed endpoint.
- `delete-endpoint.py` – deletes the SageMaker endpoint.

## Pipeline workflow

1. `python pipeline.py`
    Triggers SageMaker Pipeline execution
      Step 1: `TrainChurnModel` (runs `train.py`)
      Step 2: `EvaluateChurnModel` (runs `evaluate.py`)
      Step 3: `CheckF1Score` → if passed, registers model

2. Verify pipeline succeeded
     Check SageMaker console → Pipelines → `ChurnPredictionPipeline`
     Confirm model appears in Model Registry, `approval_status = PendingManualApproval`

3. Manually approve the model in registry (or do this via script)
     Required before `deploy.py` can find an "Approved" model

4. `python deploy.py`
     Deploys the approved model to a real-time endpoint

5. `python test_endpoint.py`
     Sends sample data, confirms predictions return correctly

6. Delete the endpoint immediately
   ```bash
   python delete-endpoint.py
   ```

Models can approve through CLI also instead of using UI:

```bash
aws sagemaker list-model-packages \
  --model-package-group-name ChurnModelPackageGroup \
  --query "ModelPackageSummaryList[0].ModelPackageArn" \
  --output text

aws sagemaker update-model-package \
  --model-package-arn <paste-arn-here> \
  --model-approval-status Approved
```

## Configuration

Open `config.py` and set the following values:

- `REGION` – AWS region
- `ACCOUNT_ID` – AWS account ID
- `ROLE_ARN` – SageMaker execution role ARN
- `BUCKET` – S3 bucket name
- `PREFIX` – S3 prefix for SageMaker assets

Example values:

```python
REGION = "us-west-2"
ACCOUNT_ID = "123456789012"
ROLE_ARN = "arn:aws:iam::123456789012:role/SageMakerExecutionRole"
BUCKET = "my-unique-sagemaker-bucket"
PREFIX = "sagemaker"
```

## S3 upload

Upload your local CSV data file to the configured bucket location before training:

```bash
aws s3 cp /path/to/file.csv s3://<UNIQUE_BUCKET_NAME>/sagemaker/data/
```

If you want to use the variables from `config.py` directly in a shell, export them first:

```bash
export BUCKET=<UNIQUE_BUCKET_NAME>
export PREFIX=sagemaker
aws s3 cp /path/to/file.csv s3://$BUCKET/$PREFIX/data/
```

The repository uses these S3 paths:

- Data URI: `s3://<BUCKET>/sagemaker/data`
- Model URI: `s3://<BUCKET>/sagemaker/models`

## Notes
- Ensure AWS CLI is configured with credentials that can access SageMaker and the target S3 bucket.
- Replace `<UNIQUE_BUCKET_NAME>` with the actual bucket name in your `config.py`.
