import boto3
import sagemaker
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import TrainingStep, ProcessingStep
from sagemaker.workflow.parameters import ParameterFloat, ParameterInteger
from sagemaker.sklearn import SKLearn
from sagemaker.processing import ScriptProcessor,ProcessingInput,ProcessingOutput
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.functions import JsonGet
from sagemaker.workflow.properties import PropertyFile
from sagemaker import image_uris
from config import REGION, ROLE_ARN, BUCKET, PREFIX

session = sagemaker.Session(default_bucket=BUCKET)

n_estimators = ParameterInteger(name="NEstimators", default_value=100)
max_depth = ParameterInteger(name="MaxDepth", default_value=5)
f1_threshold = ParameterFloat(name="F1Threshold", default_value=0.50)


# STEP 1: Training
sklearn_estimator = SKLearn(
    entry_point="train.py",
    role = ROLE_ARN,
    instance_type = "ml.m5.large",
    framework_version="1.2-1",
    py_version="py3",
    hyperparameters={
        "n-estimators" : n_estimators,
        "max-depth": max_depth
    },
    output_path = f"s3://{BUCKET}/{PREFIX}/output"
)

step_train = TrainingStep(
    name = "TrainChurnModel",
    estimator=sklearn_estimator,
    inputs={
        "train" : sagemaker.inputs.TrainingInput(
            s3_data=f"s3://{BUCKET}/sagemaker/data",
            content_type ="text/csv"
        )
    }
)

evaluation_report = PropertyFile(
    name = "EvaluationReport",
    output_name="evaluation",
    path="evaluation.json"
)

# STEP 2: Evaluation
sklearn_image_uri = image_uris.retrieve(
    framework="sklearn",
    region=REGION,
    version="1.2-1",
    py_version="py3",
    instance_type = "ml.m5.large"
)

script_processor = ScriptProcessor(
    image_uri=sklearn_image_uri,
    command=["python3"],
    instance_type = "ml.m5.large",
    instance_count=1,
    role=ROLE_ARN
)

step_evaluate = ProcessingStep(
    name="EvaluateChurnModel",
    processor=script_processor,
    inputs=[
        ProcessingInput(
            source=step_train.properties.ModelArtifacts.S3ModelArtifacts,
            destination="/opt/ml/processing/model"
        ),
        ProcessingInput(
            source=f"s3://{BUCKET}/sagemaker/data/",
            destination="/opt/ml/processing/test"
        )
    ],
    outputs=[
        ProcessingOutput(
            output_name="evaluation",
            source="/opt/ml/processing/evaluation"
        )
    ],
    code="evaluate.py",
    property_files=[evaluation_report]
)

# STEP 3 : Conditional Registration
step_register = step_train.register(
    content_types = ["text/csv"],
    response_types = ["application/json"],
    inference_instances=["ml.m5.large"],
    transform_instances = ["ml.m5.large"],
    model_package_group_name = "ChurnModelPackageGroup",
    approval_status = "PendingManualApproval"
)

step_condition = ConditionStep(
    name="CheckF1Score",
    conditions=[
        ConditionGreaterThanOrEqualTo(
            left = JsonGet(
                step_name=step_evaluate.name,
                property_file=evaluation_report,
                json_path="f1_score"
            ),
            right=f1_threshold
        )
    ],
    if_steps=[step_register],
    else_steps=[]
)

# Build and Run Pipeline
pipeline = Pipeline(
    name="ChurnPredictionPipeline",
    parameters=[n_estimators, max_depth, f1_threshold],
    steps=[step_train, step_evaluate, step_condition]
)

pipeline.upsert(role_arn=ROLE_ARN)
execution = pipeline.start()
print(f"Pipeline execution started: {execution.arn}")
execution.wait()
print("Pipeline Completed!")