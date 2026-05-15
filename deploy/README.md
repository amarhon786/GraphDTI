# Deployment

## Local Docker

```bash
docker build -f deploy/Dockerfile -t graphdti:local .
docker run --rm -p 8080:8080 \
    -v "$(pwd)/checkpoints:/opt/ml/model" \
    graphdti:local
# POST http://localhost:8080/predict
```

## AWS SageMaker

Prereqs: `pip install ".[deploy]"`, Docker, AWS credentials with ECR + SageMaker + S3 access, an existing SageMaker execution role.

```bash
python deploy/sagemaker_deploy.py \
    --image-name graphdti \
    --ckpt checkpoints/dti.pt \
    --bucket my-bucket \
    --role-arn arn:aws:iam::ACCOUNT:role/SageMakerExecutionRole \
    --endpoint-name graphdti-prod \
    --instance-type ml.c5.large \
    --region us-east-1
```

The script builds the Docker image, pushes to ECR, tars + uploads the
checkpoint to S3, and creates (or updates) the SageMaker endpoint.

Invoke the endpoint:

```python
import json, boto3
runtime = boto3.client("sagemaker-runtime", region_name="us-east-1")
resp = runtime.invoke_endpoint(
    EndpointName="graphdti-prod",
    ContentType="application/json",
    Body=json.dumps({"smiles": "CCO", "protein_sequence": "MAVKR..."}),
)
print(json.loads(resp["Body"].read()))
```

For SHAP-on-graph attributions, add `"explain": true` to the payload.
