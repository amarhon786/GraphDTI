"""End-to-end SageMaker deployment helper.

Steps:
1. Build the Docker image from `deploy/Dockerfile`.
2. Push it to ECR (creates the repo if missing).
3. Upload `checkpoints/dti.pt` to S3 inside a `model.tar.gz`.
4. Create a SageMaker `Model`, `EndpointConfig`, and `Endpoint`.

Run:
    python deploy/sagemaker_deploy.py \\
        --image-name graphdti \\
        --ckpt checkpoints/dti.pt \\
        --bucket my-s3-bucket \\
        --role-arn arn:aws:iam::123456789012:role/SageMakerExecutionRole \\
        --endpoint-name graphdti-prod \\
        --instance-type ml.c5.large
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path


def _run(cmd: list[str]) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def build_and_push_image(image_name: str, tag: str, region: str) -> str:
    try:
        import boto3
    except ImportError:
        sys.exit("Install boto3: pip install '.[deploy]'")

    sts = boto3.client("sts")
    account = sts.get_caller_identity()["Account"]
    repo = f"{account}.dkr.ecr.{region}.amazonaws.com/{image_name}"
    uri = f"{repo}:{tag}"

    ecr = boto3.client("ecr", region_name=region)
    try:
        ecr.create_repository(repositoryName=image_name)
    except ecr.exceptions.RepositoryAlreadyExistsException:
        pass

    auth = ecr.get_authorization_token()["authorizationData"][0]
    import base64

    user_pwd = base64.b64decode(auth["authorizationToken"]).decode().split(":", 1)
    _run(["docker", "login", "-u", user_pwd[0], "-p", user_pwd[1], auth["proxyEndpoint"]])
    _run(["docker", "build", "-f", "deploy/Dockerfile", "-t", uri, "."])
    _run(["docker", "push", uri])
    return uri


def upload_model(ckpt_path: Path, bucket: str, key_prefix: str) -> str:
    try:
        import boto3
    except ImportError:
        sys.exit("Install boto3: pip install '.[deploy]'")

    with tempfile.TemporaryDirectory() as tmp:
        tar_path = Path(tmp) / "model.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(ckpt_path, arcname="dti.pt")
        key = f"{key_prefix.rstrip('/')}/model.tar.gz"
        boto3.client("s3").upload_file(str(tar_path), bucket, key)
    s3_uri = f"s3://{bucket}/{key}"
    print(f"[s3] uploaded {s3_uri}")
    return s3_uri


def create_endpoint(image_uri: str, model_data: str, role_arn: str, endpoint_name: str, instance_type: str, region: str):
    try:
        import boto3
    except ImportError:
        sys.exit("Install boto3: pip install '.[deploy]'")
    sm = boto3.client("sagemaker", region_name=region)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    model_name = f"{endpoint_name}-model-{stamp}"
    config_name = f"{endpoint_name}-cfg-{stamp}"

    sm.create_model(
        ModelName=model_name,
        PrimaryContainer={"Image": image_uri, "ModelDataUrl": model_data, "Mode": "SingleModel"},
        ExecutionRoleArn=role_arn,
    )
    sm.create_endpoint_config(
        EndpointConfigName=config_name,
        ProductionVariants=[
            {
                "VariantName": "primary",
                "ModelName": model_name,
                "InitialInstanceCount": 1,
                "InstanceType": instance_type,
            }
        ],
    )
    existing = sm.list_endpoints(NameContains=endpoint_name)["Endpoints"]
    if any(e["EndpointName"] == endpoint_name for e in existing):
        sm.update_endpoint(EndpointName=endpoint_name, EndpointConfigName=config_name)
        print(f"[sagemaker] updating endpoint {endpoint_name}")
    else:
        sm.create_endpoint(EndpointName=endpoint_name, EndpointConfigName=config_name)
        print(f"[sagemaker] creating endpoint {endpoint_name}")
    print("[sagemaker] track status with: aws sagemaker describe-endpoint --endpoint-name", endpoint_name)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image-name", default="graphdti")
    p.add_argument("--tag", default="latest")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--bucket", required=True)
    p.add_argument("--key-prefix", default="graphdti/models")
    p.add_argument("--role-arn", required=True)
    p.add_argument("--endpoint-name", default="graphdti")
    p.add_argument("--instance-type", default="ml.c5.large")
    p.add_argument("--region", default="us-east-1")
    p.add_argument("--skip-build", action="store_true")
    args = p.parse_args()

    image_uri = (
        f"{args.image_name}:{args.tag}"
        if args.skip_build
        else build_and_push_image(args.image_name, args.tag, args.region)
    )
    model_data = upload_model(Path(args.ckpt), args.bucket, args.key_prefix)
    create_endpoint(image_uri, model_data, args.role_arn, args.endpoint_name, args.instance_type, args.region)


if __name__ == "__main__":
    main()
