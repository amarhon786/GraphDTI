# CPU inference image for GraphDTI (SageMaker-compatible).
# For GPU inference, base off pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime
# and add the matching torch-geometric extras wheels.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

# RDKit + native deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential libxrender1 libxext6 libsm6 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/graphdti

# Install python deps first so we can cache the layer
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --upgrade pip && pip install .

# Model artefact goes here at runtime — SageMaker mounts to /opt/ml/model.
ENV GRAPHDTI_CKPT=/opt/ml/model/dti.pt \
    GRAPHDTI_HOST=0.0.0.0 \
    GRAPHDTI_PORT=8080

EXPOSE 8080

# SageMaker invokes `serve` with no args.
COPY deploy/serve /usr/local/bin/serve
RUN chmod +x /usr/local/bin/serve

CMD ["serve"]
