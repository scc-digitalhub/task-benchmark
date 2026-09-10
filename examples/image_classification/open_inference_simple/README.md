# OpenInference image classification example

This folder contains a self-contained OpenInference server plus the benchmark client that calls it.

## Start the server from this repository

```bash
docker compose -f examples/image_classification/open_inference_simple/docker-compose.yml up --build
```

This starts a local HTTP server on:

- http://localhost:8080
- model endpoint: /v2/models/google%2Fvit-base-patch16-224/infer

The first `--build` installs pinned CPU-compatible Torch and Transformers
packages into the server image and downloads the Hugging Face model, so it can
take several minutes. The Docker image and model weights are cached; later
starts should be much faster.

The Compose setup disables the optional Xet transfer backend because regular
Hugging Face downloads are more reliable in this local Docker setup.

Wait for `Serving on http://0.0.0.0:8080` before running the benchmark client.

## Run the benchmark client

```bash
.venv/bin/python examples/image_classification/open_inference_simple/open_inference.py
```

You can override the target server/model with environment variables:

```bash
OPEN_INFERENCE_BASE_URL=http://localhost:8080 \
OPEN_INFERENCE_MODEL=google/vit-base-patch16-224 \
.venv/bin/python examples/image_classification/open_inference_simple/open_inference.py
```

This example uses a local Hugging Face model and exposes the OpenInference v2 HTTP contract directly from this repository, without needing the separate `task-inference` project.
