# OpenInference ImageNet-1000 example

This example evaluates the OpenInference implementation on 1,000 validation
images from the Tiny ImageNet dataset used by the existing ImageNet examples.
The folder is self-contained and includes its own `server.py`, `Dockerfile`,
and Docker Compose configuration.

## Start the local OpenInference server

From the `task-benchmark` repository root:

```bash
docker compose -f examples/image_classification/open_inference_imagenet/docker-compose.yml up --build
```

The server listens on `http://localhost:8080`. The first build installs the
server dependencies, and the first startup downloads the model weights. Both
the Docker image and model cache are reused on later runs.

Wait for the server to print `Serving on http://0.0.0.0:8080` before starting
the benchmark client.

## Run the benchmark

In a second terminal, from the repository root:

```bash
.venv/bin/python examples/image_classification/open_inference_imagenet/open_inference_imagenet.py
```

The script downloads the dataset through `kagglehub`, reads the validation
annotations and labels from `words.txt`, evaluates 1,000 images in batches,
and writes `report_open_inference.json` in this directory.

## Configuration

Override the defaults with environment variables:

```bash
OPEN_INFERENCE_MODEL=google/vit-base-patch16-224 \
OPEN_INFERENCE_BASE_URL=http://localhost:8080 \
OPEN_INFERENCE_DATASET_SIZE=1000 \
OPEN_INFERENCE_BATCH_SIZE=16 \
.venv/bin/python examples/image_classification/open_inference_imagenet/open_inference_imagenet.py
```

Only one server example should use port `8080` at a time.
