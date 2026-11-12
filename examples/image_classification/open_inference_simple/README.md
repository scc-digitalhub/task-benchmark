# OpenInference image classification example

This folder contains OpenInference servers for ViT and timm ResNet models, plus
the benchmark client that calls them.

The default server model is `nateraw/vit-base-patch16-224-cifar10`, fine-tuned
for CIFAR-10. The endpoint alias remains `vit-base-patch16-224`.

## Start the server from this repository

```bash
docker compose -f examples/image_classification/open_inference_simple/docker-compose.yml up --build
```

This starts both servers:

- ViT: http://localhost:9090, model endpoint
  `/v2/models/vit-base-patch16-224/infer`
- ResNet: http://localhost:9091, model endpoint
  `/v2/models/resnet18_cifar10/infer`

The first `--build` installs the server dependencies. Models are downloaded
when their containers first start, so startup may take several minutes.
Docker images and model weights are cached; later starts should be faster.

The Compose setup disables the optional Xet transfer backend because regular
Hugging Face downloads are more reliable in this local Docker setup.

Wait for both server startup messages before sending requests. To start only
one server, append its service name, `open-inference-vit` or
`open-inference-resnet`, to the `docker compose up` command.

The ResNet image uses the `detectors` package to register the custom timm model
and persists its downloaded weights in the `open_inference_resnet_cache`
volume.

## Run the benchmark client

```bash
.venv/bin/python examples/image_classification/open_inference_simple/open_inference_vit.py
```

To benchmark the timm ResNet against CIFAR-10, start its service and run the
dedicated client:

```powershell
docker compose -f examples/image_classification/open_inference_simple/docker-compose.yml up -d --build open-inference-resnet
python -m pip install -e ".[huggingface]"
python examples/image_classification/open_inference_simple/open_inference_resnet.py
```

The ResNet client defaults to
`http://localhost:9091/v2/models/resnet18_cifar10` and the CIFAR-10 `test`
split. It uses the canonical CIFAR-10 class order; set
`OPEN_INFERENCE_LABELS_PATH` if the served model's output indices use a
different mapping. The mapping must reflect the model's actual class order.

You can override the target server/model with environment variables:

```bash
OPEN_INFERENCE_BASE_URL=http://localhost:9090 \
OPEN_INFERENCE_MODEL=google/vit-base-patch16-224 \
OPEN_INFERENCE_MODEL_NAME=vit-base-patch16-224 \
.venv/bin/python examples/image_classification/open_inference_simple/open_inference_vit.py
```

This example uses a local Hugging Face model and exposes the OpenInference v2 HTTP contract directly from this repository, without needing the separate `task-inference` project.

## Hugging Face Dataset Client

`open_inference_vit.py` combines a Hugging Face dataset loader, the numeric
`OpenInferenceInvoker` via its image-classification adapter, and
`Top1ImageClassificationEvaluator`.

Rebuild and recreate the example server to enable the numeric tensor contract:

```powershell
docker compose -f examples/image_classification/open_inference_simple/docker-compose.yml up -d --build
python -m pip install -e ".[huggingface]"
python examples/image_classification/open_inference_simple/open_inference_vit.py
```

Wait for the server log `Serving on http://0.0.0.0:8080` before running the
client. The default model metadata URL is
`http://localhost:8080/v2/models/vit-base-patch16-224`; inference uses its
`/infer` endpoint. Metadata advertises a numeric `pixel_values` input (FP32,
NCHW RGB in `[0, 1]`) and a full `scores` output. The server applies model-specific
resize and normalization, without rescaling values a second time. Existing
base64 `image` requests remain supported as a legacy POST format, but are no
longer the contract advertised by the metadata endpoint.

The client streams up to 50 images from the `test` split of `uoft-cs/cifar10`,
using the `img` and `label` columns. The first run needs network access to fetch
the dataset and server model. The numeric invocation is unchanged.

The server metadata contains the standard model name, inputs, and outputs; it
does not expose model parameters or label mappings. The client loads `id2label`
from its configured model's Hugging Face configuration. For the default
CIFAR-10 pair, the client uses the model's documented class order to replace its generic
`LABEL_0` through `LABEL_9` names: airplane, automobile, bird, cat, deer, dog,
frog, horse, ship, truck. This does not change model weights.
The previously supported ImageNet pair also retains its explicit WordNet-order
mapping. Arbitrary datasets and models are not assumed to share an ordering
just because their class counts match.

The JSON report printed to stdout includes sample count, top-1 accuracy, mean
confidence, mean inference latency (HTTP plus server processing), elapsed loop
time, and per-image predictions, annotations, scores, and latency. It does not
write a report file. Setup and model/dataset construction are excluded from the
elapsed loop time. Confidence is softmax probability, not accuracy.

Configuration is available through environment variables:

| Variable | Default |
| --- | --- |
| `OPEN_INFERENCE_BASE_URL` | `http://localhost:8080/v2/models/vit-base-patch16-224` |
| `HF_DATASET_ID` | `uoft-cs/cifar10` |
| `HF_DATASET_SPLIT` | `test` for CIFAR-10, otherwise `train` |
| `HF_IMAGE_COLUMN` | `img` for CIFAR-10, otherwise `image` |
| `HF_LABEL_COLUMN` | `label` |
| `HF_CACHE_DIR` | `.cache/huggingface` |
| `HF_MAX_SAMPLES` | `50` |
| `OPEN_INFERENCE_MODEL` | `nateraw/vit-base-patch16-224-cifar10` |
| `OPEN_INFERENCE_LABELS_PATH` | Optional JSON mapping of model indices to dataset labels |

Set `OPEN_INFERENCE_MODEL` on the client to the model served by the endpoint so
it can load that model's label configuration. This setting does not affect the
server. Use `OPEN_INFERENCE_LABELS_PATH` when the model configuration is
unavailable or the desired label mapping differs.

For a custom taxonomy, the mapping file must contain all model class indices
starting at zero, for example `{"0": "cat", "1": "dog"}`. These labels must
match dataset annotations; the mapping changes label interpretation, not the
remote model or its weights. The client fails explicitly when labels cannot
be aligned. The numeric contract accepts one image per request, and the example
server currently exposes HTTP only.

To use the previous ImageNet configuration, select the matching model on the
server and the matching dataset on the client (PowerShell):

```powershell
$env:OPEN_INFERENCE_MODEL = "google/vit-base-patch16-224"
docker compose -f examples/image_classification/open_inference_simple/docker-compose.yml up -d --build
$env:HF_DATASET_ID = "mrm8488/ImageNet1K-val"
$env:HF_DATASET_SPLIT = "train"
$env:HF_IMAGE_COLUMN = "image"
python examples/image_classification/open_inference_simple/open_inference_vit.py
```

For ImageNet, `train` is the published split name of this validation export.
Streaming avoids downloading the full multi-GB dataset in advance. To return
to CIFAR-10, set `OPEN_INFERENCE_MODEL` to
`nateraw/vit-base-patch16-224-cifar10`, recreate the container, and set
`HF_DATASET_ID=uoft-cs/cifar10`, `HF_DATASET_SPLIT=test`, and `HF_IMAGE_COLUMN=img`.
