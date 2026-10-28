# Image Classification Examples

Run the built-in `task-inference` integration with temporary PNG inputs:

```sh
.venv/bin/python examples/image_classification/simple_task_inference.py
```

Run the Hugging Face dataset and Transformers example (installs the optional
`datasets` dependency if needed):

```sh
python -m pip install -e ".[huggingface]"
python examples/image_classification/huggingface/huggingface_image_classification.py
```

By default this loads five samples from CIFAR-10 and runs a CIFAR-10 fine-tuned
ViT model. Override `HF_DATASET_ID`, `HF_MODEL_ID`, `HF_DATASET_SPLIT`,
`HF_CACHE_DIR`, `HF_MAX_SAMPLES`, or `DEVICE` to change the dataset, model, cache,
sample limit, or execution device.

Run the self-contained OpenInference implementation (default endpoint:
`http://localhost:8080`):

```sh
docker compose -f examples/image_classification/open_inference_simple/docker-compose.yml up --build
```

Wait for `Serving on http://0.0.0.0:8080`, then run the client in another terminal:

```sh
.venv/bin/python examples/image_classification/open_inference_simple/open_inference.py
```

Use `OPEN_INFERENCE_BASE_URL` and `OPEN_INFERENCE_MODEL` to target another
server or model:

```sh
OPEN_INFERENCE_BASE_URL=http://localhost:8080 \
OPEN_INFERENCE_MODEL=google/vit-base-patch16-224 \
.venv/bin/python examples/image_classification/open_inference_simple/open_inference.py
```

For a real dataset run, see [the OpenInference ImageNet-1000 example](open_inference_imagenet/README.md).

`always_first_class.py` defines and registers an in-process baseline that predicts the alphabetically first label. `most_common.py` loads the self-registering baseline in `my_custom_model/` through `implementation_import_path`.

```sh
.venv/bin/python examples/image_classification/always_first_class.py
.venv/bin/python examples/image_classification/most_common.py
```

See [ImageNet-1000 workflows](imagenet-1000/README.md) for Tiny-ImageNet runs and [the custom model](my_custom_model/README.md) for the imported implementation.
