# task-benchmark

`task-benchmark` is a lightweight benchmarking library for evaluating ML task implementations behind a unified API.

Current built-in scope:
- Tasks: `image-classification`, more to be added
- Built-in implementation: `task-inference`



## Base usage


```python
from task_benchmark import evaluate
from task_benchmark.tasks.image_classification.task import ImageClassificationDataObject

data_object = ImageClassificationDataObject(
    images_path=["/path/to/img1.jpg", "/path/to/img2.jpg"],
    labels=["tench", "goldfish"],
)

report = evaluate(
    task="image-classification",
    implementation="my-custom",
    data_object=data_object,
    implementation_import_path="my_package.my_module",
    device="cpu",
)
```

### Parameters

Required:
- `task`: task name (currently `image-classification`)
- `implementation`: implementation name (for example `task-inference` or a custom one)
- `data_object`: task-specific input data object

Optional common:
- `profile`: default `"default"`
- `report_path`: save JSON report to disk
- `device`: execution device (for example `cpu` or `cuda`)

Task/implementation-specific kwargs (image classification):
- `model_name`: model id for implementations that need it (required by `task-inference`)
- `batch_size`: batch size (default `8`)
- `implementation_import_path`: optional module path to import before lookup (used to trigger self-registration for external implementations)

## Input format (image-classification)

The image-classification task expects a data object with:
- `images_path`: list of image file paths
- `labels`: list of ground-truth labels (same length as `images_path`)

Example:

```python
ImageClassificationDataObject(
    images_path=["/data/images/img1.jpg", "/data/images/img2.jpg"],
    labels=["tench", "goldfish"],
)
```

## Report output

The report returned by `evaluate(...)` is a dictionary that includes:
- task metadata: `task`, `implementation`, `profile`
- task metrics: `top1_accuracy`, `top5_accuracy`, `dataset_images_evaluated`, `dataset_images_skipped`
- runtime metrics: wall/cpu time, memory, per-batch aggregates


## Writing a custom implementation

A custom implementation should:
1. subclass the task's class
2. implement necessary methods for the task
3. register itself with `implementation_registry.register(...)`

Minimal skeleton for image-classification task:

```python
from task_benchmark.implementations import implementation_registry
from task_benchmark.tasks.image_classification import ImageClassificationModel, Prediction

class MyClassifier(ImageClassificationModel):
    predicts_wnid = True

    def __init__(self, model_name: str = "", device: str = "cpu", class_descriptions: dict[str, str] | None = None):
        ...

    def predict_batch(self, inputs: list[bytes], top_k: int = 5) -> list[list[Prediction]]:
        ...

implementation_registry.register(
    task="image-classification",
    implementation="my-custom",
    implementation_cls=MyClassifier,
)
```

Then call:

```python
report = evaluate(
    task="image-classification",
    implementation="my-custom",
    data_object=data_object,
    implementation_import_path="my_package.my_module",
    device="cpu",
)
```

## DigitalHub usage

Use the standalone example at `examples/imagenet-1000/tiny_imagenet_1000_digitalhub.py`.

The example contains a `@handler` entrypoint that:
- accepts input artifact(s) from DigitalHub
- resolves/downloads them in runtime
- builds the task data object
- runs `evaluate(...)`
- logs the evaluation report artifact

## License

Apache-2.0
