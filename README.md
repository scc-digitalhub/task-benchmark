# task-benchmark

`task-benchmark` is a lightweight benchmarking library for evaluating ML task implementations behind a unified API.

Current built-in scope:
- Tasks: `image-classification`, more to be added
- Built-in implementation: `task-inference`


## Main entry points

- Library API: `task_benchmark.evaluate`
- DigitalHub adapter: `task_benchmark.evaluate_model`

## Base usage


```python
from task_benchmark import evaluate

report = evaluate(
    dataset_path="/path/to/dataset.csv",
    task="image-classification",
    implementation="task-inference",
    device="cpu",
    task_inputs_dir_path="/path/to/images",
    model_name="microsoft/cvt-13-384",
    batch_size=8,
)
```

### Parameters

Required:
- `dataset_path`: CSV path
- `task`: task name (currently `image-classification`)
- `implementation`: implementation name (for example `task-inference` or a custom one)

Optional common:
- `device`: default `"cpu"`
- `report_path`: save JSON report to disk

Task/implementation-specific kwargs (image classification):
- `task_inputs_dir_path`: directory containing images/files referenced by dataset
- `model_name`: model id for implementations that need it (required by `task-inference`)
- `batch_size`: batch size (default `8`)
- `implementation_import_path`: optional module path to import before lookup (used to trigger self-registration for external implementations)

## Dataset format (image-classification)

Expected CSV columns:
- `image_path`
- `wnid`
- `class_description`

Example row:

```csv
image_path,wnid,class_description
img1.bin,n01440764,tench
```

## Report output

The report returned by `evaluate(...)` is a dictionary that includes:
- task metadata: `task`, `implementation`, `device`
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
    dataset_path="/path/to/dataset.csv",
    task="image-classification",
    implementation="my-custom",
    task_inputs_dir_path="/path/to/images",
    implementation_import_path="my_package.my_module",
    device="cpu",
)
```

## DigitalHub usage

`evaluate_model(...)` wraps the same core logic for DigitalHub runtime.

Core-like explicit adapter parameters:
- `dataset`
- `task`
- `implementation`
- `device`
- `report_path`

Task-specific / optional parameters are forwarded through `**kwargs` exactly like `evaluate(...)`.
Examples: `task_inputs_dir_path`, `model_name`, `batch_size`, `implementation_import_path`.

Example:

```python
run = eval_fn.run(
    action="job",
    inputs={
        "dataset": dataset_dataitem.key,
    },
    parameters={
        "task": "image-classification",
        "implementation": "task-inference",
        "device": "cuda",
        "task_inputs_dir_path": "/path/available/in/runtime/images",
        "model_name": "microsoft/cvt-13-384",
        "batch_size": 128,
    },
    profile="1xV100",
    wait=True,
)
```

## License

Apache-2.0
