# Image Classification Examples

Run the built-in `task-inference` integration with temporary PNG inputs:

```sh
.venv/bin/python examples/image_classification/simple_task_inference.py
```

`always_first_class.py` defines and registers an in-process baseline that predicts the alphabetically first label. `most_common.py` loads the self-registering baseline in `my_custom_model/` through `implementation_import_path`.

```sh
.venv/bin/python examples/image_classification/always_first_class.py
.venv/bin/python examples/image_classification/most_common.py
```

See [ImageNet-1000 workflows](imagenet-1000/README.md) for Tiny-ImageNet runs and [the custom model](my_custom_model/README.md) for the imported implementation.
