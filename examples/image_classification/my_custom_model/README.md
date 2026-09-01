# Most Common Class Model

`most_common.py` is an importable, self-registering implementation. It predicts the most frequent label in the evaluation data and is exercised by the parent example.

From the repository root:

```sh
.venv/bin/python examples/image_classification/most_common.py
```

Use its import path with `evaluate` when loading it elsewhere:

```python
implementation="most-common-class"
implementation_import_path="my_custom_model.most_common"
```
