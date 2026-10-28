# DataLoader and Image Classification Datasets

## DataLoader

`DataLoader` is the abstract base class for dataset loader implementations. Its
constructor is:

```python
DataLoader(dataset_id: str, local_path: str, params: dict[str, Any] | None = None)
```

The class stores `dataset_id`, `local_path`, and `params`. The latter is
optional; custom implementations can use it for implementation-specific
configuration. If omitted, `params` is an empty dictionary.

Subclasses must implement:

```python
next() -> tuple[str, object]
```

The first tuple element uniquely identifies a sample; the second contains the
data to process. The method can raise `StopIteration` when no samples remain.

## ImageClassificationDataLoader

`ImageClassificationDataLoader` reads classes, images, and annotations from a
local directory. The expected structure is:

```text
dataset/
    classes.txt
    images.txt
    annotations.csv
    images/
        cat-001.jpg
        dog-001.jpg
```

`classes.txt` contains one class per line:

```text
cat
dog
pet
```

`images.txt` lists image paths relative to the dataset directory:

```text
images/cat-001.jpg
images/dog-001.jpg
```

`annotations.csv` must have `image` and `labels` columns. The `image` values are
relative paths as in `images.txt`; `labels` contains one or more classes
separated by semicolons:

```csv
image,labels
images/cat-001.jpg,cat;pet
images/dog-001.jpg,dog
```

The loader checks that the files exist, each image has annotations, and all
annotated classes are declared in `classes.txt`.

## Uso

```python
from task_benchmark.implementations.data_loader.image_classification import (
    ImageClassificationDataLoader,
)

loader = ImageClassificationDataLoader(
    dataset_id="validation-set",
    local_path="./dataset",
)

image_id, image_tensor = loader.next()
ground_truth = loader.labels_for(image_id)
```

`image_id` is the relative image path, normalized to use `/` separators.
`image_tensor` is an RGB PyTorch tensor in `CHW` format, with `float32` dtype
and values in the `[0, 1]` range. Annotations are available as a tuple of labels
through `labels_for(image_id)`.

The loader tracks its current position (`current_index`) and also implements the
Python iterator protocol:

```python
for image_id, image_tensor in loader:
    labels = loader.labels_for(image_id)
    # Pass image_tensor to the inference component.
```

When the dataset is exhausted, `next()` and the iterator raise `StopIteration`.
The loader does not resize or normalize images for a specific model; apply
model-specific preprocessing in the inference component or a dedicated
transformation.

## Hugging Face Datasets

`HuggingFaceImageClassificationDataLoader` loads an image-classification split
with the Hugging Face `datasets` library. Install the optional project extra:

```powershell
python -m pip install "task-benchmark[huggingface]"
```

Pass the Hugging Face dataset ID as `dataset_id`. `local_path` is used as the
dataset cache directory. The default split is `test`, and the default image and
label columns are `image` and `label`:

```python
from task_benchmark.implementations.data_loader.image_classification import (
    HuggingFaceImageClassificationDataLoader,
)

loader = HuggingFaceImageClassificationDataLoader(
    dataset_id="zalando-datasets/fashion_mnist",
    local_path="./.cache/huggingface",
    params={
        "split": "test",
        "image_column": "image",
        "label_column": "label",
        "revision": "main",
    },
)

image_id, image_tensor = loader.next()
ground_truth = loader.labels_for(image_id)
```

`params` also accepts `name` for a dataset configuration, `id_column` for a
dataset-provided sample identifier, and `classes` for explicit class names.
Other entries are forwarded to `datasets.load_dataset`. If no `id_column` (or
default `id` column) is available, the loader generates an identifier from the
dataset ID, split, and row index.

When the dataset exposes Hugging Face `ClassLabel` features, numeric labels are
converted to class names and `loader.classes` is populated from the feature.
For string labels without class metadata, encountered class names are added to
`loader.classes` as rows are read. `labels_for(image_id)` is available after the
corresponding row has been returned by `next()`.