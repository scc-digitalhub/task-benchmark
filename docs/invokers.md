# Model Invokers

## Generic `Invoker`

`Invoker[InputT, OutputT]` is the abstract interface for model inference. An
implementation defines `invoke()` for its input and output types:

```python
from task_benchmark.implementations.invoker import Invoker


class MyInvoker(Invoker[InputType, OutputType]):
    def invoke(self, test_data: InputType) -> OutputType:
        ...
```

The current implementations are `HuggingFaceImageClassificationInvoker` for
local Transformers inference and `OpenInferenceInvoker` for remote Open
Inference v2 services.

## Hugging Face Image Classification

`HuggingFaceImageClassificationInvoker` loads an image processor and
classification model using Transformers `from_pretrained`. `model_name` can be
a Hub model ID or a local model directory. The processor applies the
preprocessing configured by the model; the invoker returns the model logits as
a `torch.Tensor`, without applying softmax.

```python
import torch

from task_benchmark.implementations.invoker.image_classification import (
    HuggingFaceImageClassificationInvoker,
)

invoker = HuggingFaceImageClassificationInvoker(
    model_name="nateraw/vit-base-patch16-224-cifar10",
    device="cpu",
)

image = torch.rand(3, 32, 32)  # CHW, RGB, values in [0, 1]
logits = invoker.invoke(image)
predicted_id = int(logits[0].argmax().item())
predicted_label = invoker.label_for(predicted_id)
```

`invoke()` accepts either a single CHW image tensor or a BCHW batch. It moves
the input to CPU for processor preprocessing, then moves the resulting model
inputs to the configured device. The model is loaded in evaluation mode and
inference runs without gradient tracking.

The optional `model_kwargs` and `processor_kwargs` dictionaries are passed to
the corresponding `from_pretrained()` methods. For example, use them to set a
revision or a cache directory:

```python
invoker = HuggingFaceImageClassificationInvoker(
    model_name="nateraw/vit-base-patch16-224-cifar10",
    device="cuda",
    model_kwargs={"revision": "main", "cache_dir": "./.cache/huggingface"},
    processor_kwargs={"revision": "main", "cache_dir": "./.cache/huggingface"},
)
```

`label_for(class_id)` looks up the class name in the model's `id2label`
configuration. If the ID has no configured label, it returns the ID as a string.
The class mapping can therefore be used without reaching into
`invoker.model.config` directly.

The `transformers` package is a project dependency. Loading a Hub model may
download model files; use a local model path or configure the cache arguments
when needed.

## Open Inference v2

`OpenInferenceInvoker` invokes a model over HTTP or gRPC. Its `service_url` must
include the host, model name, and full inference path:

```text
http://localhost:8000/v2/models/my-model/infer
grpc://localhost:8001/v2/models/my-model/infer
```

`https` and `grpcs` are also supported. URL-encode special characters in the
model name, for example `/` as `%2F`.

For a model with one input, pass one tensor. The first output declared in the
model metadata is returned:

```python
import torch

from task_benchmark.implementations.invoker.open_inference import (
    OpenInferenceInvoker,
)

invoker = OpenInferenceInvoker(
    "http://localhost:8000/v2/models/my-model/infer"
)
try:
    logits = invoker.invoke(torch.rand(1, 3, 224, 224))
finally:
    invoker.close()
```

For models with multiple inputs or outputs, pass a mapping from input names to
tensors. Names must exactly match the model metadata; the returned dictionary
maps output names to tensors:

```python
inputs = {
    "image": torch.rand(1, 3, 224, 224),
    "threshold": torch.tensor([0.5]),
}
outputs = invoker.invoke(inputs)
scores = outputs["scores"]
embedding = outputs["embedding"]
```

Open Inference datatypes `BOOL`, `UINT8`, `INT8`, `INT16`, `INT32`, `INT64`,
`FP16`, `FP32`, and `FP64` are supported. Inputs are moved to CPU and converted
to the datatype declared by model metadata. Model-specific resizing and
normalization remain the caller's responsibility.

HTTP uses Python's standard-library `urllib`. gRPC requires the
`tritonclient[grpc]` project dependency. `close()` closes the gRPC client and
has no effect for HTTP.