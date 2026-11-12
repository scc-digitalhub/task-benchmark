from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import unquote

import torch
from torch import Tensor, nn


MODEL = os.getenv("OPEN_INFERENCE_MODEL", "resnet18_cifar10")
MODEL_NAME = os.getenv("OPEN_INFERENCE_MODEL_NAME", "resnet18_cifar10")
HOST = os.getenv("OPEN_INFERENCE_HOST", "0.0.0.0")
PORT = int(os.getenv("OPEN_INFERENCE_PORT", "8080"))
DEVICE_NAME = os.getenv("OPEN_INFERENCE_DEVICE", "cpu")

classifier: nn.Module | None = None
device: torch.device | None = None
input_size: tuple[int, int, int] | None = None
mean: Tensor | None = None
std: Tensor | None = None
num_classes: int | None = None


def load_model() -> None:
    global classifier, device, input_size, mean, std, num_classes

    import detectors  # noqa: F401 - registers the custom timm model.
    import timm

    selected_device = torch.device(DEVICE_NAME)
    if selected_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device '{DEVICE_NAME}' is not available")

    print(f"Loading {MODEL}...", flush=True)
    model = timm.create_model(MODEL, pretrained=True)
    data_config = timm.data.resolve_model_data_config(model)
    model_input_size = tuple(int(value) for value in data_config["input_size"])
    if len(model_input_size) != 3 or model_input_size[0] != 3:
        raise ValueError("The model must accept three-channel RGB images")

    model_mean = tuple(float(value) for value in data_config["mean"])
    model_std = tuple(float(value) for value in data_config["std"])
    if len(model_mean) != 3 or len(model_std) != 3 or any(
        value <= 0 for value in model_std
    ):
        raise ValueError("The model must provide three-channel normalization values")

    classifier_head = model.get_classifier()
    classes = getattr(classifier_head, "out_features", model.num_classes)
    classes = int(classes)
    if classes < 1:
        raise ValueError("The model must provide at least one output class")

    classifier = model.to(selected_device).eval()
    device = selected_device
    input_size = model_input_size
    mean = torch.tensor(model_mean, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(model_std, dtype=torch.float32).view(1, 3, 1, 1)
    num_classes = classes


def model_metadata() -> dict:
    if input_size is None or num_classes is None:
        raise RuntimeError("The model has not been loaded")
    return {
        "name": MODEL_NAME,
        "inputs": [{
            "name": "pixel_values",
            "datatype": "FP32",
            "shape": [1, *input_size],
        }],
        "outputs": [{
            "name": "scores",
            "datatype": "FP32",
            "shape": [1, num_classes],
        }],
    }


def infer_tensor(body: dict) -> dict:
    if classifier is None or device is None or input_size is None:
        raise RuntimeError("The model has not been loaded")
    if mean is None or std is None or num_classes is None:
        raise RuntimeError("The model preprocessing has not been configured")

    inputs = body.get("inputs")
    if not isinstance(inputs, list) or len(inputs) != 1:
        raise ValueError("Exactly one input named 'pixel_values' is required")
    input_tensor = inputs[0]
    if not isinstance(input_tensor, dict) or input_tensor.get("name") != "pixel_values":
        raise ValueError("The input must be named 'pixel_values'")
    if input_tensor.get("datatype") != "FP32":
        raise ValueError("pixel_values must use FP32")

    shape = input_tensor.get("shape")
    if (
        not isinstance(shape, list)
        or any(type(dimension) is not int or dimension < 1 for dimension in shape)
        or len(shape) != 4
        or shape[0] != 1
        or shape[1:] != list(input_size)
    ):
        raise ValueError(
            f"pixel_values must have shape [1, {input_size[0]}, "
            f"{input_size[1]}, {input_size[2]}]"
        )

    pixels = torch.tensor(input_tensor.get("data"), dtype=torch.float32)
    pixels = pixels.reshape(shape)
    if not torch.isfinite(pixels).all() or pixels.min() < 0 or pixels.max() > 1:
        raise ValueError("pixel_values must contain finite values in [0, 1]")

    normalized = (pixels.to(device) - mean.to(device)) / std.to(device)
    with torch.inference_mode():
        scores = classifier(normalized)
    if (
        not isinstance(scores, Tensor)
        or tuple(scores.shape) != (1, num_classes)
        or not scores.is_floating_point()
        or not torch.isfinite(scores).all().item()
    ):
        raise ValueError("The model must return finite class scores with shape [1, classes]")

    scores = scores.detach().cpu().float()
    return {"outputs": [{
        "name": "scores",
        "datatype": "FP32",
        "shape": list(scores.shape),
        "data": scores.reshape(-1).tolist(),
    }]}


class Handler(BaseHTTPRequestHandler):
    def send_json(self, body: dict, status: int = 200) -> None:
        encoded = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:
        if unquote(self.path) == f"/v2/models/{MODEL_NAME}":
            self.send_json(model_metadata())
        else:
            self.send_json({"error": "not found"}, 404)

    def do_POST(self) -> None:
        if unquote(self.path) != f"/v2/models/{MODEL_NAME}/infer":
            self.send_json({"error": "not found"}, 404)
            return
        try:
            content_length = int(self.headers["Content-Length"])
            body = json.loads(self.rfile.read(content_length))
            if not isinstance(body, dict):
                raise ValueError("The request body must be a JSON object")
            self.send_json(infer_tensor(body))
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            self.send_json({"error": str(exc)}, 400)


def main() -> None:
    load_model()
    print(f"Serving on http://{HOST}:{PORT}", flush=True)
    HTTPServer((HOST, PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
