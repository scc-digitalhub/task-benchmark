import base64
import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from io import BytesIO
from urllib.parse import unquote

import torch
from PIL import Image


MODEL = os.getenv("OPEN_INFERENCE_MODEL", "nateraw/vit-base-patch16-224-cifar10")
MODEL_NAME = os.getenv("OPEN_INFERENCE_MODEL_NAME", "vit-base-patch16-224")
HOST = os.getenv("OPEN_INFERENCE_HOST", "0.0.0.0")
PORT = int(os.getenv("OPEN_INFERENCE_PORT", "8080"))
TOP_K = int(os.getenv("OPEN_INFERENCE_TOP_K", "5"))
DEVICE = -1 if os.getenv("OPEN_INFERENCE_DEVICE", "cpu") == "cpu" else 0

classifier = None


def model_metadata():
    return {
        "name": MODEL_NAME,
        "inputs": [{"name": "pixel_values", "datatype": "FP32", "shape": [1, 3, -1, -1]}],
        "outputs": [{"name": "scores", "datatype": "FP32",
                     "shape": [1, classifier.model.config.num_labels]}],
        "parameters": {"id2label": classifier.model.config.id2label, "model_id": MODEL},
    }


def infer_tensor(body):
    input_tensor = next(item for item in body["inputs"] if item["name"] == "pixel_values")
    if input_tensor["datatype"] != "FP32":
        raise ValueError("pixel_values must use FP32")
    pixels = torch.tensor(input_tensor["data"], dtype=torch.float32).reshape(input_tensor["shape"])
    if pixels.ndim != 4 or pixels.shape[0] != 1 or pixels.shape[1] != 3:
        raise ValueError("pixel_values must have shape [1, 3, height, width]")
    if not torch.isfinite(pixels).all() or pixels.min() < 0 or pixels.max() > 1:
        raise ValueError("pixel_values must contain finite values in [0, 1]")
    inputs = classifier.image_processor(
        images=pixels,
        return_tensors="pt",
        do_rescale=False,
        input_data_format="channels_first",
    )
    inputs = {name: value.to(classifier.model.device) for name, value in inputs.items()}
    with torch.inference_mode():
        scores = classifier.model(**inputs).logits.detach().cpu().float()
    return {"outputs": [{
        "name": "scores", "datatype": "FP32", "shape": list(scores.shape),
        "data": scores.reshape(-1).tolist(),
    }]}


class Handler(BaseHTTPRequestHandler):
    def send_json(self, body, status=200):
        encoded = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self):
        if unquote(self.path) == f"/v2/models/{MODEL_NAME}":
            self.send_json(model_metadata())
        else:
            self.send_json({"error": "not found"}, 404)

    def do_POST(self):
        if unquote(self.path) != f"/v2/models/{MODEL_NAME}/infer":
            self.send_json({"error": "not found"}, 404)
            return
        try:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if any(item["name"] == "pixel_values" for item in body["inputs"]):
                self.send_json(infer_tensor(body))
                return
            data = next(item["data"] for item in body["inputs"] if item["name"] == "image")
            images = [Image.open(BytesIO(base64.b64decode(item))).convert("RGB") for item in data]
            results = classifier(images, top_k=TOP_K)
            if not isinstance(results[0], list):
                results = [[item] for item in results]
        except Exception as exc:
            self.send_json({"error": str(exc)}, 400)
            return

        self.send_json({"outputs": [
            {"name": "label", "data": [[item["label"] for item in row] for row in results]},
            {"name": "score", "data": [[item["score"] for item in row] for row in results]},
        ]})


def main():
    from transformers import pipeline

    global classifier
    print(f"Loading {MODEL}...", flush=True)
    classifier = pipeline("image-classification", model=MODEL, device=DEVICE)
    print(f"Serving on http://{HOST}:{PORT}", flush=True)
    HTTPServer((HOST, PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
