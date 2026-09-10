import base64
import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from io import BytesIO
from urllib.parse import unquote

from PIL import Image
from transformers import pipeline


MODEL = os.getenv("OPEN_INFERENCE_MODEL", "google/vit-base-patch16-224")
HOST = os.getenv("OPEN_INFERENCE_HOST", "0.0.0.0")
PORT = int(os.getenv("OPEN_INFERENCE_PORT", "8080"))
TOP_K = int(os.getenv("OPEN_INFERENCE_TOP_K", "5"))
DEVICE = -1 if os.getenv("OPEN_INFERENCE_DEVICE", "cpu") == "cpu" else 0

print(f"Loading {MODEL}...", flush=True)
classifier = pipeline("image-classification", model=MODEL, device=DEVICE)


class Handler(BaseHTTPRequestHandler):
    def send_json(self, body, status=200):
        encoded = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self):
        if unquote(self.path) == f"/v2/models/{MODEL}":
            self.send_json({
                "name": MODEL,
                "inputs": [{"name": "image", "datatype": "BYTES", "shape": [-1]}],
                "outputs": [{"name": "label"}, {"name": "score"}],
            })
        else:
            self.send_json({"error": "not found"}, 404)

    def do_POST(self):
        if unquote(self.path) != f"/v2/models/{MODEL}/infer":
            self.send_json({"error": "not found"}, 404)
            return
        try:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
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


print(f"Serving on http://{HOST}:{PORT}", flush=True)
HTTPServer((HOST, PORT), Handler).serve_forever()
