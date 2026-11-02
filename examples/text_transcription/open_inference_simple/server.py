from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from math import gcd
from urllib.parse import unquote

import numpy as np
import torch


class WhisperTranscriber:
    """CPU Whisper backend for short mono audio clips."""

    def __init__(self, model_id: str) -> None:
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id).to("cpu")
        self.model.eval()
        self.sampling_rate = int(self.processor.feature_extractor.sampling_rate)
        self.max_duration = float(getattr(self.processor.feature_extractor, "chunk_length", 30))

    def transcribe(self, audio: np.ndarray, sampling_rate: int) -> str:
        if sampling_rate != self.sampling_rate:
            from scipy.signal import resample_poly

            divisor = gcd(sampling_rate, self.sampling_rate)
            audio = resample_poly(audio, self.sampling_rate // divisor, sampling_rate // divisor)
        inputs = self.processor(
            audio, sampling_rate=self.sampling_rate,
            return_tensors="pt", return_attention_mask=True,
        )
        with torch.inference_mode():
            tokens = self.model.generate(**inputs)
        return self.processor.batch_decode(tokens, skip_special_tokens=True)[0].strip()


def create_server(transcriber, host="0.0.0.0", port=9080, model_name="whisper-tiny.en"):
    """Create a serial HTTP server; model loading is separate for testing."""
    model_path = f"/v2/models/{model_name}"

    class Handler(BaseHTTPRequestHandler):
        def send_json(self, body, status=200):
            data = json.dumps(body, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            path = unquote(self.path)
            if path in {"/v2/health/live", "/v2/health/ready", f"{model_path}/ready"}:
                self.send_json({})
            elif path == "/v2":
                self.send_json({"name": "whisper-open-inference", "version": "1", "extensions": []})
            elif path == model_path:
                self.send_json({
                    "name": model_name, "versions": ["1"], "platform": "pytorch_whisper",
                    "inputs": [
                        {"name": "audio", "datatype": "FP32", "shape": [-1]},
                        {"name": "sampling_rate", "datatype": "INT64", "shape": [1]},
                    ],
                    "outputs": [{"name": "text", "datatype": "BYTES", "shape": [1]}],
                })
            else:
                self.send_json({"error": "not found"}, 404)

        def do_POST(self):
            if unquote(self.path) != f"{model_path}/infer":
                self.send_json({"error": "not found"}, 404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length > 16 * 1024 * 1024:
                    self.close_connection = True
                    self.send_json({"error": "request exceeds 16 MiB"}, 413)
                    return
                if length <= 0:
                    raise ValueError("A non-empty JSON body is required")
                body = json.loads(self.rfile.read(length))
                if not isinstance(body, dict) or not isinstance(body.get("inputs"), list):
                    raise ValueError("inputs must be a list")
                inputs = {item["name"]: item for item in body["inputs"]}
                if len(inputs) != 2 or len(body["inputs"]) != 2 or set(inputs) != {"audio", "sampling_rate"}:
                    raise ValueError("Exactly audio and sampling_rate inputs are required")
                audio_input = inputs["audio"]
                rate_input = inputs["sampling_rate"]
                samples = audio_input["data"]
                if (
                    audio_input["datatype"] != "FP32" or not isinstance(samples, list)
                    or not samples or audio_input["shape"] != [len(samples)]
                    or any(type(value) not in (int, float) for value in samples)
                ):
                    raise ValueError("audio must be a non-empty FP32 vector with matching shape")
                rate_values = rate_input["data"]
                if (
                    rate_input["datatype"] != "INT64" or rate_input["shape"] != [1]
                    or not isinstance(rate_values, list) or len(rate_values) != 1
                    or type(rate_values[0]) is not int or not 1 <= rate_values[0] <= 192000
                ):
                    raise ValueError("sampling_rate must be an INT64 [1] value from 1 to 192000 Hz")
                sampling_rate = rate_values[0]
                if len(samples) / sampling_rate > transcriber.max_duration:
                    raise ValueError(f"Audio exceeds {transcriber.max_duration:g} seconds")
                audio = np.asarray(samples, dtype=np.float32)
                if not np.isfinite(audio).all():
                    raise ValueError("Audio must contain finite values")
                outputs = body.get("outputs", [{"name": "text"}])
                if outputs != [{"name": "text"}]:
                    raise ValueError("Only the text output is supported")
                request_id = body.get("id")
                if request_id is not None and not isinstance(request_id, str):
                    raise ValueError("id must be a string")
            except (ValueError, KeyError, TypeError, OverflowError, UnicodeDecodeError) as exc:
                self.send_json({"error": str(exc)}, 400)
                return
            try:
                text = transcriber.transcribe(audio, sampling_rate)
            except Exception:
                self.log_error("Whisper inference failed")
                self.send_json({"error": "Whisper inference failed"}, 500)
                return
            result = {
                "model_name": model_name, "model_version": "1",
                "outputs": [{"name": "text", "datatype": "BYTES", "shape": [1], "data": [text]}],
            }
            if request_id is not None:
                result["id"] = request_id
            self.send_json(result)

    return HTTPServer((host, port), Handler)


def main() -> None:
    model_id = os.getenv("OPEN_INFERENCE_MODEL", "openai/whisper-tiny.en")
    model_name = os.getenv("OPEN_INFERENCE_MODEL_NAME", "whisper-tiny.en")
    host = os.getenv("OPEN_INFERENCE_HOST", "0.0.0.0")
    port = int(os.getenv("OPEN_INFERENCE_PORT", "9080"))
    print(f"Loading {model_id} on CPU...", flush=True)
    backend = WhisperTranscriber(model_id)
    with create_server(backend, host, port, model_name) as server:
        print(f"Serving on http://{host}:{port}/v2/models/{model_name}", flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()