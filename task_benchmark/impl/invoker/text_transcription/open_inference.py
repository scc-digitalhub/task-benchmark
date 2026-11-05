from __future__ import annotations

import json
from math import isfinite
from urllib import error, parse, request

import torch

from task_benchmark.abstract.audio_sample import AudioSample

from .base_invoker import TextTranscriptionInvoker


class OpenInferenceTextTranscriptionInvoker(TextTranscriptionInvoker):
    """Send numeric audio over Open Inference V2 HTTP and decode UTF-8 text."""

    def __init__(self, service_url: str, timeout: float = 120.0) -> None:
        parsed = parse.urlsplit(service_url)
        if (
            parsed.scheme not in {"http", "https"} or not parsed.netloc
            or parsed.query or parsed.fragment or not parsed.path.rstrip("/").endswith("/infer")
        ):
            raise ValueError("service_url must be an HTTP(S) model URL ending in /infer")
        if not isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be positive and finite")
        self.service_url = service_url.rstrip("/")
        self.timeout = timeout

    def invoke(self, test_data: AudioSample) -> str:
        if not isinstance(test_data, AudioSample):
            raise TypeError("test_data must be an AudioSample")
        waveform = test_data.waveform
        if (
            not isinstance(waveform, torch.Tensor) or waveform.ndim != 1
            or waveform.numel() == 0 or not waveform.is_floating_point()
            or not torch.isfinite(waveform).all().item()
        ):
            raise ValueError("waveform must be a non-empty, finite mono floating-point tensor")
        if type(test_data.sampling_rate) is not int or test_data.sampling_rate <= 0:
            raise ValueError("sampling_rate must be a positive integer in Hz")
        payload = {
            "inputs": [
                {"name": "audio", "datatype": "FP32", "shape": [waveform.numel()],
                 "data": waveform.detach().to(device="cpu", dtype=torch.float32).tolist()},
                {"name": "sampling_rate", "datatype": "INT64", "shape": [1],
                 "data": [test_data.sampling_rate]},
            ],
            "outputs": [{"name": "text"}],
        }
        http_request = request.Request(
            self.service_url, data=json.dumps(payload, allow_nan=False).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            with request.urlopen(http_request, timeout=self.timeout) as response:
                result = json.load(response)
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Open Inference returned HTTP {exc.code}: {body}") from exc
        except (error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Open Inference transcription request failed: {exc}") from exc
        if not isinstance(result, dict) or not isinstance(result.get("outputs"), list):
            raise RuntimeError("Invalid Open Inference transcription response")
        outputs = [item for item in result["outputs"]
                   if isinstance(item, dict) and item.get("name") == "text"]
        if len(outputs) != 1:
            raise RuntimeError("Response must contain exactly one 'text' output")
        output = outputs[0]
        data = output.get("data")
        if (
            output.get("datatype") != "BYTES" or output.get("shape") != [1]
            or not isinstance(data, list) or len(data) != 1 or not isinstance(data[0], str)
        ):
            raise RuntimeError("Expected a BYTES text output with shape [1] and one UTF-8 string")
        return data[0]