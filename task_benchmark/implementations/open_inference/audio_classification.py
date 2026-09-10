from __future__ import annotations

import base64
import json
from typing import Any
from urllib import error, parse, request

from task_benchmark.implementations.registry import implementation_registry
from task_benchmark.tasks.audio_classification.task import (
    AudioClassificationModel,
    Prediction,
)


class OpenInferenceAudioClassifier(AudioClassificationModel):
    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        labels: list[str] | None = None,
        model_params: dict[str, Any] | None = None,
    ) -> None:
        _ = device, labels
        params = dict(model_params or {})
        self.base_url = str(params.pop("base_url", "http://localhost:8080"))
        self.timeout = float(params.pop("timeout", 60.0))
        if params:
            raise TypeError(f"Unexpected model parameters: {', '.join(params)}")
        self.model_name = model_name

    def predict_batch(
        self,
        inputs: list[bytes],
        sample_rate: int,
        top_k: int = 5,
    ) -> list[list[Prediction]]:
        payload = {
            "parameters": {"sample_rate": sample_rate},
            "inputs": [{
                "name": "audio",
                "shape": [len(inputs)],
                "datatype": "BYTES",
                "data": [base64.b64encode(item).decode("ascii") for item in inputs],
            }],
        }
        endpoint = (
            f"{self.base_url.rstrip('/')}/v2/models/"
            f"{parse.quote(self.model_name, safe='')}/infer"
        )
        try:
            http_request = request.Request(
                endpoint,
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with request.urlopen(http_request, timeout=self.timeout) as response:
                response = json.loads(response.read().decode())
        except error.HTTPError as exc:
            raise RuntimeError(exc.read().decode(errors="replace")) from exc
        except (error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"OpenInference request failed: {exc}") from exc

        outputs = {item["name"]: item["data"] for item in response.get("outputs", [])}
        labels = outputs.get("label", outputs.get("labels"))
        scores = outputs.get("score", outputs.get("scores"))
        if not isinstance(labels, list) or not isinstance(scores, list):
            raise RuntimeError("OpenInference response must contain labels and scores")
        if len(labels) != len(inputs) or len(scores) != len(inputs):
            raise RuntimeError("OpenInference response batch size does not match request")

        return [
            [
                Prediction(label=str(label), score=float(score))
                for label, score in zip(item_labels[:top_k], item_scores[:top_k])
            ]
            for item_labels, item_scores in zip(labels, scores)
        ]


implementation_registry.register(
    task="audio-classification",
    implementation="open-inference",
    implementation_cls=OpenInferenceAudioClassifier,
)
