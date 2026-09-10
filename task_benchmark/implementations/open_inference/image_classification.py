# SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
import json
from typing import Any
from urllib import error, parse, request

from task_benchmark.tasks.image_classification.task import (
    ImageClassificationModel,
    Prediction,
)
from task_benchmark.implementations.registry import implementation_registry


class OpenInferenceImageClassifier(ImageClassificationModel):
    """Image classifier client for an OpenInference v2 HTTP server."""

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
            unknown = ", ".join(sorted(params))
            raise TypeError(f"Unexpected model parameters: {unknown}")
        if not model_name:
            raise ValueError("model_name is required for OpenInference")

        self.model_name = model_name

    def predict_batch(
        self,
        inputs: list[bytes],
        top_k: int = 5,
    ) -> list[list[Prediction]]:
        if not inputs:
            return []
        if top_k < 1:
            raise ValueError("top_k must be greater than zero")

        payload = {
            "inputs": [
                {
                    "name": "image",
                    "shape": [len(inputs)],
                    "datatype": "BYTES",
                    "data": [
                        base64.b64encode(image).decode("ascii")
                        for image in inputs
                    ],
                }
            ]
        }
        endpoint = (
            f"{self.base_url.rstrip('/')}/v2/models/"
            f"{parse.quote(self.model_name, safe='')}/infer"
        )

        try:
            http_request = request.Request(
                endpoint,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with request.urlopen(http_request, timeout=self.timeout) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"OpenInference server returned HTTP {exc.code}: {body}"
            ) from exc
        except (error.URLError, TimeoutError) as exc:
            raise RuntimeError(
                f"Could not reach OpenInference server at {endpoint}: {exc}"
            ) from exc
        except json.JSONDecodeError as exc:
            raise RuntimeError("OpenInference server returned invalid JSON") from exc

        return self._parse_predictions(
            response_payload=response_payload,
            batch_size=len(inputs),
            top_k=top_k,
        )

    @staticmethod
    def _parse_predictions(
        response_payload: dict[str, Any],
        batch_size: int,
        top_k: int,
    ) -> list[list[Prediction]]:
        outputs_raw = response_payload.get("outputs")
        if outputs_raw is None:
            raise RuntimeError(
                "OpenInference server returned no outputs. "
                "Check that the model endpoint is healthy and the input image is valid."
            )

        outputs = {
            output.get("name"): output.get("data")
            for output in outputs_raw
            if isinstance(output, dict)
        }
        labels = outputs.get("labels")
        scores = outputs.get("scores")
        if labels is None and outputs.get("label") is not None:
            labels = outputs.get("label")
        if scores is None and outputs.get("score") is not None:
            scores = outputs.get("score")

        if labels is None or scores is None:
            raise RuntimeError(
                "OpenInference response must contain label(s)/score(s) fields"
            )

        if not isinstance(labels, list) or not isinstance(scores, list):
            raise RuntimeError("OpenInference labels and scores must be list-valued")

        if len(labels) != batch_size and len(labels) == 1 and batch_size > 1:
            labels = labels * batch_size
        if len(scores) != batch_size and len(scores) == 1 and batch_size > 1:
            scores = scores * batch_size
        if len(labels) != batch_size or len(scores) != batch_size:
            raise RuntimeError("OpenInference response batch size does not match request")

        predictions: list[list[Prediction]] = []
        for image_idx, (image_labels, image_scores) in enumerate(zip(labels, scores)):
            if isinstance(image_labels, str):
                try:
                    image_labels = json.loads(image_labels)
                except json.JSONDecodeError as exc:
                    raise RuntimeError("OpenInference labels are not valid JSON") from exc

            if not isinstance(image_labels, list):
                image_labels = [image_labels]
            if not isinstance(image_scores, list):
                image_scores = [image_scores]
            if len(image_labels) != len(image_scores):
                raise RuntimeError(
                    f"OpenInference labels and scores are misaligned for item {image_idx}"
                )

            predictions.append(
                [
                    Prediction(label=str(label), score=float(score))
                    for label, score in zip(
                        image_labels[:top_k],
                        image_scores[:top_k],
                    )
                ]
            )

        return predictions


implementation_registry.register(
    task="image-classification",
    implementation="open-inference",
    implementation_cls=OpenInferenceImageClassifier,
)