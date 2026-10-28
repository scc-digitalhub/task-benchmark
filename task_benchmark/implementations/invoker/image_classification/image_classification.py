# SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor

from task_benchmark.implementations.invoker.base_invoker import Invoker


class HuggingFaceImageClassificationInvoker(Invoker[Tensor, Tensor]):
    """Run Hugging Face image-classification models on input image tensors.

    ``model_name`` can be a Hugging Face Hub model ID or a local model path.
    The image processor handles resizing and normalization; ``invoke`` returns
    the model's logits tensor without applying softmax.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        model_kwargs: dict[str, Any] | None = None,
        processor_kwargs: dict[str, Any] | None = None,
    ) -> None:
        if not model_name:
            raise ValueError("model_name must not be empty")

        try:
            from transformers import AutoImageProcessor, AutoModelForImageClassification
        except ImportError as exc:
            raise ImportError(
                "Hugging Face model inference requires the 'transformers' package"
            ) from exc

        self.device = torch.device(device)
        self.processor = AutoImageProcessor.from_pretrained(
            model_name,
            **dict(processor_kwargs or {}),
        )
        self.model = AutoModelForImageClassification.from_pretrained(
            model_name,
            **dict(model_kwargs or {}),
        )
        self.model.to(self.device)
        self.model.eval()

    def label_for(self, class_id: int) -> str:
        """Return the configured label for a class ID, or the ID as text."""
        if not isinstance(class_id, int) or isinstance(class_id, bool):
            raise TypeError("class_id must be an integer")

        id_to_label = getattr(self.model.config, "id2label", None)
        if isinstance(id_to_label, Mapping):
            label = id_to_label.get(class_id, id_to_label.get(str(class_id)))
            if label is not None:
                return str(label)
        return str(class_id)

    def invoke(self, test_data: Tensor) -> Tensor:
        """Preprocess an image tensor and return the model logits."""
        if not isinstance(test_data, Tensor):
            raise TypeError("test_data must be a torch.Tensor")
        if test_data.ndim not in {3, 4}:
            raise ValueError(
                "test_data must have CHW or BCHW dimensions, "
                f"got {test_data.ndim} dimensions"
            )

        model_inputs = self.processor(
            images=test_data.detach().cpu(),
            return_tensors="pt",
        )
        model_inputs = {
            name: value.to(self.device) if isinstance(value, Tensor) else value
            for name, value in model_inputs.items()
        }

        with torch.inference_mode():
            model_output = self.model(**model_inputs)

        logits = getattr(model_output, "logits", None)
        if not isinstance(logits, Tensor):
            raise RuntimeError(
                "The image-classification model did not return a logits tensor"
            )
        return logits