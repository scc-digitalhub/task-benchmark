# SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Iterable, Mapping
from io import BytesIO
from numbers import Integral
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import Tensor

from task_benchmark.implementations.data_loader.image_classification.base_data_loader import (
    ImageClassificationDataLoader,
)


class HuggingFaceImageClassificationDataLoader(ImageClassificationDataLoader):
    """Iterate over an image-classification split from Hugging Face Datasets.

    ``local_path`` is used as the Hugging Face cache directory. Configure
    ``split``, ``image_column``, ``label_column``, ``id_column``, ``classes``,
    and the dataset config through ``params``. Other params are forwarded to
    :func:`datasets.load_dataset`.

    Images are returned as RGB CHW float32 tensors in the ``[0, 1]`` range.
    Model-specific resizing and normalization are left to the caller.
    """

    def __init__(
        self,
        dataset_id: str,
        local_path: str,
        params: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(dataset_id, local_path, params)

        options = dict(self.params)
        self.split = str(options.pop("split", "test"))
        self.image_column = str(options.pop("image_column", "image"))
        self.label_column = str(options.pop("label_column", "label"))
        self.id_column = options.pop("id_column", None)
        configured_classes = options.pop("classes", None)
        config_name = options.pop("name", None)
        cache_dir = options.pop("cache_dir", local_path or None)

        if not dataset_id:
            raise ValueError("dataset_id must not be empty")
        if not self.split:
            raise ValueError("split must not be empty")

        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError(
                "Hugging Face dataset loading requires the 'datasets' package; "
                "install it with `pip install 'task-benchmark[huggingface]'`."
            ) from exc

        self.dataset = load_dataset(
            dataset_id,
            name=config_name,
            split=self.split,
            cache_dir=cache_dir,
            **options,
        )
        self._rows = iter(self.dataset)
        self._index = 0
        self._seen_image_ids: set[str] = set()
        self.annotations: dict[str, tuple[str, ...]] = {}

        features = getattr(self.dataset, "features", {})
        label_feature = (
            features.get(self.label_column)
            if isinstance(features, Mapping)
            else None
        )
        feature_names = self._feature_class_names(label_feature)
        if configured_classes is not None:
            if not isinstance(configured_classes, Iterable) or isinstance(
                configured_classes, (str, bytes)
            ):
                raise TypeError("params['classes'] must be a sequence of class names")
            self.classes = tuple(str(class_name) for class_name in configured_classes)
        else:
            self.classes = feature_names

    @property
    def current_index(self) -> int:
        """Index of the next dataset row to be returned."""
        return self._index

    def labels_for(self, image_id: str) -> tuple[str, ...]:
        """Return the labels observed for a previously yielded image."""
        try:
            return self.annotations[image_id]
        except KeyError as exc:
            raise KeyError(f"No annotations found for image '{image_id}'") from exc

    def next(self) -> tuple[str, Tensor]:
        row = next(self._rows)
        if not isinstance(row, Mapping):
            raise TypeError("Hugging Face dataset rows must be mappings")
        if self.image_column not in row:
            raise ValueError(
                f"Image column '{self.image_column}' is missing from the dataset"
            )
        if self.label_column not in row:
            raise ValueError(
                f"Label column '{self.label_column}' is missing from the dataset"
            )

        image_id = self._image_id(row)
        if image_id in self._seen_image_ids:
            raise ValueError(f"Dataset contains duplicate image identifier '{image_id}'")

        labels = self._normalize_labels(row[self.label_column])
        image_tensor = self._image_to_tensor(row[self.image_column])

        self._seen_image_ids.add(image_id)
        self.annotations[image_id] = labels
        self._index += 1
        return image_id, image_tensor

    def _image_id(self, row: Mapping[str, Any]) -> str:
        id_column = self.id_column
        if id_column is None and "id" in row:
            id_column = "id"
        if id_column is not None:
            if id_column not in row or row[id_column] is None:
                raise ValueError(f"Image identifier column '{id_column}' is missing")
            image_id = str(row[id_column])
            if not image_id:
                raise ValueError(f"Image identifier column '{id_column}' is empty")
            return image_id
        return f"{self.dataset_id}:{self.split}:{self._index}"

    def _normalize_labels(self, value: Any) -> tuple[str, ...]:
        if value is None:
            raise ValueError("Dataset image has no class label")
        raw_labels = value if isinstance(value, (list, tuple)) else [value]
        if not raw_labels:
            raise ValueError("Dataset image has no class label")

        normalized: list[str] = []
        for label in raw_labels:
            if isinstance(label, Integral) and not isinstance(label, bool):
                if self.classes:
                    if label < 0 or label >= len(self.classes):
                        raise ValueError(f"Class label index {label} is out of range")
                    label = self.classes[label]
            class_name = str(label)
            if self.classes and class_name not in self.classes:
                raise ValueError(f"Unknown class label '{class_name}'")
            if class_name not in normalized:
                normalized.append(class_name)

        if not self.classes:
            self.classes = tuple(
                dict.fromkeys((*self.classes, *normalized))
            )
        return tuple(normalized)

    @staticmethod
    def _feature_class_names(feature: Any) -> tuple[str, ...]:
        names = getattr(feature, "names", None)
        if names is None:
            names = getattr(getattr(feature, "feature", None), "names", None)
        if names is None:
            return ()
        return tuple(str(name) for name in names)

    @staticmethod
    def _image_to_tensor(value: Any) -> Tensor:
        if isinstance(value, Image.Image):
            image = value.convert("RGB")
        elif isinstance(value, Mapping):
            image_bytes = value.get("bytes")
            image_path = value.get("path")
            if image_bytes is not None:
                with Image.open(BytesIO(image_bytes)) as source_image:
                    image = source_image.convert("RGB")
            elif image_path:
                with Image.open(Path(image_path)) as source_image:
                    image = source_image.convert("RGB")
            else:
                raise ValueError("Image mapping must contain 'bytes' or 'path'")
        elif isinstance(value, (str, Path)):
            with Image.open(value) as source_image:
                image = source_image.convert("RGB")
        else:
            raise TypeError(
                "Image column must contain a PIL image, image path, or bytes mapping"
            )

        width, height = image.size
        pixels = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
        return (
            pixels.reshape(height, width, 3)
            .permute(2, 0, 1)
            .to(dtype=torch.float32)
            .div_(255.0)
        )