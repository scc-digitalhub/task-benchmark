# SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import csv
from pathlib import Path

import torch
from PIL import Image
from torch import Tensor

from task_benchmark.implementations.data_loader.data_loader import DataLoader


class ImageClassificationDataLoader(DataLoader):
    """Load named RGB images and their class annotations from a dataset folder.

    The folder must contain ``classes.txt`` (one class per line),
    ``images.txt`` (one image path relative to the folder per line), and
    ``annotations.csv`` with ``image,labels`` columns. Separate multiple labels
    in the CSV ``labels`` field with semicolons.

    Images are returned as CHW float32 tensors with values in ``[0, 1]``.
    Resizing and model-specific normalization are intentionally left to the
    inference pipeline. Ground-truth labels can be retrieved with
    :meth:`labels_for` using the image identifier returned by :meth:`next`.
    """

    def __init__(self, dataset_id: str, local_path: str) -> None:
        super().__init__(dataset_id=dataset_id, local_path=local_path)
        self.root = Path(local_path).expanduser().resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f"Dataset directory does not exist: {self.root}")

        self.classes = self._read_classes(self.root / "classes.txt")
        self.image_paths = self._read_images(self.root / "images.txt")
        self.annotations = self._read_annotations(self.root / "annotations.csv")
        self._index = 0

    @property
    def current_index(self) -> int:
        """Index of the next image to be returned."""
        return self._index

    def labels_for(self, image_id: str) -> tuple[str, ...]:
        """Return the annotated classes for an image identifier."""
        try:
            return self.annotations[image_id]
        except KeyError as exc:
            raise KeyError(f"No annotations found for image '{image_id}'") from exc

    def next(self) -> tuple[str, Tensor]:
        if self._index >= len(self.image_paths):
            raise StopIteration

        image_id, image_path = self.image_paths[self._index]
        with Image.open(image_path) as source_image:
            image = source_image.convert("RGB")
            width, height = image.size
            pixels = torch.frombuffer(
                bytearray(image.tobytes()),
                dtype=torch.uint8,
            )

        image_tensor = (
            pixels.reshape(height, width, 3)
            .permute(2, 0, 1)
            .to(dtype=torch.float32)
            .div_(255.0)
        )
        self._index += 1
        return image_id, image_tensor

    def _read_classes(self, classes_file: Path) -> tuple[str, ...]:
        self._require_file(classes_file)
        classes = tuple(
            line.strip()
            for line in classes_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        if not classes:
            raise ValueError("classes.txt must contain at least one class")
        if len(set(classes)) != len(classes):
            raise ValueError("classes.txt contains duplicate class names")
        return classes

    def _read_images(self, images_file: Path) -> list[tuple[str, Path]]:
        self._require_file(images_file)
        images: list[tuple[str, Path]] = []
        seen_image_ids: set[str] = set()

        for line in images_file.read_text(encoding="utf-8").splitlines():
            image_value = line.strip()
            if not image_value:
                continue
            image_id, image_path = self._resolve_image_path(image_value)
            if image_id in seen_image_ids:
                raise ValueError(f"images.txt contains duplicate image '{image_id}'")
            if not image_path.is_file():
                raise FileNotFoundError(f"Image file does not exist: {image_path}")
            seen_image_ids.add(image_id)
            images.append((image_id, image_path))

        if not images:
            raise ValueError("images.txt must contain at least one image")
        return images

    def _read_annotations(self, annotations_file: Path) -> dict[str, tuple[str, ...]]:
        self._require_file(annotations_file)
        annotations: dict[str, tuple[str, ...]] = {}
        known_classes = set(self.classes)

        with annotations_file.open(encoding="utf-8", newline="") as file_handle:
            reader = csv.DictReader(file_handle)
            if reader.fieldnames is None or not {"image", "labels"}.issubset(
                reader.fieldnames
            ):
                raise ValueError(
                    "annotations.csv must contain 'image' and 'labels' columns"
                )

            for row in reader:
                image_value = (row.get("image") or "").strip()
                image_labels = (row.get("labels") or "").strip()
                if not image_value or not image_labels:
                    raise ValueError(
                        "annotations.csv rows must include an image and labels"
                    )

                image_id, _ = self._resolve_image_path(image_value)
                if image_id in annotations:
                    raise ValueError(
                        f"annotations.csv contains duplicate image '{image_id}'"
                    )

                labels = tuple(
                    label.strip()
                    for label in image_labels.split(";")
                    if label.strip()
                )
                if not labels:
                    raise ValueError(
                        f"annotations for image '{image_id}' must include a label"
                    )
                unknown_classes = set(labels) - known_classes
                if unknown_classes:
                    unknown_text = ", ".join(sorted(unknown_classes))
                    raise ValueError(
                        f"Unknown class(es) for image '{image_id}': {unknown_text}"
                    )
                annotations[image_id] = labels

        image_ids = {image_id for image_id, _ in self.image_paths}
        missing_annotations = image_ids - annotations.keys()
        extra_annotations = annotations.keys() - image_ids
        if missing_annotations or extra_annotations:
            details = []
            if missing_annotations:
                details.append(
                    "missing annotations for: "
                    + ", ".join(sorted(missing_annotations))
                )
            if extra_annotations:
                details.append(
                    "annotations for unlisted images: "
                    + ", ".join(sorted(extra_annotations))
                )
            raise ValueError("; ".join(details))

        return annotations

    def _resolve_image_path(self, image_value: str) -> tuple[str, Path]:
        relative_path = Path(image_value)
        if relative_path.is_absolute():
            raise ValueError(f"Image path must be relative to dataset: {image_value}")

        image_path = (self.root / relative_path).resolve()
        try:
            image_id = image_path.relative_to(self.root).as_posix()
        except ValueError as exc:
            raise ValueError(
                f"Image path must stay inside dataset directory: {image_value}"
            ) from exc
        return image_id, image_path

    @staticmethod
    def _require_file(file_path: Path) -> None:
        if not file_path.is_file():
            raise FileNotFoundError(f"Required dataset file does not exist: {file_path}")