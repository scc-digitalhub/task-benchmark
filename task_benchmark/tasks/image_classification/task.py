# SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...abstract import (
    BaseDataObject,
    BaseImplementation,
    BaseTask,
    RuntimeMetricsCollector,
)


@dataclass
class Prediction:
    label: str
    score: float


@dataclass
class ImageClassificationDataObject(BaseDataObject):
    images_path: list[str]
    labels: list[str]

    def __post_init__(
        self,
    ) -> None:
        if not self.images_path:
            raise ValueError("images_path cannot be empty")

        if len(self.images_path) != len(self.labels):
            raise ValueError(
                "images_path and labels must have the same length"
            )


class ImageClassificationModel(BaseImplementation):
    @abstractmethod
    def predict_batch(
        self,
        inputs: list[bytes],
        top_k: int = 5,
    ) -> list[list[Prediction]]:
        pass


class ImageClassificationTask(BaseTask[ImageClassificationDataObject]):
    """
    Task-level adapter for image classification.
    """

    task = "image-classification"

    def build_data_object(
        self,
        payload: dict[str, Any],
    ) -> BaseDataObject:
        images_path = payload.get("images_path")
        labels = payload.get("labels")

        if not isinstance(images_path, list) or not isinstance(labels, list):
            raise ValueError(
                "For image-classification, data_object must include "
                "'images_path' and 'labels' lists."
            )

        return ImageClassificationDataObject(
            images_path=[str(path) for path in images_path],
            labels=[str(label) for label in labels],
        )

    def _extract_labels_from_data_object(
        self,
        data_object: ImageClassificationDataObject,
    ) -> list[str]:
        # Preserve deterministic order while dropping duplicates.
        return list(dict.fromkeys(data_object.labels))

    def _validate_data_object(
        self,
        data_object: BaseDataObject,
    ) -> ImageClassificationDataObject:
        if not isinstance(data_object, ImageClassificationDataObject):
            raise TypeError(
                "data_object must be an ImageClassificationDataObject "
                f"instance, got {type(data_object).__name__}."
            )

        return data_object

    def get_ground_truth(
        self,
        data_object: ImageClassificationDataObject,
        index: int,
    ) -> str:
        return data_object.labels[index]

    def create_task_metrics(
        self,
    ) -> dict[str, int]:
        return {
            "top1_correct": 0,
            "top5_correct": 0,
            "skipped": 0,
            "total": 0,
        }

    def record_skipped_items(
        self,
        task_metrics: dict[str, int],
        count: int = 1,
    ) -> None:
        task_metrics["skipped"] += count

    def execute_batch(
        self,
        implementation: ImageClassificationModel,
        batch_inputs: list[bytes],
    ) -> list[list[str]]:
        predictions = implementation.predict_batch(
            batch_inputs,
            top_k=5,
        )

        return [
            [prediction.label for prediction in per_image]
            for per_image in predictions
        ]

    def update_task_metrics_from_batch(
        self,
        task_metrics: dict[str, int],
        batch_output: list[list[str]],
        batch_gt_labels: list[str],
    ) -> None:
        for pred_labels, gt_label in zip(batch_output, batch_gt_labels):
            if pred_labels and (
                pred_labels[0] in gt_label or gt_label in pred_labels[0]
            ):
                task_metrics["top1_correct"] += 1

            if any(
                pred in gt_label or gt_label in pred for pred in pred_labels[:5]
            ):
                task_metrics["top5_correct"] += 1

            task_metrics["total"] += 1

    def finalize_task_metrics(
        self,
        task_metrics: dict[str, int],
    ) -> dict[str, float | int]:
        total = task_metrics["total"]

        if total == 0:
            raise RuntimeError("No images evaluated.")

        return {
            "top1_accuracy": task_metrics["top1_correct"] / total,
            "top5_accuracy": task_metrics["top5_correct"] / total,
            "dataset_images_evaluated": total,
            "dataset_images_skipped": task_metrics["skipped"],
        }

    def build_task_summary_lines(
        self,
        report: dict[str, float | int],
    ) -> list[str]:
        return [
            f"Images evaluated: {report['dataset_images_evaluated']}",
            f"Images skipped: {report['dataset_images_skipped']}",
            f"Top-1 accuracy: {report['top1_accuracy']:.4f}",
            f"Top-5 accuracy: {report['top5_accuracy']:.4f}",
        ]

    def execute_evaluation(
        self,
        implementation: str,
        data_object: BaseDataObject,
        runtime_metrics: RuntimeMetricsCollector,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Execute full image classification evaluation.

        Orchestrates batch iteration, inference, and metrics collection.
        Delegates to task-specific helper methods for each stage.
        """

        batch_size = kwargs.get("batch_size", 8)

        implementation_import_path = kwargs.get("implementation_import_path", "")
        if implementation_import_path:
            importlib.import_module(implementation_import_path)

        data_object = self._validate_data_object(data_object)
        labels = self._extract_labels_from_data_object(
            data_object=data_object,
        )

        print(f"Loaded {len(data_object.images_path)} dataset rows.")

        from task_benchmark.implementations import implementation_registry

        implementation_kwargs = dict(kwargs)
        for key in (
            "batch_size",
            "implementation_import_path",
        ):
            implementation_kwargs.pop(key, None)

        classifier = implementation_registry.create(
            task=self.task,
            implementation=implementation,
            labels=labels,
            **implementation_kwargs,
        )

        task_metrics = self.create_task_metrics()

        for batch_start in range(0, len(data_object.images_path), batch_size):
            batch_wall_start, batch_cpu_start = runtime_metrics.start_batch()

            batch_image_paths = data_object.images_path[
                batch_start : batch_start + batch_size
            ]
            batch_inputs: list[bytes] = []
            batch_gt_labels: list[str] = []

            for batch_index, image_path_value in enumerate(batch_image_paths):
                global_index = batch_start + batch_index
                gt_label = self.get_ground_truth(
                    data_object,
                    global_index,
                )

                image_path = Path(image_path_value)
                if not image_path.is_file():
                    print(f"[WARNING] Missing input file: {image_path}")
                    self.record_skipped_items(task_metrics=task_metrics)
                    continue

                try:
                    batch_inputs.append(image_path.read_bytes())
                    batch_gt_labels.append(gt_label)
                except Exception as exc:
                    print(f"Failed reading input file: {image_path}")
                    print(exc)
                    self.record_skipped_items(task_metrics=task_metrics)

            if not batch_inputs:
                continue

            try:
                batch_output = self.execute_batch(
                    implementation=classifier,
                    batch_inputs=batch_inputs,
                )
            except Exception as exc:
                print(f"Task execution failed for batch {batch_start}")
                print(exc)
                self.record_skipped_items(
                    task_metrics=task_metrics,
                    count=len(batch_inputs),
                )
                continue

            self.update_task_metrics_from_batch(
                task_metrics=task_metrics,
                batch_output=batch_output,
                batch_gt_labels=batch_gt_labels,
            )

            runtime_metrics.finish_batch(
                batch_wall_start=batch_wall_start,
                batch_cpu_start=batch_cpu_start,
            )

        return self.finalize_task_metrics(task_metrics=task_metrics)
