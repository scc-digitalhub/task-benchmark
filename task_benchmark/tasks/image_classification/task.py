# SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any


from task_benchmark.abstract import (
    BaseImplementation,
    BaseTask,
    RuntimeMetricsCollector,
)


@dataclass
class Prediction:
    label: str
    score: float


class ImageClassificationModel(BaseImplementation):
    @abstractmethod
    def predict_batch(
        self,
        inputs: list[bytes],
        top_k: int = 5,
    ) -> list[list[Prediction]]:
        pass


class ImageClassificationTask(BaseTask):
    """
    Task-level adapter for image classification.
    """

    task = "image-classification"
    input_column = "image_path"
    label_column = "label"
    description_column = "class_description"

    def _resolve_dataset_paths(
        self,
        dataset_path: Path,
        dataset_csv: str = "dataset.csv",
        image_dir: str = "images",
    ) -> tuple[Path, Path]:
        if not dataset_path.is_dir():
            raise ValueError(
                "dataset_path must be a directory containing the dataset CSV "
                "and the input files directory."
            )

        row_path = dataset_path / dataset_csv
        dataset_root = dataset_path

        return row_path, dataset_root / image_dir

    @property
    def required_columns(self) -> set[str]:
        return {
            self.input_column,
            self.label_column,
        }

    def _extract_class_descriptions(
        self,
        rows: list[dict[str, str]],
    ) -> dict[str, str]:
        class_descriptions: dict[str, str] = {}

        for row in rows:
            class_id = row[self.label_column]
            if class_id not in class_descriptions:
                class_descriptions[class_id] = row.get(
                    self.description_column,
                    class_id,
                )

        return class_descriptions

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
            [
                prediction.label
                for prediction in per_image
            ]
            for per_image in predictions
        ]

    def update_task_metrics_from_batch(
        self,
        task_metrics: dict[str, int],
        batch_output: list[list[str]],
        batch_gt_labels: list[str],
    ) -> None:
        for pred_labels, gt_label in zip(
            batch_output,
            batch_gt_labels,
        ):
            pred_task_labels = pred_labels

            if (
                pred_task_labels
                and pred_task_labels[0]
                == gt_label
            ):
                task_metrics["top1_correct"] += 1

            if gt_label in pred_task_labels[:5]:
                task_metrics["top5_correct"] += 1

            task_metrics["total"] += 1

    def finalize_task_metrics(
        self,
        task_metrics: dict[str, int],
    ) -> dict[str, float | int]:
        total = task_metrics["total"]

        if total == 0:
            raise RuntimeError(
                "No images evaluated."
            )

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
        dataset_path: Path,
        implementation: str,
        runtime_metrics: RuntimeMetricsCollector,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Execute full image classification evaluation.

        Orchestrates batch iteration, inference, and metrics collection.
        Delegates to task-specific helper methods for each stage.
        """

        dataset_path = Path(dataset_path)
        row_path, image_path = self._resolve_dataset_paths(
            dataset_path=dataset_path,
            dataset_csv=kwargs.get("dataset_csv", "dataset.csv"),
            image_dir=kwargs.get("image_dir", "images"),
        )

        # Extract task-specific parameters from kwargs
        batch_size = kwargs.get("batch_size", 8)

        implementation_import_path = kwargs.get("implementation_import_path", "")
        if implementation_import_path:
            importlib.import_module(implementation_import_path)

        # Load dataset
        rows = self.load_dataset_rows(row_path)
        print(f"Loaded {len(rows)} dataset rows.")

        # Extract class descriptions
        class_descriptions = self._extract_class_descriptions(rows=rows)
    
        # Create implementation
        from task_benchmark.implementations import implementation_registry

        implementation_kwargs = dict(kwargs)
        for key in (
            "batch_size",
            "implementation_import_path",
            "dataset_csv",
            "image_dir",
        ):
            implementation_kwargs.pop(key, None)

        classifier = implementation_registry.create(
            task=self.task,
            implementation=implementation,
            class_descriptions=class_descriptions,
            **implementation_kwargs,
        )

        # Initialize task metrics
        task_metrics = self.create_task_metrics()

        # Orchestrate batch iteration
        for batch_start in range(0, len(rows), batch_size):
            batch_wall_start, batch_cpu_start = runtime_metrics.start_batch()

            batch_rows = rows[batch_start : batch_start + batch_size]

            batch_inputs: list[bytes] = []
            batch_gt_labels: list[str] = []

            # Load batch inputs from disk
            for row in batch_rows:
                input_path = self.resolve_input_path(
                    row[self.input_column],
                    image_path,
                )

                gt_label = self.get_ground_truth(row)

                if not input_path.is_file():
                    print(f"[WARNING] Missing input file: {input_path}")
                    self.record_skipped_items(task_metrics=task_metrics)
                    continue

                try:
                    batch_inputs.append(input_path.read_bytes())
                    batch_gt_labels.append(gt_label)
                except Exception as exc:
                    print(f"Failed reading input file: {input_path}")
                    print(exc)
                    self.record_skipped_items(task_metrics=task_metrics)

            if not batch_inputs:
                continue

            # Execute batch inference
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

            # Update task metrics from batch
            self.update_task_metrics_from_batch(
                task_metrics=task_metrics,
                batch_output=batch_output,
                batch_gt_labels=batch_gt_labels,
            )

            runtime_metrics.finish_batch(
                batch_wall_start=batch_wall_start,
                batch_cpu_start=batch_cpu_start,
            )

        # Finalize and return metrics
        return self.finalize_task_metrics(task_metrics=task_metrics)
