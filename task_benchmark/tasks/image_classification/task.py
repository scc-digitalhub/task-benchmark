# SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import Any

from transformers import AutoConfig

from task_benchmark.abstract import (
    BaseImplementation,
    BaseTask,
    RuntimeMetricsCollector,
)

from .implementations.factory import create_image_classifier


def _build_label_to_wnid(
    class_descriptions: dict[str, str],
    model_id2label: dict[int, str],
) -> dict[str, str]:
    lower_model_labels = {
        label.lower(): label
        for label in model_id2label.values()
    }

    label_to_wnid: dict[str, str] = {}
    unmapped: list[str] = []

    for wnid, description in class_descriptions.items():
        desc_lower = description.lower()

        if desc_lower in lower_model_labels:
            label_to_wnid[lower_model_labels[desc_lower]] = wnid
            continue

        matched = None

        for model_lower, model_orig in lower_model_labels.items():
            if desc_lower in model_lower or model_lower in desc_lower:
                matched = model_orig
                break

        if matched:
            label_to_wnid[matched] = wnid
        else:
            unmapped.append(f"{wnid} ({description})")

    if unmapped:
        print(
            f"[WARNING] Could not map {len(unmapped)} classes.\n"
            + "\n".join(unmapped[:10])
        )

    return label_to_wnid


class ImageClassificationTask(BaseTask):
    """
    Task-level adapter for image classification.
    """

    task_name = "image-classification"
    input_column = "image_path"
    label_column = "wnid"
    description_column = "class_description"

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
        implementation: BaseImplementation,
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
        label_mapping: dict[str, str],
        implementation: BaseImplementation,
    ) -> None:
        predicts_task_labels = getattr(
            implementation,
            "predicts_wnid",
            False,
        )

        for pred_labels, gt_label in zip(
            batch_output,
            batch_gt_labels,
        ):
            if not predicts_task_labels:
                pred_task_labels = [
                    label_mapping.get(pred_label)
                    for pred_label in pred_labels
                ]
            else:
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

    def build_label_mapping(
        self,
        implementation_name: str,
        model_name: str,
        class_descriptions: dict[str, str],
    ) -> dict[str, str]:
        if implementation_name != "task-inference":
            return {}

        if not model_name:
            raise ValueError(
                "model_name is required when implementation_name='task-inference'."
            )

        print(f"Loading model config: {model_name}")
        config = AutoConfig.from_pretrained(model_name)

        id2label = {
            int(k): v
            for k, v in config.id2label.items()
        }

        print("Building label mapping...")

        mapping = _build_label_to_wnid(
            class_descriptions=class_descriptions,
            model_id2label=id2label,
        )

        print(f"Mapped {len(mapping)} labels.")

        return mapping

    def create_implementation(
        self,
        implementation_name: str,
        model_name: str,
        custom_model_name: str,
        custom_model_import_path: str,
        class_descriptions: dict[str, str],
        device: str,
    ) -> BaseImplementation:
        return create_image_classifier(
            implementation_name=implementation_name,
            model_name=model_name,
            custom_model_name=custom_model_name,
            custom_model_import_path=custom_model_import_path,
            class_descriptions=class_descriptions,
            device=device,
        )

    def execute_evaluation(
        self,
        dataset_path: Path,
        implementation_name: str,
        device: str,
        runtime_metrics: RuntimeMetricsCollector,
        **kwargs,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """
        Execute full image classification evaluation.

        Orchestrates batch iteration, inference, and metrics collection.
        Delegates to task-specific helper methods for each stage.
        """

        dataset_path = Path(dataset_path)

        # Extract task-specific parameters from kwargs
        task_inputs_dir_path = Path(kwargs.get("task_inputs_dir_path", "."))
        model_name = kwargs.get("model_name", "")
        batch_size = kwargs.get("batch_size", 8)
        custom_model_name = kwargs.get("custom_model_name", "hash-baseline")
        custom_model_import_path = kwargs.get("custom_model_import_path", "")

        # Load dataset
        rows = self.load_dataset_rows(dataset_path)
        print(f"Loaded {len(rows)} dataset rows.")

        # Extract class descriptions
        class_descriptions = self.extract_class_descriptions(rows=rows)

        # Build label mapping (task-specific)
        label_mapping = self.build_label_mapping(
            implementation_name=implementation_name,
            model_name=model_name,
            class_descriptions=class_descriptions,
        )

        # Create implementation
        classifier = self.create_implementation(
            implementation_name=implementation_name,
            model_name=model_name,
            custom_model_name=custom_model_name,
            custom_model_import_path=custom_model_import_path,
            class_descriptions=class_descriptions,
            device=device,
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
                    task_inputs_dir_path,
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
                label_mapping=label_mapping,
                implementation=classifier,
            )

            runtime_metrics.finish_batch(
                batch_wall_start=batch_wall_start,
                batch_cpu_start=batch_cpu_start,
            )

        # Finalize and return metrics
        task_report = self.finalize_task_metrics(task_metrics=task_metrics)

        return task_metrics, task_report
