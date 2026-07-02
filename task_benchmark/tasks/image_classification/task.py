# SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from transformers import AutoConfig

from task_benchmark.abstract import (
    BaseImplementation,
    BaseTask,
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
            "total": 0,
        }

    def update_task_metrics(
        self,
        task_metrics: dict[str, int],
        pred_labels: list[str],
        gt_label: str,
        predicts_task_labels: bool,
        label_mapping: dict[str, str],
    ) -> None:
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
        }

    def build_task_summary_lines(
        self,
        report: dict[str, float | int],
    ) -> list[str]:
        return [
            f"Images evaluated: {report['dataset_images_evaluated']}",
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
