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
class AudioClassificationDataObject(BaseDataObject):
    audio_paths: list[str]
    labels: list[str]
    sample_rate: int

    def __post_init__(
        self,
    ) -> None:
        if not self.audio_paths:
            raise ValueError("audio_paths cannot be empty")

        if len(self.audio_paths) != len(self.labels):
            raise ValueError(
                "audio_paths and labels must have the same length"
            )

        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be a positive integer")


class AudioClassificationModel(BaseImplementation):
    @abstractmethod
    def predict_batch(
        self,
        inputs: list[bytes],
        sample_rate: int,
        top_k: int = 5,
    ) -> list[list[Prediction]]:
        pass


class AudioClassificationTask(BaseTask[AudioClassificationDataObject]):
    """
    Task-level adapter for audio classification.
    """

    task = "audio-classification"

    def build_data_object(
        self,
        payload: dict[str, Any],
    ) -> BaseDataObject:
        audio_paths = payload.get("audio_paths")
        labels = payload.get("labels")
        sample_rate = payload.get("sample_rate")

        if not isinstance(audio_paths, list) or not isinstance(labels, list):
            raise ValueError(
                "For audio-classification, data_object must include "
                "'audio_paths' and 'labels' lists."
            )

        if not isinstance(sample_rate, int):
            raise ValueError(
                "For audio-classification, data_object must include "
                "an integer 'sample_rate'."
            )

        return AudioClassificationDataObject(
            audio_paths=[str(path) for path in audio_paths],
            labels=[str(label) for label in labels],
            sample_rate=sample_rate,
        )

    def _extract_labels_from_data_object(
        self,
        data_object: AudioClassificationDataObject,
    ) -> list[str]:
        # Preserve deterministic order while dropping duplicates.
        return list(dict.fromkeys(data_object.labels))

    def _validate_data_object(
        self,
        data_object: BaseDataObject,
    ) -> AudioClassificationDataObject:
        if not isinstance(data_object, AudioClassificationDataObject):
            raise TypeError(
                "data_object must be an AudioClassificationDataObject "
                f"instance, got {type(data_object).__name__}."
            )

        return data_object

    def get_ground_truth(
        self,
        data_object: AudioClassificationDataObject,
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
        implementation: AudioClassificationModel,
        batch_inputs: list[bytes],
        sample_rate: int,
    ) -> list[list[str]]:
        predictions = implementation.predict_batch(
            batch_inputs,
            sample_rate=sample_rate,
            top_k=5,
        )

        return [
            [prediction.label for prediction in per_audio]
            for per_audio in predictions
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
            raise RuntimeError("No audio files evaluated.")

        return {
            "top1_accuracy": task_metrics["top1_correct"] / total,
            "top5_accuracy": task_metrics["top5_correct"] / total,
            "dataset_audio_evaluated": total,
            "dataset_audio_skipped": task_metrics["skipped"],
        }

    def build_task_summary_lines(
        self,
        report: dict[str, float | int],
    ) -> list[str]:
        return [
            f"Audio files evaluated: {report['dataset_audio_evaluated']}",
            f"Audio files skipped: {report['dataset_audio_skipped']}",
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
        Execute full audio classification evaluation.

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

        print(f"Loaded {len(data_object.audio_paths)} dataset rows.")

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

        for batch_start in range(0, len(data_object.audio_paths), batch_size):
            batch_wall_start, batch_cpu_start = runtime_metrics.start_batch()

            batch_audio_paths = data_object.audio_paths[
                batch_start : batch_start + batch_size
            ]
            batch_inputs: list[bytes] = []
            batch_gt_labels: list[str] = []

            for batch_index, audio_path_value in enumerate(batch_audio_paths):
                global_index = batch_start + batch_index
                gt_label = self.get_ground_truth(
                    data_object,
                    global_index,
                )

                audio_path = Path(audio_path_value)
                if not audio_path.is_file():
                    print(f"[WARNING] Missing input file: {audio_path}")
                    self.record_skipped_items(task_metrics=task_metrics)
                    continue

                try:
                    batch_inputs.append(audio_path.read_bytes())
                    batch_gt_labels.append(gt_label)
                except Exception as exc:
                    print(f"Failed reading input file: {audio_path}")
                    print(exc)
                    self.record_skipped_items(task_metrics=task_metrics)

            if not batch_inputs:
                continue

            try:
                batch_output = self.execute_batch(
                    implementation=classifier,
                    batch_inputs=batch_inputs,
                    sample_rate=data_object.sample_rate,
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
