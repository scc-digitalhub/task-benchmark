from __future__ import annotations

import time
from typing import Any, Generic, TypeVar, cast

from task_benchmark.impl.data_loader.data_loader import DataLoader
from task_benchmark.impl.evaluator.base_evaluator import Evaluator
from task_benchmark.impl.invoker.base_invoker import Invoker
from task_benchmark.impl.metrics_manager.base_metrics_manager import JSONValue


InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


class BenchmarkManager(Generic[InputT, OutputT]):
    """Run inference and evaluation for every item in a dataset."""

    def __init__(
        self,
        data_loader: DataLoader,
        invoker: Invoker[InputT, OutputT],
        evaluator: Evaluator[OutputT],
        service_id: str | None = None,
    ) -> None:
        if service_id is not None and not service_id.strip():
            raise ValueError("service_id must not be empty")

        inferred_service_id = getattr(invoker, "service_id", None)
        self.service_id = service_id or (
            inferred_service_id
            if isinstance(inferred_service_id, str) and inferred_service_id.strip()
            else type(invoker).__name__
        )
        self.data_loader = data_loader
        self.invoker = invoker
        self.evaluator = evaluator

    def run_benchmark(self) -> JSONValue:
        """Run the full dataset and return metrics in the metrics-schema format."""
        window_from = int(time.time())
        inference_values: list[dict[str, int | float]] = []
        quality_score_values: list[dict[str, int | float]] = []

        for sample_id, test_data in self.data_loader:
            invocation_timestamp = int(time.time())
            inference_started = time.perf_counter()
            service_response = self.invoker.invoke(cast(InputT, test_data))
            inference_time = time.perf_counter() - inference_started
            inference_values.append({
                "timestamp": invocation_timestamp,
                "value": inference_time,
            })

            evaluation_timestamp = int(time.time())
            quality_score = self.evaluator.evaluate(sample_id, service_response)
            quality_score_values.append({
                "timestamp": evaluation_timestamp,
                "value": quality_score,
            })
            print(
                f"{sample_id}: inference_time={inference_time:.9f}s, "
                f"quality_score={quality_score:.4f}"
            )

        return {
            "service_id": self.service_id,
            "window_from": window_from,
            "window_to": int(time.time()),
            "metrics": [
                self._metric("inference_time", "seconds", inference_values),
                self._metric("quality_score", "ratio", quality_score_values),
            ],
        }

    @staticmethod
    def _metric(
        name: str,
        unit: str,
        values: list[dict[str, int | float]],
    ) -> dict[str, Any]:
        measurements = [float(item["value"]) for item in values]
        summary: list[dict[str, str | float]] = []
        if measurements:
            summary = [
                {"name": "avg", "value": sum(measurements) / len(measurements)},
                {"name": "max", "value": max(measurements)},
                {"name": "min", "value": min(measurements)},
            ]
        return {"name": name, "unit": unit, "values": values, "summary": summary}