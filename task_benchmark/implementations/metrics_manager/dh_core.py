from __future__ import annotations

import json
from math import isfinite
from pathlib import Path
from typing import Any

from .base_metrics_manager import JSONValue, MetricsManager


class DHCoreMetricsManager(MetricsManager):
    """Convert a local DHCore metrics export into the service metrics schema.

    Select entries by exact entry ID or pod, service, container, or dhcore_run
    label value. Preserve units; recalculate standard summaries when filtering.
    """

    def __init__(
        self,
        metrics_path: str | Path = "examples/json/dh_core_metrics.json",
    ) -> None:
        self.metrics_path = Path(metrics_path).expanduser()

    def collect_metrics(
        self,
        service_id: str,
        params: dict[str, Any] | None = None,
        *,
        window_from: int | None = None,
        window_to: int | None = None,
    ) -> JSONValue:
        """Read and convert matching series; request params are currently unused."""
        self._validate_window(window_from, window_to)
        if not isinstance(service_id, str):
            raise TypeError("service_id must be a string")
        if not service_id.strip():
            raise ValueError("service_id must not be empty")

        with self.metrics_path.open(encoding="utf-8") as source:
            entries = json.load(source)
        if not isinstance(entries, list):
            raise ValueError("DHCore metrics must be a list of entries")

        converted = []
        timestamps: list[int] = []
        matched = False
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError("Each DHCore entry must be an object")
            if not self._matches(entry, service_id):
                continue
            matched = True
            for metric in entry["metrics"]:
                values = [
                    {
                        "timestamp": self._timestamp(point["timestamp"]),
                        "value": self._number(point["value"]),
                    }
                    for point in metric["metrics"]
                ]
                summary = [
                    {
                        "name": self._text(statistic["name"]),
                        "value": self._number(statistic["value"]),
                    }
                    for statistic in metric["summary"]
                ]
                if window_from is not None or window_to is not None:
                    values = [
                        point for point in values
                        if (window_from is None or point["timestamp"] >= window_from)
                        and (window_to is None or point["timestamp"] <= window_to)
                    ]
                    if values:
                        measurements = [point["value"] for point in values]
                        total = sum(measurements)
                        summary = [
                            {"name": "sum", "value": total},
                            {"name": "avg", "value": total / len(measurements)},
                            {"name": "max", "value": max(measurements)},
                            {"name": "min", "value": min(measurements)},
                        ]
                    else:
                        summary = []
                converted.append({"metric": {
                    "name": self._text(metric["name"]),
                    "unit": self._text(metric["unit"]),
                    "values": values,
                    "summary": summary,
                }})
                timestamps.extend(point["timestamp"] for point in values)

        if not matched:
            raise ValueError(f"No DHCore metrics found for '{service_id}'")
        if not timestamps and (window_from is None or window_to is None):
            raise ValueError("Cannot determine a measurement window without timestamps")

        return {
            "service_id": service_id,
            "window_from": window_from if window_from is not None else min(timestamps),
            "window_to": window_to if window_to is not None else max(timestamps),
            "metrics": converted,
        }

    @staticmethod
    def _matches(entry: dict[str, Any], service_id: str) -> bool:
        if entry.get("id") == service_id:
            return True
        labels = entry.get("metadata", {}).get("labels", [])
        return any(
            f"{name}:{service_id}" in labels
            for name in ("pod", "service", "container", "dhcore_run")
        )

    @staticmethod
    def _timestamp(value: Any) -> int:
        if type(value) is not int:
            raise ValueError("DHCore timestamps must be integer Unix timestamps")
        return value

    @staticmethod
    def _number(value: Any) -> int | float:
        if type(value) not in (int, float) or not isfinite(value):
            raise ValueError("DHCore measurement values must be finite numbers")
        return value

    @staticmethod
    def _text(value: Any) -> str:
        if not isinstance(value, str):
            raise ValueError("DHCore metric names, units and summary names must be strings")
        return value