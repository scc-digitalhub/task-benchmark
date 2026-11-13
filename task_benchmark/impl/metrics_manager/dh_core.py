from __future__ import annotations

import json
import os
from configparser import ConfigParser
from math import isfinite
from pathlib import Path
from typing import Any
from urllib.parse import quote
from urllib.request import Request, urlopen

from .base_metrics_manager import JSONValue, MetricsManager


DEFAULT_RUNS_URL = (
    "https://core.dev.atlas.fbk.eu/api/v1/-/test-hf/resource_metrics/runs"
)


class DHCoreMetricsManager(MetricsManager):
    """Fetch and convert a DHCore run's metrics into the service metrics schema.

    Select entries by exact entry ID or pod, service, container, or dhcore_run
    label value. Preserve units; recalculate standard summaries when filtering.
    """

    def __init__(
        self,
        runs_url: str | None = None,
    ) -> None:
        configured_url = (
            runs_url
            if runs_url is not None
            else os.getenv("DHCORE_RESOURCE_METRICS_RUNS_URL", DEFAULT_RUNS_URL)
        )
        if not isinstance(configured_url, str):
            raise TypeError("runs_url must be a string")
        if not configured_url.strip():
            raise ValueError("runs_url must not be empty")
        self.runs_url = configured_url.rstrip("/")

    def collect_metrics(
        self,
        service_id: str,
        params: dict[str, Any] | None = None,
        *,
        window_from: int | None = None,
        window_to: int | None = None,
        metric_names: list[str] | None = None,
    ) -> JSONValue:
        """Fetch and convert matching series, optionally filtering metric names."""
        self._validate_window(window_from, window_to)
        if not isinstance(service_id, str):
            raise TypeError("service_id must be a string")
        if not service_id.strip():
            raise ValueError("service_id must not be empty")
        if metric_names is not None:
            if not isinstance(metric_names, list):
                raise TypeError("metric_names must be a list of strings or None")
            if any(not isinstance(name, str) for name in metric_names):
                raise TypeError("metric_names must contain only strings")

        access_token = self._access_token()
        url = f"{self.runs_url}/{quote(service_id, safe='')}"
        request = Request(
            url,
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {access_token}",
            },
        )
        with urlopen(request, timeout=30) as response:
            entries = json.load(response)
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
                if metric_names is not None and metric["name"] not in metric_names:
                    continue
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
                converted.append({
                    "name": self._text(metric["name"]),
                    "unit": self._text(metric["unit"]),
                    "values": values,
                    "summary": summary,
                })
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
    def _access_token() -> str:
        access_token = os.getenv("DHCORE_ACCESS_TOKEN")
        if access_token and access_token.strip():
            return access_token.strip()

        config_path = Path.home() / ".dhcore.ini"
        if not config_path.is_file():
            raise RuntimeError(
                "Set DHCORE_ACCESS_TOKEN or configure dhcore_access_token "
                "in the active environment in ~/.dhcore.ini"
            )

        config = ConfigParser(interpolation=None)
        if not config.read(config_path, encoding="utf-8"):
            raise RuntimeError("Unable to read ~/.dhcore.ini")
        active_environment = config.defaults().get("current_environment")
        if not active_environment or not config.has_section(active_environment):
            raise RuntimeError(
                "The active DigitalHub environment is not configured in ~/.dhcore.ini"
            )

        access_token = config.get(
            active_environment,
            "dhcore_access_token",
            fallback="",
        )
        if not access_token.strip():
            raise RuntimeError(
                "No dhcore_access_token is configured for the active "
                "DigitalHub environment in ~/.dhcore.ini"
            )
        return access_token.strip()

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