# Metrics Managers

`MetricsManager` is an abstract interface for fetching pod or service metrics:

```python
collect_metrics(service_id: str, params: dict[str, Any] | None = None) -> JSONValue
```

`JSONValue` represents decoded JSON data: objects, arrays, strings, numbers,
booleans, or null. The response is not a serialized JSON string and does not
require a particular metrics schema. Concrete implementations define how the
identifier selects a pod or service and how measurements are collected.

`params` optionally supplies implementation-specific request parameters, such
as a time range or metric filters. Existing calls without `params` remain valid.

## MockupMetricsManager

The mock reads a fixed response from a JSON file at construction time:

```python
import json

from task_benchmark.implementations.metrics_manager import MockupMetricsManager

manager = MockupMetricsManager("examples/json/metrics.json")
metrics = manager.collect_metrics("my-pod-or-service")
serialized_metrics = json.dumps(metrics)
```

Run this example from the repository root, or supply an absolute file path.
The mock returns the same fixture for every non-empty identifier. It does not
contact a service, filter measurements, rewrite metadata, or update timestamps.
Optional `params` are accepted but ignored and left unchanged by the mock.
Each response is a deep copy so callers cannot mutate subsequent responses.
The fixture remains available even if its file is removed after construction.

The provided fixture contains CPU, requested CPU, memory, requested memory,
volume usage/count, and empty network measurement series. Its values and units
are preserved as supplied, without interpretation or aggregation.

Missing files and invalid JSON raise the standard file/JSON parsing exceptions
during construction. Empty service identifiers raise `ValueError`.