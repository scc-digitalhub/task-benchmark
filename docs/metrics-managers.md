# Metrics Managers

`MetricsManager` is an abstract interface for fetching pod or service metrics:

```python
collect_metrics(
	service_id: str,
	params: dict[str, Any] | None = None,
	*,
	window_from: int | None = None,
	window_to: int | None = None,
) -> JSONValue
```

`JSONValue` represents decoded JSON data: objects, arrays, strings, numbers,
booleans, or null. The response is not a serialized JSON string and does not
require a particular metrics schema. Concrete implementations define how the
identifier selects a pod or service and how measurements are collected.

`params` optionally supplies implementation-specific request parameters, such
as a time range or metric filters. Existing calls without `params` remain valid.

`window_from` and `window_to` are optional keyword-only inclusive Unix timestamp
bounds in seconds. Python `int` supports long integer values. Omit either bound
for an open-ended range. Boolean, string, or floating-point bounds are rejected;
`window_from` must not exceed `window_to`.

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
Time bounds are validated but do not filter the mock's fixed fixture.
Each response is a deep copy so callers cannot mutate subsequent responses.
The fixture remains available even if its file is removed after construction.

The provided fixture contains CPU, requested CPU, memory, requested memory,
volume usage/count, and empty network measurement series. Its values and units
are preserved as supplied, without interpretation or aggregation.

Missing files and invalid JSON raise the standard file/JSON parsing exceptions
during construction. Empty service identifiers raise `ValueError`.

## DHCoreMetricsManager

`DHCoreMetricsManager` reads a DHCore JSON export on each call and converts its
matching series into the format defined in `json/metrics_schema.json`:

```python
from task_benchmark.implementations.metrics_manager import DHCoreMetricsManager

manager = DHCoreMetricsManager("examples/json/dh_core_metrics.json")
metrics = manager.collect_metrics("949c8a888235496f875a1992bab1b32b")
```

The constructor defaults to this fixture path, relative to the working directory.
You can instead provide an absolute path. `service_id` must match an entry's
complete `id`, or a `pod`, `service`, `container`, or `dhcore_run` label value
exactly. The example uses the fixture's DHCore run ID. Unmatched entries are
excluded; an identifier with no matching entries raises `ValueError`.

The response contains the requested `service_id`, `window_from`, `window_to`,
and `metrics`. Nested DHCore `metrics` samples become `metric.values`, while
names, units, and summary statistics are preserved. Source metadata, quotas,
and other extra fields are omitted. No units are converted and no summaries
are recalculated.

When time bounds are provided, samples outside the inclusive window are removed.
Original summaries are replaced with `sum`, arithmetic `avg`, `max`, and `min`
computed from retained samples. Other source summary statistics are omitted
because they may no longer describe the selected window. Empty series have an
empty summary. Without bounds, original summaries are preserved unchanged.

```python
metrics = manager.collect_metrics(
	"949c8a888235496f875a1992bab1b32b",
	window_from=1791007803,
	window_to=1791011643,
)
```

Provided bounds are returned as the response window boundaries. Omitted bounds
are inferred from the minimum or maximum retained sample timestamp. Empty series
are preserved when other series provide the window. If no retained sample has
a timestamp and a bound must be inferred, the manager raises `ValueError` rather
than inventing a time window. With both bounds provided, an empty result uses
those bounds and contains empty values and summaries. Timestamps must be integer Unix timestamps;
measurement and summary values must be finite numbers.

This implementation does not contact DHCore, and `params` is currently ignored.
Missing files, invalid JSON, and malformed exports raise errors rather than
returning fabricated measurements.

Install the optional test dependencies and run the conversion tests, including
Draft 7 validation of the real fixture response:

```powershell
python -m pip install -e ".[test]"
python -m unittest discover -s tests -p test_dh_core_metrics_manager.py -v
```