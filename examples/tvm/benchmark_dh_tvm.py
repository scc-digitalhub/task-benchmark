from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from contextlib import contextmanager, nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any, Generator, TextIO
from urllib.parse import parse_qs, quote, unquote, urlsplit, urlunsplit
from urllib import request
from urllib import error

import torch
from torch import Tensor
from torch.nn import functional as F
import yaml

from task_benchmark import BenchmarkManager
from task_benchmark.impl.data_loader.image_classification import (
    ImageNet10ImageClassificationDataLoader,
)
from task_benchmark.impl.evaluator.image_classification import (
    Top1ImageClassificationEvaluator,
)
from task_benchmark.impl.invoker.base_invoker import Invoker
from task_benchmark.impl.invoker.image_classification import (
    OpenInferenceImageClassificationInvoker,
)
from task_benchmark.impl.invoker.open_inference import OpenInferenceInvoker
from task_benchmark.impl.metrics_manager import DHCoreMetricsManager
from task_benchmark.impl.deploy_manager import DHCoreDeployManager

MODEL_CLASSES = 1000
IMAGE_SIZE = 224
_RUN_ID_PATTERN = re.compile(
    r"(?i)(?<![a-z0-9])([a-z0-9]{32}|[0-9a-f]{8}-[0-9a-f]{4}-"
    r"[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})(?![a-z0-9])"
)
_TVMSERVE_RUN_ID_PATTERN = re.compile(
    r"(?:^|-)tvmserve-([a-z0-9]+)(?=\.)",
    re.IGNORECASE,
)
DEFAULT_DHCLI_PATH = Path(
    r"C:\Users\palan\Documents\Tools\dhcli\dhcli.exe"
)


def save_metrics(metrics: dict[str, Any]) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = Path.cwd() / "out"
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = 0
    while True:
        collision_suffix = "" if suffix == 0 else f"_{suffix}"
        metrics_path = output_dir / f"metrics_yolo_{timestamp}{collision_suffix}.json"
        try:
            with metrics_path.open("x", encoding="utf-8") as metrics_file:
                json.dump(metrics, metrics_file, indent=2)
                metrics_file.write("\n")
            return metrics_path
        except FileExistsError:
            suffix += 1


class BatchedImageInvoker(Invoker[Tensor, Tensor]):
    def __init__(self, invoker: OpenInferenceImageClassificationInvoker) -> None:
        self.invoker = invoker

    def invoke(self, test_data: Tensor) -> Tensor:
        if test_data.ndim != 3 or test_data.shape[0] != 3:
            raise ValueError("test_data must have RGB CHW shape")
        if not torch.isfinite(test_data).all().item():
            raise ValueError("test_data must contain finite values")
        if test_data.min() < 0 or test_data.max() > 1:
            raise ValueError("test_data values must be in [0, 1]")

        height, width = test_data.shape[-2:]
        scale = IMAGE_SIZE / min(height, width)
        resized_height = max(IMAGE_SIZE, round(height * scale))
        resized_width = max(IMAGE_SIZE, round(width * scale))
        resized = F.interpolate(
            test_data.unsqueeze(0),
            size=(resized_height, resized_width),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        top = (resized_height - IMAGE_SIZE) // 2
        left = (resized_width - IMAGE_SIZE) // 2
        pixels = resized[
            :,
            :,
            top : top + IMAGE_SIZE,
            left : left + IMAGE_SIZE,
        ]
        return self.invoker.invoke(pixels)


def class_labels(
    loader: ImageNet10ImageClassificationDataLoader,
) -> dict[int, str]:
    if len(loader.classes) != 10:
        raise ValueError("ImageNet10 must provide exactly 10 classes")

    mapping_path = os.getenv("OPEN_INFERENCE_LABELS_PATH")
    if mapping_path:
        raw_mapping = json.loads(Path(mapping_path).read_text(encoding="utf-8"))
        if not isinstance(raw_mapping, dict):
            raise ValueError("OPEN_INFERENCE_LABELS_PATH must contain a JSON object")
        labels: dict[int, str] = {}
        for class_id, label in raw_mapping.items():
            try:
                index = int(class_id)
            except (TypeError, ValueError) as exc:
                raise ValueError("Label mapping keys must be integer class indices") from exc
            if index in labels:
                raise ValueError(f"Duplicate class index {index} in label mapping")
            labels[index] = label
        if set(labels) != set(range(MODEL_CLASSES)):
            raise ValueError("Label mapping must contain all 1000 output class indices")
    else:
        labels = {
            index: f"__imagenet_class_{index}__"
            for index in range(MODEL_CLASSES)
        }
        labels.update(enumerate(loader.classes))

    if any(not isinstance(label, str) or not label.strip() for label in labels.values()):
        raise ValueError("Labels must contain non-empty strings")
    for index, expected_label in enumerate(loader.classes):
        if labels[index].strip().casefold() != expected_label.strip().casefold():
            raise ValueError(
                "The first 10 label indices must match the ImageNet10 class order"
            )
    return labels


def merge_dhcore_metrics(
    benchmark_metrics: dict[str, Any],
    run_id: str,
    metric_names: list[str]
) -> dict[str, Any]:
    window_from = benchmark_metrics.get("window_from")
    window_to = benchmark_metrics.get("window_to")
    benchmark_series = benchmark_metrics.get("metrics")
    if type(window_from) is not int or type(window_to) is not int:
        raise ValueError("Benchmark metrics must contain integer window bounds")
    if not isinstance(benchmark_series, list):
        raise ValueError("Benchmark metrics must contain a metrics list")

    dhcore_metrics = DHCoreMetricsManager().collect_metrics(
        run_id,
        window_from=window_from,
        window_to=window_to,
        metric_names=metric_names
    )
    if not isinstance(dhcore_metrics, dict):
        raise ValueError("DHCore metrics response must be an object")
    dhcore_series = dhcore_metrics.get("metrics")
    if not isinstance(dhcore_series, list):
        raise ValueError("DHCore metrics response must contain a metrics list")

    combined_metrics = dict(benchmark_metrics)
    combined_metrics["metrics"] = [*benchmark_series, *dhcore_series]
    return combined_metrics


def extract_run_id(service_url: str) -> str:
    """Extract a DigitalHub run ID embedded in a service URL."""
    parsed_url = urlsplit(_add_http_scheme(service_url))
    if not parsed_url.scheme or not parsed_url.netloc:
        raise ValueError("Service URL must be an absolute URL")

    hostname = parsed_url.hostname or ""
    tvmserve_match = _TVMSERVE_RUN_ID_PATTERN.search(hostname)
    if tvmserve_match:
        return tvmserve_match.group(1)

    for key, values in parse_qs(parsed_url.query).items():
        if key.casefold() in {"run_id", "runid"} and values and values[0].strip():
            return values[0].strip()

    path_segments = [
        unquote(segment)
        for segment in parsed_url.path.split("/")
        if segment
    ]
    for index, segment in enumerate(path_segments[:-1]):
        if segment.casefold() in {"run", "runs"}:
            run_id = path_segments[index + 1].strip()
            if run_id:
                return run_id

    components = [*hostname.split("."), *path_segments]
    for component in components:
        match = _RUN_ID_PATTERN.search(component)
        if match:
            return match.group(1)

    raise ValueError(
        "Could not extract a DigitalHub run ID from the deployed service URL"
    )


def open_inference_base_url(service_url: str, served_name: str) -> str:
    """Normalize a deployed service URL to its Open Inference model metadata URL."""
    parsed_url = urlsplit(_add_http_scheme(service_url))
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
        raise ValueError("Service URL must use HTTP(S) and contain a host")
    if parsed_url.query or parsed_url.fragment:
        raise ValueError("Service URL must not contain a query or fragment")
    if not isinstance(served_name, str) or not served_name.strip():
        raise ValueError("served_name must not be empty")

    path = parsed_url.path.rstrip("/")
    if path.endswith("/infer"):
        path = path.removesuffix("/infer")
    path_segments = [unquote(segment) for segment in path.split("/") if segment]
    has_model_route = (
        len(path_segments) >= 3
        and path_segments[-3] == "v2"
        and path_segments[-2] == "models"
    )
    if not has_model_route:
        model_path = f"/v2/models/{quote(served_name, safe='')}"
        path = f"{path}{model_path}"

    return urlunsplit((parsed_url.scheme, parsed_url.netloc, path, "", ""))


def _add_http_scheme(service_url: str) -> str:
    if not isinstance(service_url, str) or not service_url.strip():
        raise ValueError("Service URL must not be empty")
    service_url = service_url.strip()
    if "://" in service_url:
        return service_url
    return f"http://{service_url}"


def resolve_dhcli_path() -> Path:
    candidates: list[Path] = []
    configured_path = os.getenv("DHCLI_PATH")
    if configured_path:
        candidates.append(Path(configured_path))
    configured_home = os.getenv("DHCLI_HOME")
    if configured_home:
        configured_path = Path(configured_home)
        candidates.append(
            configured_path / "dhcli.exe"
            if configured_path.is_dir() or configured_path.suffix == ""
            else configured_path
        )
    candidates.append(DEFAULT_DHCLI_PATH)
    path_command = shutil.which("dhcli")
    if path_command:
        candidates.append(Path(path_command))

    for candidate in candidates:
        executable = candidate / "dhcli.exe" if candidate.is_dir() else candidate
        if executable.is_file():
            return executable
    raise FileNotFoundError(
        "Could not find dhcli.exe; set DHCLI_PATH or DHCLI_HOME "
        "or add dhcli to PATH"
    )


@contextmanager
def port_forward(
    project: str,
    run_id: str,
    local_port: int,
) -> Generator[tuple[subprocess.Popen[str], TextIO], None, None]:
    executable = resolve_dhcli_path()
    command = [
        str(executable),
        "port-forward",
        "-p",
        project,
        run_id,
        "-l",
        str(local_port),
    ]
    print(
        f"Starting dhcli port-forward for project '{project}', "
        f"run '{run_id}' on local port {local_port}"
    )
    with tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as output:
        process = subprocess.Popen(
            command,
            stdout=output,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            yield process, output
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


def _read_port_forward_output(output: TextIO) -> str:
    output.flush()
    output.seek(0)
    return output.read().strip()


def wait_for_port_forward(
    base_url: str,
    process: subprocess.Popen[str],
    output: TextIO,
    timeout_seconds: float = 60.0,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None
    while True:
        exit_code = process.poll()
        if exit_code is not None:
            details = _read_port_forward_output(output)
            suffix = f": {details}" if details else ""
            raise RuntimeError(
                f"dhcli port-forward exited with code {exit_code}{suffix}"
            )

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            details = _read_port_forward_output(output)
            suffix = f": {details}" if details else ""
            raise TimeoutError(
                f"Timed out after {timeout_seconds:g} seconds waiting for "
                f"the port-forward at {base_url}{suffix}"
            ) from last_error

        try:
            with request.urlopen(base_url, timeout=min(2.0, remaining)) as response:
                metadata = json.load(response)
            if not isinstance(metadata, dict):
                raise ValueError("Open Inference metadata must be a JSON object")
            return metadata
        except error.HTTPError as exc:
            if exc.code not in {502, 503, 504}:
                raise RuntimeError(
                    f"Open Inference metadata endpoint returned HTTP {exc.code}"
                ) from exc
            last_error = exc
        except (error.URLError, TimeoutError, ConnectionError) as exc:
            last_error = exc

        remaining = deadline - time.monotonic()
        if remaining > 0:
            time.sleep(min(0.5, remaining))


def benchmark_service(
    service_url: str,
    project: str,
    profile: str,
    target_architecture: str,
    served_name: str,
    max_samples: int,
    local_port: int,
    use_port_forward: bool = True,
) -> Path:
    run_id = extract_run_id(service_url)
    if use_port_forward:
        base_url = open_inference_base_url(
            f"http://localhost:{local_port}",
            served_name,
        )
        forward_context = port_forward(project, run_id, local_port)
    else:
        base_url = open_inference_base_url(service_url, served_name)
        forward_context = nullcontext(None)

    with forward_context as forwarding:
        if use_port_forward:
            if forwarding is None:
                raise RuntimeError("Port-forward context did not start")
            process, output = forwarding
            metadata = wait_for_port_forward(base_url, process, output)
        else:
            with request.urlopen(base_url, timeout=60.0) as response:
                metadata = json.load(response)
            if not isinstance(metadata, dict):
                raise ValueError("Open Inference metadata must be a JSON object")

        expected_input = {
            "name": "images",
            "datatype": "FP32",
            "shape": [1, 3, IMAGE_SIZE, IMAGE_SIZE],
        }
        expected_output = {
            "name": "output0",
            "datatype": "FP32",
            "shape": [1, MODEL_CLASSES],
        }
        if metadata.get("inputs") != [expected_input]:
            raise ValueError("Expected the YOLO FP32 input [1, 3, 224, 224]")
        if metadata.get("outputs") != [expected_output]:
            raise ValueError("Expected the YOLO output0 with 1000 FP32 class scores")

        loader = ImageNet10ImageClassificationDataLoader(
            local_path=os.getenv("IMAGENET10_CACHE_DIR", ".cache/imagenet10"),
            split=os.getenv("IMAGENET10_SPLIT", "val"),
            max_samples=max_samples,
        )
        labels = class_labels(loader)
        client = OpenInferenceInvoker(f"{base_url}/infer")
        classifier = OpenInferenceImageClassificationInvoker(
            client,
            input_name="images",
            output_name="output0",
            labels=labels,
        )
        evaluator = Top1ImageClassificationEvaluator(loader, classifier)
        try:
            benchmark_metrics = BenchmarkManager(
                data_loader=loader,
                invoker=BatchedImageInvoker(classifier),
                evaluator=evaluator,
                service_id=base_url,
            ).run_benchmark()
            if not isinstance(benchmark_metrics, dict):
                raise ValueError("Benchmark metrics response must be an object")
            metric_names = ["cpu", "memory"]
            metrics = merge_dhcore_metrics(benchmark_metrics, run_id, metric_names)
            metrics["profile"] = profile
            metrics["target_architecture"] = target_architecture
            metrics["run_id"] = run_id
            print(json.dumps(metrics, indent=2))
            metrics_path = save_metrics(metrics)
            print(f"Metrics saved to: {metrics_path}")
            return metrics_path
        finally:
            classifier.close()


def benchmark_deployments(
    deployments: list[tuple[str, str, str]],
    config: dict[str, Any],
    max_samples: int,
    local_port: int,
    use_port_forward: bool = True,
) -> list[Path]:
    nodes = config.get("nodes")
    default_served_name = config.get("served_name")
    project = config.get("project")
    if not isinstance(nodes, list) or len(nodes) != len(deployments):
        raise ValueError("Deployment results must correspond to YAML nodes")
    if not isinstance(default_served_name, str) or not default_served_name.strip():
        raise ValueError("TVM deployment config requires a non-empty served_name")
    if not isinstance(project, str) or not project.strip():
        raise ValueError("TVM deployment config requires a non-empty project")
    if use_port_forward and (
        not 1 <= local_port <= 65535
        or local_port + len(deployments) - 1 > 65535
    ):
        raise ValueError("Local port range must be between 1 and 65535")

    metrics_paths: list[Path] = []
    for index, ((profile, architecture, service_url), node) in enumerate(
        zip(deployments, nodes)
    ):
        if not isinstance(node, dict):
            raise ValueError("Each TVM node must be a mapping")
        if (
            node.get("profile") != profile
            or node.get("target_architecture") != architecture
        ):
            raise ValueError("Deployment results do not match the YAML node order")
        served_name = node.get("served_name", default_served_name)
        if not isinstance(served_name, str) or not served_name.strip():
            raise ValueError("TVM node served_name must be a non-empty string")

        metrics_path = benchmark_service(
            service_url=service_url,
            project=project.strip(),
            profile=profile,
            target_architecture=architecture,
            served_name=served_name.strip(),
            max_samples=max_samples,
            local_port=local_port + index,
            use_port_forward=use_port_forward,
        )
        metrics_paths.append(metrics_path)
    return metrics_paths


def main() -> None:
    config_path = Path(__file__).resolve().with_name("dev.yml")
    with config_path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise ValueError("TVM deployment YAML must contain a mapping")

    max_samples = int(os.getenv("IMAGENET10_MAX_SAMPLES", "12"))
    if max_samples < 1:
        raise ValueError("IMAGENET10_MAX_SAMPLES must be greater than zero")
    local_port = int(os.getenv("OPEN_INFERENCE_LOCAL_PORT", "8080"))
    use_port_forward = _environment_flag(
        "OPEN_INFERENCE_USE_PORT_FORWARD",
        default=True,
    )

    deployments = DHCoreDeployManager().deployTVM(config_path)
    if not deployments:
        raise RuntimeError("TVM deployment returned no services to benchmark")
    benchmark_deployments(
        deployments,
        config,
        max_samples,
        local_port,
        use_port_forward,
    )


def _environment_flag(name: str, *, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().casefold()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value")


if __name__ == "__main__":
    main()
