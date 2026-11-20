from __future__ import annotations

import json
import logging
import math
import os
import re
import shutil
import subprocess
import tempfile
import time
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator, TextIO
from urllib import error, request
from urllib.parse import parse_qs, quote, unquote, urlsplit, urlunsplit

import digitalhub as dh
import yaml

logger = logging.getLogger(__name__)

_RUN_ID_PATTERN = re.compile(
    r"(?i)(?<![a-z0-9])([a-z0-9]{32}|[0-9a-f]{8}-[0-9a-f]{4}-"
    r"[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})(?![a-z0-9])"
)
_TVMSERVE_RUN_ID_PATTERN = re.compile(
    r"(?:^|-)tvmserve-([a-z0-9]+)(?=\.)",
    re.IGNORECASE,
)


class DHCoreDeployManager:
    """Create runs from a DHCore run using each requested deployment profile."""

    def stop_run(self, project: str, run_id: str) -> None:
        """Stop a DigitalHub run by project and run ID."""
        if not isinstance(project, str) or not project.strip():
            raise ValueError("project must be a non-empty string")
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError("run_id must be a non-empty string")

        dh.get_run(run_id.strip(), project=project.strip()).stop()

    def deploy(
        self,
        id_project: str,
        id_run: str,
        profiles: list[str],
    ) -> list[str]:
        if not isinstance(id_project, str) or not id_project.strip():
            raise ValueError("id_project must be a non-empty string")
        if not isinstance(id_run, str) or not id_run.strip():
            raise ValueError("id_run must be a non-empty string")
        if not isinstance(profiles, list):
            raise TypeError("profiles must be a list of strings")
        if any(not isinstance(profile, str) or not profile.strip() for profile in profiles):
            raise ValueError("profiles must contain non-empty strings")

        id_project = id_project.strip()
        id_run = id_run.strip()
        profiles = [profile.strip() for profile in profiles]
        if not profiles:
            return []

        source_run = dh.get_run(id_run, project=id_project)
        run_spec = source_run.spec.to_dict()
        function_reference = run_spec.get("function")
        if not isinstance(function_reference, str) or not function_reference:
            raise ValueError("The source run does not reference a function")

        function_name, function_version = self._function_identity(function_reference)
        function = dh.get_function(
            function_name,
            project=id_project,
            entity_id=function_version or None,
        )

        action = source_run.kind.rpartition("+")[2]
        if not action:
            raise ValueError(f"Cannot determine run action from kind '{source_run.kind}'")

        run_parameters = {
            key: value
            for key, value in run_spec.items()
            if key not in {"task", "function", "workflow", "profile"}
        }
        extensions = getattr(source_run, "extensions", None)

        service_urls: list[str] = []
        for profile in profiles:
            cloned_run = function.run(
                action,
                wait=True,
                log_info=False,
                extensions=extensions,
                profile=profile,
                **run_parameters,
            )
            service = getattr(cloned_run.status, "service", None)
            service_url = service.get("url") if isinstance(service, dict) else None
            if not isinstance(service_url, str) or not service_url.strip():
                raise RuntimeError(
                    f"Run created with profile '{profile}' did not expose a service URL"
                )
            service_urls.append(service_url.strip())

        return service_urls

    @staticmethod
    def _function_identity(reference: str) -> tuple[str, str]:
        function_path = reference.split("://", 1)[-1].rsplit("/", 1)[-1]
        function_name, separator, function_version = function_path.rpartition(":")
        if not separator:
            function_name = function_path
            function_version = ""
        if not function_name:
            raise ValueError("The source run has an invalid function reference")
        return function_name, function_version

    @staticmethod
    def extract_run_id(service_url: str) -> str:
        """Extract a DigitalHub run ID embedded in a service URL."""
        parsed_url = urlsplit(DHCoreDeployManager._add_http_scheme(service_url))
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

    @staticmethod
    def open_inference_base_url(service_url: str, served_name: str) -> str:
        """Normalize a service URL to its Open Inference model metadata URL."""
        parsed_url = urlsplit(DHCoreDeployManager._add_http_scheme(service_url))
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

    @contextmanager
    def open_inference_service(
        self,
        service_url: str,
        *,
        project: str,
        served_name: str,
        local_port: int = 8080,
        use_port_forward: bool = True,
        timeout_seconds: float = 60.0,
    ) -> Generator[tuple[str, dict[str, Any]], None, None]:
        """Yield an Open Inference metadata URL and its metadata, managing forwarding."""
        if not isinstance(project, str) or not project.strip():
            raise ValueError("project must be a non-empty string")
        if type(use_port_forward) is not bool:
            raise TypeError("use_port_forward must be a bool")
        if use_port_forward and (
            type(local_port) is not int or not 1 <= local_port <= 65535
        ):
            raise ValueError("local_port must be an integer between 1 and 65535")
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be a positive finite number")

        if use_port_forward:
            run_id = self.extract_run_id(service_url)
            base_url = self.open_inference_base_url(
                f"http://localhost:{local_port}",
                served_name,
            )
            with self._port_forward(project.strip(), run_id, local_port) as (
                process,
                output,
            ):
                metadata = self._wait_for_service_metadata(
                    base_url,
                    process,
                    output,
                    timeout_seconds,
                )
                yield base_url, metadata
            return

        base_url = self.open_inference_base_url(service_url, served_name)
        metadata = self._get_service_metadata(base_url)
        yield base_url, metadata

    @staticmethod
    def _add_http_scheme(service_url: str) -> str:
        if not isinstance(service_url, str) or not service_url.strip():
            raise ValueError("Service URL must not be empty")
        service_url = service_url.strip()
        if "://" in service_url:
            return service_url
        return f"http://{service_url}"

    @staticmethod
    def _resolve_dhcli_path() -> Path:
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
    def _port_forward(
        self,
        project: str,
        run_id: str,
        local_port: int,
    ) -> Generator[tuple[subprocess.Popen[str], TextIO], None, None]:
        executable = self._resolve_dhcli_path()
        command = [
            str(executable),
            "port-forward",
            "-p",
            project,
            run_id,
            "-l",
            str(local_port),
        ]
        logger.info(
            "Starting dhcli port-forward for project '%s', run '%s' "
            "on local port %s",
            project,
            run_id,
            local_port,
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

    @staticmethod
    def _read_port_forward_output(output: TextIO) -> str:
        output.flush()
        output.seek(0)
        return output.read().strip()

    @staticmethod
    def _get_service_metadata(
        base_url: str,
        timeout_seconds: float = 60.0,
    ) -> dict[str, Any]:
        with request.urlopen(base_url, timeout=timeout_seconds) as response:
            metadata = json.load(response)
        if not isinstance(metadata, dict):
            raise ValueError("Open Inference metadata must be a JSON object")
        return metadata

    def _wait_for_service_metadata(
        self,
        base_url: str,
        process: subprocess.Popen[str],
        output: TextIO,
        timeout_seconds: float,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_seconds
        last_error: Exception | None = None
        while True:
            exit_code = process.poll()
            if exit_code is not None:
                details = self._read_port_forward_output(output)
                suffix = f": {details}" if details else ""
                raise RuntimeError(
                    f"dhcli port-forward exited with code {exit_code}{suffix}"
                )

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                details = self._read_port_forward_output(output)
                suffix = f": {details}" if details else ""
                raise TimeoutError(
                    f"Timed out after {timeout_seconds:g} seconds waiting for "
                    f"the port-forward at {base_url}{suffix}"
                ) from last_error

            try:
                return self._get_service_metadata(
                    base_url,
                    timeout_seconds=min(2.0, remaining),
                )
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

    def deployTVM(
        self,
        yml_file: str | Path,
        timeout_seconds: float = 30.0,
    ) -> list[tuple[str, str, str]]:
        """Build, compile, and serve a TVM model using the supplied YAML plan."""
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be a positive finite number")

        yml_path = Path(yml_file).expanduser()
        with yml_path.open(encoding="utf-8") as source:
            config = yaml.safe_load(source)
        if not isinstance(config, dict):
            raise ValueError("TVM deployment YAML must contain a mapping")

        project_name = self._required_text(config, "project")
        function_name = self._required_text(config, "function")
        model = self._required_text(config, "model")
        served_name = self._required_text(config, "served_name")
        build_resources = self._resource_mapping(config, "build-resources")
        targets = self._required_list(config, "targets")
        nodes = self._required_list(config, "nodes")

        project = dh.get_project(project_name)
        logger.info(
            "Creating TVM main function '%s' in project '%s'",
            function_name,
            project_name,
        )
        main_func = dh.new_function(
            project=project.name,
            name=function_name,
            kind="tvm",
            format="onnx",
            model=model,
        )
        logger.info("Created TVM main function '%s'", function_name)

        logger.info("Starting TVM build for function '%s'", function_name)
        build_run = main_func.run(
            "build",
            resources=build_resources,
            wait=True,
            log_info=False,
        )
        self._raise_if_run_failed(build_run, "build")
        logger.info("TVM build completed for function '%s'", function_name)

        so_models: dict[str, str] = {}
        for target in targets:
            if not isinstance(target, dict):
                raise ValueError("Each TVM target must be a mapping")
            architecture = self._required_text(target, "target_architecture")
            if architecture in so_models:
                raise ValueError(f"Duplicate TVM target architecture '{architecture}'")
            resources = self._resource_mapping(
                target,
                "compile-resources",
                "complile-resources",
            )
            logger.info("Starting TVM compile for target '%s'", architecture)
            compile_run = main_func.run(
                "compile",
                target_architecture=architecture,
                tag=architecture,
                resources=resources,
                wait=True,
                log_info=False,
            )
            self._raise_if_run_failed(compile_run, f"compile ({architecture})")
            so_model = self._compiled_model_reference(compile_run)
            if not so_model:
                so_model = self._model_reference(getattr(compile_run, "spec", None))
            if not so_model:
                main_func.refresh()
                so_model = self._model_reference(main_func.spec)
            if not so_model:
                raise RuntimeError(
                    f"TVM compile run for '{architecture}' did not produce so_model metadata"
                )
            so_models[architecture] = so_model
            logger.info(
                "TVM compile completed for target '%s'; model: %s",
                architecture,
                so_model,
            )

        deployments: list[tuple[str, str, str]] = []
        for node in nodes:
            if not isinstance(node, dict):
                raise ValueError("Each TVM node must be a mapping")
            architecture = self._required_text(node, "target_architecture")
            profile = self._required_text(node, "profile")
            try:
                model_path = so_models[architecture]
            except KeyError as exc:
                raise ValueError(
                    f"No compiled so_model found for target architecture '{architecture}'"
                ) from exc
            node_served_name = node.get("served_name", served_name)
            if not isinstance(node_served_name, str) or not node_served_name.strip():
                raise ValueError("TVM node served_name must be a non-empty string")
            logger.info(
                "Starting TVM serve for profile '%s', target '%s', model '%s'",
                profile,
                architecture,
                node_served_name.strip(),
            )
            service_run = main_func.run(
                "serve",
                profile=profile,
                served_name=node_served_name.strip(),
                model_path=model_path,
                resources=self._resource_mapping(node, "run-resources"),
                wait=False,
                log_info=False,
            )
            self._wait_for_service_run(
                service_run,
                f"serve ({profile}, {architecture})",
                timeout_seconds,
            )
            service = getattr(service_run.status, "service", None)
            service_url = service.get("url") if isinstance(service, Mapping) else None
            if not isinstance(service_url, str) or not service_url.strip():
                raise RuntimeError(
                    f"TVM service for target architecture '{architecture}' "
                    "did not expose a service URL"
                )
            deployments.append((profile, architecture, service_url.strip()))
            logger.info(
                "TVM service is running for profile '%s', target '%s': %s",
                profile,
                architecture,
                service_url.strip(),
            )

        return deployments

    @staticmethod
    def _raise_if_run_failed(run: Any, action: str) -> None:
        refresh = getattr(run, "refresh", None)
        if not callable(refresh):
            raise RuntimeError(f"Cannot refresh DigitalHub {action} run status")
        refresh()

        status = getattr(run, "status", None)
        state = getattr(status, "state", None)
        state = getattr(state, "value", state)
        if state is None:
            raise RuntimeError(f"Cannot determine DigitalHub {action} run state")
        if str(state).upper() == "ERROR":
            DHCoreDeployManager._raise_run_error(run, action, "ERROR")

    @staticmethod
    def _wait_for_service_run(run: Any, action: str, timeout_seconds: float) -> None:
        refresh = getattr(run, "refresh", None)
        if not callable(refresh):
            raise RuntimeError(f"Cannot refresh DigitalHub {action} run status")
        deadline = time.monotonic() + timeout_seconds
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"Timed out after {timeout_seconds:g} seconds waiting for "
                    f"DigitalHub {action} run to reach RUNNING"
                )

            refresh()
            status = getattr(run, "status", None)
            state = getattr(status, "state", None)
            state = getattr(state, "value", state)
            if state is None:
                raise RuntimeError(f"Cannot determine DigitalHub {action} run state")
            state_name = str(state).upper()
            if state_name == "ERROR":
                DHCoreDeployManager._raise_run_error(run, action, state_name)
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Timed out after {timeout_seconds:g} seconds waiting for "
                    f"DigitalHub {action} run to reach RUNNING"
                )
            if state_name == "RUNNING":
                return

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"Timed out after {timeout_seconds:g} seconds waiting for "
                    f"DigitalHub {action} run to reach RUNNING"
                )
            time.sleep(min(1.0, remaining))

    @staticmethod
    def _raise_run_error(run: Any, action: str, state: str) -> None:
        status = getattr(run, "status", None)
        message = getattr(status, "message", None)
        run_id = getattr(run, "id", None)
        run_reference = f" {run_id}" if run_id else ""
        detail = f": {message}" if isinstance(message, str) and message else ""
        raise RuntimeError(
            f"DigitalHub {action} run{run_reference} failed with state {state}{detail}"
        )

    @staticmethod
    def _required_text(data: dict[str, Any], key: str) -> str:
        value = data.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"TVM deployment config requires non-empty '{key}'")
        return value.strip()

    @classmethod
    def _required_list(cls, data: dict[str, Any], key: str) -> list[Any]:
        value = data.get(key)
        if not isinstance(value, list):
            raise ValueError(f"TVM deployment config requires '{key}' to be a list")
        return value

    @staticmethod
    def _resource_mapping(
        data: dict[str, Any],
        key: str,
        legacy_key: str | None = None,
    ) -> dict[str, Any]:
        resources = data.get(key)
        if resources is None and legacy_key is not None:
            resources = data.get(legacy_key)
        if not isinstance(resources, dict):
            raise ValueError(f"TVM deployment config requires '{key}' to be a mapping")
        return resources

    @classmethod
    def _compiled_model_reference(cls, run: Any) -> str | None:
        metadata = getattr(run, "metadata", None)
        if isinstance(metadata, Mapping):
            return cls._model_reference(metadata)
        to_dict = getattr(metadata, "to_dict", None)
        if callable(to_dict):
            metadata_dict = to_dict()
            if isinstance(metadata_dict, Mapping):
                model_reference = cls._model_reference(metadata_dict)
                if model_reference:
                    return model_reference
        return None

    @staticmethod
    def _model_reference(data: Any) -> str | None:
        if isinstance(data, Mapping):
            model_reference = data.get("so_model")
        else:
            model_reference = getattr(data, "so_model", None)
            if not isinstance(model_reference, str):
                to_dict = getattr(data, "to_dict", None)
                if callable(to_dict):
                    model_data = to_dict()
                    if isinstance(model_data, Mapping):
                        model_reference = model_data.get("so_model")
        if isinstance(model_reference, str) and model_reference.strip():
            return model_reference.strip()
        return None
