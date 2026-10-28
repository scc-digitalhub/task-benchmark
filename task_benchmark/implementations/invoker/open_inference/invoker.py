# SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, overload
from urllib import error, parse, request

import torch
from torch import Tensor

from task_benchmark.implementations.invoker.base_invoker import Invoker


_TORCH_DTYPES: dict[str, torch.dtype] = {
    "BOOL": torch.bool,
    "UINT8": torch.uint8,
    "INT8": torch.int8,
    "INT16": torch.int16,
    "INT32": torch.int32,
    "INT64": torch.int64,
    "FP16": torch.float16,
    "FP32": torch.float32,
    "FP64": torch.float64,
}


class OpenInferenceInvoker(
    Invoker[
        Tensor | Mapping[str, Tensor],
        Tensor | dict[str, Tensor],
    ]
):
    """Invoke a single-input/output model using the Open Inference v2 protocol.

    ``service_url`` must include the model inference path, for example
    ``http://localhost:8000/v2/models/my-model/infer`` or
    ``grpc://localhost:8001/v2/models/my-model/infer``.
    """

    def __init__(self, service_url: str) -> None:
        parsed_url = parse.urlsplit(service_url)
        if parsed_url.scheme not in {"http", "https", "grpc", "grpcs"}:
            raise ValueError(
                "service_url must use http, https, grpc, or grpcs"
            )
        if not parsed_url.netloc or parsed_url.query or parsed_url.fragment:
            raise ValueError("service_url must contain a host and no query or fragment")

        path_parts = [
            parse.unquote(part)
            for part in parsed_url.path.split("/")
            if part
        ]
        if (
            len(path_parts) != 4
            or path_parts[0] != "v2"
            or path_parts[1] != "models"
            or path_parts[3] != "infer"
            or not path_parts[2]
        ):
            raise ValueError(
                "service_url path must be /v2/models/{model_name}/infer"
            )

        self._scheme = parsed_url.scheme
        self._authority = parsed_url.netloc
        self._model_name = path_parts[2]
        self._http_endpoint = service_url.rstrip("/")
        self._http_metadata_endpoint = self._http_endpoint.removesuffix("/infer")
        self._metadata: dict[str, Any] | None = None
        self._grpc_client: Any = None

        if self._scheme in {"grpc", "grpcs"}:
            try:
                from tritonclient.grpc import InferenceServerClient
            except ImportError as exc:
                raise ImportError(
                    "gRPC invocation requires the 'tritonclient[grpc]' package"
                ) from exc

            self._grpc_client = InferenceServerClient(
                url=self._authority,
                ssl=self._scheme == "grpcs",
            )

    @overload
    def invoke(self, test_data: Tensor) -> Tensor: ...

    @overload
    def invoke(self, test_data: Mapping[str, Tensor]) -> dict[str, Tensor]: ...

    def invoke(
        self,
        test_data: Tensor | Mapping[str, Tensor],
    ) -> Tensor | dict[str, Tensor]:
        """Invoke a model with one tensor or a mapping of named tensors.

        The single-tensor form requires a model with one input and returns its
        first output. The mapping form accepts all named inputs and returns all
        named outputs.
        """
        metadata = self._get_metadata()
        input_metadata = {
            item["name"]: item for item in metadata.get("inputs", [])
        }
        output_metadata = metadata.get("outputs", [])
        if not output_metadata:
            raise RuntimeError("OpenInference model metadata contains no outputs")

        single_tensor = isinstance(test_data, Tensor)
        if single_tensor:
            if len(input_metadata) != 1:
                raise RuntimeError(
                    "The single-tensor invoke form requires a model with "
                    f"one input, found {len(input_metadata)}"
                )
            input_name = next(iter(input_metadata))
            provided_inputs = {input_name: test_data}
            selected_outputs = output_metadata[:1]
        elif isinstance(test_data, Mapping):
            provided_inputs = dict(test_data)
            missing_inputs = input_metadata.keys() - provided_inputs.keys()
            unexpected_inputs = provided_inputs.keys() - input_metadata.keys()
            if missing_inputs or unexpected_inputs:
                details = []
                if missing_inputs:
                    details.append(f"missing: {', '.join(sorted(missing_inputs))}")
                if unexpected_inputs:
                    details.append(
                        f"unexpected: {', '.join(sorted(unexpected_inputs))}"
                    )
                raise ValueError(
                    "Input names do not match model metadata ("
                    + "; ".join(details)
                    + ")"
                )
            selected_outputs = output_metadata
        else:
            raise TypeError("test_data must be a torch.Tensor or a mapping of tensors")

        prepared_inputs: dict[str, Tensor] = {}
        for input_name, tensor in provided_inputs.items():
            if not isinstance(tensor, Tensor):
                raise TypeError(f"Input '{input_name}' must be a torch.Tensor")
            prepared_inputs[input_name] = tensor.detach().to(
                device="cpu",
                dtype=self._torch_dtype(input_metadata[input_name]["datatype"]),
            ).contiguous()

        if self._grpc_client is not None:
            results = self._invoke_grpc(
                prepared_inputs,
                input_metadata,
                selected_outputs,
            )
        else:
            results = self._invoke_http(
                prepared_inputs,
                input_metadata,
                selected_outputs,
            )

        if single_tensor:
            return results[selected_outputs[0]["name"]]
        return results

    def close(self) -> None:
        """Close the gRPC channel, if this invoker uses gRPC."""
        if self._grpc_client is not None:
            self._grpc_client.close()

    def _get_metadata(self) -> dict[str, Any]:
        if self._metadata is not None:
            return self._metadata

        if self._grpc_client is not None:
            self._metadata = self._grpc_client.get_model_metadata(
                self._model_name
            )
            return self._metadata

        http_request = request.Request(self._http_metadata_endpoint, method="GET")
        try:
            with request.urlopen(http_request, timeout=60.0) as response:
                self._metadata = json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"OpenInference server returned HTTP {exc.code}: {body}"
            ) from exc
        except (error.URLError, TimeoutError) as exc:
            raise RuntimeError(
                f"Could not retrieve model metadata from "
                f"{self._http_metadata_endpoint}: {exc}"
            ) from exc
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "OpenInference server returned invalid JSON metadata"
            ) from exc

        return self._metadata

    def _invoke_http(
        self,
        test_data: Mapping[str, Tensor],
        input_metadata: Mapping[str, dict[str, Any]],
        output_metadata: list[dict[str, Any]],
    ) -> dict[str, Tensor]:
        payload = {
            "inputs": [
                {
                    "name": input_name,
                    "shape": list(tensor.shape),
                    "datatype": input_metadata[input_name]["datatype"],
                    "data": tensor.reshape(-1).tolist(),
                }
                for input_name, tensor in test_data.items()
            ],
            "outputs": [{"name": output["name"]} for output in output_metadata],
        }
        http_request = request.Request(
            self._http_endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(http_request, timeout=60.0) as response:
                result = json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"OpenInference server returned HTTP {exc.code}: {body}"
            ) from exc
        except (error.URLError, TimeoutError) as exc:
            raise RuntimeError(
                f"Could not invoke OpenInference server at "
                f"{self._http_endpoint}: {exc}"
            ) from exc
        except json.JSONDecodeError as exc:
            raise RuntimeError("OpenInference server returned invalid JSON") from exc

        result_outputs = {
            item.get("name"): item
            for item in result.get("outputs", [])
            if isinstance(item, dict)
        }
        tensors: dict[str, Tensor] = {}
        for output_metadata_item in output_metadata:
            output_name = output_metadata_item["name"]
            output = result_outputs.get(output_name)
            if output is None or "data" not in output or "shape" not in output:
                raise RuntimeError(
                    f"OpenInference response contains no output '{output_name}'"
                )
            tensors[output_name] = torch.tensor(
                output["data"],
                dtype=self._torch_dtype(output["datatype"]),
            ).reshape(output["shape"])

        return tensors

    def _invoke_grpc(
        self,
        test_data: Mapping[str, Tensor],
        input_metadata: Mapping[str, dict[str, Any]],
        output_metadata: list[dict[str, Any]],
    ) -> dict[str, Tensor]:
        from tritonclient.grpc import InferInput, InferRequestedOutput

        grpc_inputs = []
        for input_name, tensor in test_data.items():
            grpc_input = InferInput(
                input_name,
                list(tensor.shape),
                input_metadata[input_name]["datatype"],
            )
            grpc_input.set_data_from_numpy(tensor.numpy())
            grpc_inputs.append(grpc_input)
        grpc_outputs = [
            InferRequestedOutput(output["name"])
            for output in output_metadata
        ]
        result = self._grpc_client.infer(
            self._model_name,
            grpc_inputs,
            outputs=grpc_outputs,
        )
        tensors: dict[str, Tensor] = {}
        for output_metadata_item in output_metadata:
            output_name = output_metadata_item["name"]
            output = result.as_numpy(output_name)
            if output is None:
                raise RuntimeError(
                    f"OpenInference response contains no output '{output_name}'"
                )
            tensors[output_name] = torch.from_numpy(output.copy())
        return tensors

    @staticmethod
    def _torch_dtype(datatype: str) -> torch.dtype:
        try:
            return _TORCH_DTYPES[datatype]
        except KeyError as exc:
            raise TypeError(
                f"OpenInference datatype '{datatype}' is not supported"
            ) from exc