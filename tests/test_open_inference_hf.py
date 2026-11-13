from __future__ import annotations

import importlib.util
import json
import os
import tempfile
import threading
import unittest
from http.server import HTTPServer
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch


EXAMPLE_DIR = (Path(__file__).resolve().parents[1] / "examples" /
               "image_classification" / "open_inference_simple")


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class OpenInferenceServerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.server = load_module("numeric_example_server", EXAMPLE_DIR / "server.py")

    def test_numeric_contract(self) -> None:
        logits = torch.tensor([[0.1, 0.9]])
        model = Mock(return_value=SimpleNamespace(logits=logits))
        model.device = torch.device("cpu")
        model.config = SimpleNamespace(num_labels=2, id2label={0: "cat", 1: "dog"})
        processor = Mock(return_value={"pixel_values": torch.zeros(1, 3, 32, 32)})
        self.server.classifier = SimpleNamespace(model=model, image_processor=processor)
        payload = {"inputs": [{"name": "pixel_values", "datatype": "FP32",
                                "shape": [1, 3, 2, 2], "data": [0.5] * 12}]}

        result = self.server.infer_tensor(payload)

        metadata = self.server.model_metadata()
        self.assertEqual(metadata["inputs"][0]["datatype"], "FP32")
        self.assertEqual(set(metadata), {"name", "inputs", "outputs"})
        self.assertEqual(result["outputs"][0]["shape"], [1, 2])
        self.assertEqual(result["outputs"][0]["datatype"], "FP32")
        self.assertEqual(result["outputs"][0]["data"], logits.flatten().tolist())
        self.assertFalse(processor.call_args.kwargs["do_rescale"])

    def test_existing_base64_request_still_works(self) -> None:
        from PIL import Image
        import base64

        image_bytes = BytesIO()
        Image.new("RGB", (2, 2)).save(image_bytes, format="PNG")
        payload = json.dumps({"inputs": [{"name": "image", "data": [
            base64.b64encode(image_bytes.getvalue()).decode("ascii")
        ]}]}).encode()
        self.server.classifier = Mock(return_value=[[{"label": "cat", "score": 0.9}]])
        handler = object.__new__(self.server.Handler)
        handler.path = f"/v2/models/{self.server.MODEL_NAME}/infer"
        handler.headers = {"Content-Length": str(len(payload))}
        handler.rfile = BytesIO(payload)
        handler.send_json = Mock()
        handler.do_POST()
        response = handler.send_json.call_args.args[0]
        self.assertEqual(response["outputs"][0]["data"], [["cat"]])

    def test_main_against_local_v2_server(self) -> None:
        class_names = (
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck",
        )
        logits = torch.zeros(1, 10)
        logits[0, 5] = 8.0
        self.server.MODEL = "nateraw/vit-base-patch16-224-cifar10"
        model = Mock(return_value=SimpleNamespace(logits=logits))
        model.device = torch.device("cpu")
        model.config = SimpleNamespace(
            num_labels=10, id2label={index: f"LABEL_{index}" for index in range(10)}
        )
        self.server.classifier = SimpleNamespace(
            model=model,
            image_processor=Mock(return_value={"pixel_values": torch.zeros(1, 3, 32, 32)}),
        )

        class DatasetLoader:
            classes = class_names

            def __iter__(self):
                return iter([("image-1", torch.zeros(3, 2, 2))])

            def labels_for(self, image_id):
                return ("dog",)

        example = load_module("remote_vit_example", EXAMPLE_DIR / "open_inference_vit.py")
        server = HTTPServer(("127.0.0.1", 0), self.server.Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as output_directory:
                with (
                    patch.dict(os.environ, {
                        "OPEN_INFERENCE_BASE_URL": f"http://127.0.0.1:{server.server_port}/v2/models/{self.server.MODEL_NAME}",
                            "OPEN_INFERENCE_MODEL": "nateraw/vit-base-patch16-224-cifar10",
                            "HF_MAX_SAMPLES": "1",
                            "OPEN_INFERENCE_LABELS_PATH": "",
                        }, clear=True),
                    patch("transformers.AutoConfig.from_pretrained", return_value=model.config),
                    patch.object(example, "HuggingFaceImageClassificationDataLoader", return_value=DatasetLoader()) as loader_factory,
                    patch.object(example.Path, "cwd", return_value=Path(output_directory)),
                    patch.object(example, "print") as output,
                ):
                    example.main()
                self.assertEqual(loader_factory.call_args.kwargs["max_samples"], 1)
                self.assertEqual(loader_factory.call_args.kwargs["params"]["split"], "test")
                self.assertEqual(loader_factory.call_args.kwargs["params"]["image_column"], "img")
                printed_metrics = next(
                    call.args[0]
                    for call in output.call_args_list
                    if call.args[0].startswith("{")
                )
                metrics = json.loads(printed_metrics)
                saved_metrics = list(
                    (Path(output_directory) / "out").glob("metrics_vit_*.json")
                )
                self.assertEqual(len(saved_metrics), 1)
                self.assertEqual(
                    json.loads(saved_metrics[0].read_text(encoding="utf-8")),
                    metrics,
                )
                self.assertEqual(
                    [metric["name"] for metric in metrics["metrics"]],
                    ["inference_time", "quality_score"],
                )
                self.assertEqual(metrics["metrics"][1]["values"][0]["value"], 1.0)
                self.assertEqual(
                    self.server.classifier.image_processor.call_args.kwargs["images"].shape,
                    (1, 3, 2, 2),
                )
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_vit_and_resnet_clients_save_unique_metrics_files(self) -> None:
        for script_name, prefix in (
            ("open_inference_vit.py", "metrics_vit_"),
            ("open_inference_resnet.py", "metrics_resnet_"),
        ):
            example = load_module(
                f"metrics_{prefix.rstrip('_')}",
                EXAMPLE_DIR / script_name,
            )
            with tempfile.TemporaryDirectory() as temporary_directory:
                with patch.object(
                    example.Path,
                    "cwd",
                    return_value=Path(temporary_directory),
                ):
                    first_path = example.save_metrics({"quality_score": 1.0})
                    second_path = example.save_metrics({"quality_score": 0.5})

                self.assertNotEqual(first_path, second_path)
                self.assertTrue(first_path.name.startswith(prefix))
                self.assertTrue(second_path.name.startswith(prefix))
                self.assertEqual(
                    json.loads(first_path.read_text(encoding="utf-8")),
                    {"quality_score": 1.0},
                )
                self.assertEqual(
                    json.loads(second_path.read_text(encoding="utf-8")),
                    {"quality_score": 0.5},
                )

    def test_default_imagenet_pair_uses_verified_class_order(self) -> None:
        example = load_module("remote_vit_labels", EXAMPLE_DIR / "open_inference_vit.py")
        classes = tuple(f"n{class_id:08d}" for class_id in range(1000))
        loader = SimpleNamespace(classes=classes)
        model_id = "google/vit-base-patch16-224"
        model_config = SimpleNamespace(
            id2label={str(class_id): f"name-{class_id}" for class_id in range(1000)}
        )
        with patch.dict(os.environ, {"OPEN_INFERENCE_LABELS_PATH": ""}):
            with patch("transformers.AutoConfig.from_pretrained", return_value=model_config):
                labels = example.class_labels(loader, "mrm8488/ImageNet1K-val", model_id)
            self.assertEqual(labels[42], classes[42])
            with self.assertRaisesRegex(ValueError, "labels differ"):
                with patch("transformers.AutoConfig.from_pretrained", return_value=model_config):
                    example.class_labels(loader, "another/dataset", model_id)
            with self.assertRaises(ValueError):
                with patch("transformers.AutoConfig.from_pretrained", return_value=model_config):
                    example.class_labels(
                        SimpleNamespace(classes=classes[::-1]),
                        "mrm8488/ImageNet1K-val",
                        model_id,
                    )


if __name__ == "__main__":
    unittest.main()