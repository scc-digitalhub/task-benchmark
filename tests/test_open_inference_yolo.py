from __future__ import annotations

import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch


EXAMPLE_PATH = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "image_classification"
    / "open_inference_simple"
    / "open_inference_yolo.py"
)
spec = importlib.util.spec_from_file_location("open_inference_yolo", EXAMPLE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError("Could not load the YOLO Open Inference example")
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)


class OpenInferenceYoloClientTests(unittest.TestCase):
    class_names = (
        "tench",
        "goldfish",
        "great white shark",
        "tiger shark",
        "hammerhead shark",
        "electric ray",
        "stingray",
        "cock",
        "hen",
        "ostrich",
    )

    def test_default_labels_cover_model_outputs_and_image_net10(self) -> None:
        with patch.dict(os.environ, {"OPEN_INFERENCE_LABELS_PATH": ""}):
            labels = example.class_labels(SimpleNamespace(classes=self.class_names))

        self.assertEqual(len(labels), 1000)
        self.assertEqual([labels[index] for index in range(10)], list(self.class_names))
        self.assertTrue(labels[10].startswith("__imagenet_class_"))

    def test_full_label_mapping_must_preserve_first_ten_classes(self) -> None:
        labels = {index: f"class-{index}" for index in range(1000)}
        with tempfile.TemporaryDirectory() as temporary_directory:
            mapping_path = Path(temporary_directory) / "labels.json"
            mapping_path.write_text(json.dumps(labels), encoding="utf-8")
            with patch.dict(
                os.environ,
                {"OPEN_INFERENCE_LABELS_PATH": str(mapping_path)},
            ):
                with self.assertRaisesRegex(ValueError, "first 10"):
                    example.class_labels(SimpleNamespace(classes=self.class_names))

    def test_invoker_resizes_and_center_crops_to_model_input(self) -> None:
        remote_invoker = Mock()
        client = example.BatchedImageInvoker(remote_invoker)

        client.invoke(torch.rand(3, 100, 200))

        remote_invoker.invoke.assert_called_once()
        model_input = remote_invoker.invoke.call_args.args[0]
        self.assertEqual(tuple(model_input.shape), (1, 3, 224, 224))

    def test_metrics_are_saved_to_unique_json_files(self) -> None:
        metrics = {"quality_score": 1.0}
        with tempfile.TemporaryDirectory() as temporary_directory:
            with patch.object(
                example.Path,
                "cwd",
                return_value=Path(temporary_directory),
            ):
                first_path = example.save_metrics(metrics)
                second_path = example.save_metrics(metrics)

            self.assertNotEqual(first_path, second_path)
            self.assertTrue(first_path.name.startswith("metrics_yolo_"))
            self.assertEqual(json.loads(first_path.read_text(encoding="utf-8")), metrics)
            self.assertEqual(json.loads(second_path.read_text(encoding="utf-8")), metrics)


if __name__ == "__main__":
    unittest.main()
