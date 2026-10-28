# SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMPLEMENTATIONS_DIR = PROJECT_ROOT / "task_benchmark" / "implementations"
DATA_LOADER_DIR = IMPLEMENTATIONS_DIR / "data_loader"
IMAGE_CLASSIFICATION_DIR = DATA_LOADER_DIR / "image_classification"


def load_huggingface_data_loader():
    package_paths = {
        "task_benchmark": PROJECT_ROOT / "task_benchmark",
        "task_benchmark.implementations": IMPLEMENTATIONS_DIR,
        "task_benchmark.implementations.data_loader": DATA_LOADER_DIR,
        "task_benchmark.implementations.data_loader.image_classification": (
            IMAGE_CLASSIFICATION_DIR
        ),
    }
    package_modules = {}
    for package_name, package_path in package_paths.items():
        package_module = types.ModuleType(package_name)
        package_module.__path__ = [str(package_path)]
        package_modules[package_name] = package_module

    base_module_name = "task_benchmark.implementations.data_loader.data_loader"
    base_spec = importlib.util.spec_from_file_location(
        base_module_name,
        DATA_LOADER_DIR / "data_loader.py",
    )
    if base_spec is None or base_spec.loader is None:
        raise RuntimeError("Could not load DataLoader base class")
    base_module = importlib.util.module_from_spec(base_spec)

    loader_spec = importlib.util.spec_from_file_location(
        "huggingface_image_loader_under_test",
        IMAGE_CLASSIFICATION_DIR / "huggingface.py",
    )
    if loader_spec is None or loader_spec.loader is None:
        raise RuntimeError("Could not load Hugging Face image loader")
    loader_module = importlib.util.module_from_spec(loader_spec)

    with patch.dict(sys.modules, package_modules):
        sys.modules[base_module_name] = base_module
        base_spec.loader.exec_module(base_module)
        loader_spec.loader.exec_module(loader_module)

    return loader_module.HuggingFaceImageClassificationDataLoader


HuggingFaceImageClassificationDataLoader = load_huggingface_data_loader()


class ClassLabelFeature:
    names = ["cat", "dog"]


class SequenceFeature:
    feature = ClassLabelFeature()


class FakeDataset:
    features = {"targets": SequenceFeature()}

    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows

    def __iter__(self):
        return iter(self.rows)


class HuggingFaceImageClassificationDataLoaderTests(unittest.TestCase):
    def make_datasets_module(self, dataset: FakeDataset) -> tuple[types.ModuleType, Mock]:
        datasets_module = types.ModuleType("datasets")
        load_dataset = Mock(return_value=dataset)
        datasets_module.load_dataset = load_dataset
        return datasets_module, load_dataset

    def test_loads_configured_split_and_returns_named_image_tensor(self) -> None:
        dataset = FakeDataset(
            [
                {
                    "sample_key": "sample-001",
                    "pixels": Image.new("RGB", (2, 1), color=(255, 0, 0)),
                    "targets": [0, 1],
                }
            ]
        )
        datasets_module, load_dataset = self.make_datasets_module(dataset)

        with tempfile.TemporaryDirectory() as cache_dir:
            with patch.dict(sys.modules, {"datasets": datasets_module}):
                loader = HuggingFaceImageClassificationDataLoader(
                    dataset_id="example/image-dataset",
                    local_path=cache_dir,
                    params={
                        "split": "validation",
                        "name": "small-config",
                        "image_column": "pixels",
                        "label_column": "targets",
                        "id_column": "sample_key",
                        "revision": "main",
                    },
                )
                image_id, image_tensor = loader.next()

        load_dataset.assert_called_once_with(
            "example/image-dataset",
            name="small-config",
            split="validation",
            cache_dir=cache_dir,
            revision="main",
        )
        self.assertEqual(image_id, "sample-001")
        self.assertEqual(loader.classes, ("cat", "dog"))
        self.assertEqual(loader.labels_for(image_id), ("cat", "dog"))
        self.assertEqual(tuple(image_tensor.shape), (3, 1, 2))
        self.assertEqual(image_tensor.dtype, torch.float32)
        torch.testing.assert_close(
            image_tensor[:, 0, 0],
            torch.tensor([1.0, 0.0, 0.0]),
        )

    def test_generates_identifier_and_classes_for_string_labels(self) -> None:
        dataset = FakeDataset(
            [
                {
                    "image": Image.new("L", (1, 1), color=128),
                    "label": "bird",
                }
            ]
        )
        dataset.features = {}
        datasets_module, _ = self.make_datasets_module(dataset)

        with patch.dict(sys.modules, {"datasets": datasets_module}):
            loader = HuggingFaceImageClassificationDataLoader(
                dataset_id="example/birds",
                local_path="",
            )
            image_id, image_tensor = loader.next()

        self.assertEqual(image_id, "example/birds:test:0")
        self.assertEqual(loader.classes, ("bird",))
        self.assertEqual(loader.labels_for(image_id), ("bird",))
        self.assertEqual(tuple(image_tensor.shape), (3, 1, 1))


if __name__ == "__main__":
    unittest.main()