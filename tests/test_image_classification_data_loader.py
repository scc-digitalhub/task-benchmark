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
from unittest.mock import patch

import torch
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMPLEMENTATIONS_DIR = PROJECT_ROOT / "task_benchmark" / "implementations"
DATA_LOADER_DIR = IMPLEMENTATIONS_DIR / "data_loader"


def load_local_store_image_classification_data_loader():
    package_paths = {
        "task_benchmark": PROJECT_ROOT / "task_benchmark",
        "task_benchmark.implementations": IMPLEMENTATIONS_DIR,
        "task_benchmark.implementations.data_loader": DATA_LOADER_DIR,
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

    image_loader_spec = importlib.util.spec_from_file_location(
        "local_store_image_classification_loader_under_test",
        DATA_LOADER_DIR / "image_classification" / "local_store.py",
    )
    if image_loader_spec is None or image_loader_spec.loader is None:
        raise RuntimeError("Could not load local-store image classification DataLoader")
    image_loader_module = importlib.util.module_from_spec(image_loader_spec)

    with patch.dict(sys.modules, package_modules):
        sys.modules[base_module_name] = base_module
        base_spec.loader.exec_module(base_module)
        image_loader_spec.loader.exec_module(image_loader_module)

    return image_loader_module.LocalStoreImageClassificationDataLoader


LocalStoreImageClassificationDataLoader = (
    load_local_store_image_classification_data_loader()
)


class LocalStoreImageClassificationDataLoaderTests(unittest.TestCase):
    def test_classifier_requires_labels_for(self) -> None:
        classifier_base = LocalStoreImageClassificationDataLoader.__bases__[0]
        self.assertEqual(classifier_base.__name__, "ImageClassificationDataLoader")

        class IncompleteLoader(classifier_base):
            def next(self) -> tuple[str, torch.Tensor]:
                raise StopIteration

        with self.assertRaises(TypeError):
            IncompleteLoader("sample", "./data")

    def create_dataset(self, root: Path) -> None:
        (root / "images").mkdir()
        (root / "classes.txt").write_text("cat\ndog\n", encoding="utf-8")
        (root / "images.txt").write_text(
            "images/cat.png\nimages/dog.png\n",
            encoding="utf-8",
        )
        (root / "annotations.csv").write_text(
            "image,labels\nimages/cat.png,cat;pet\nimages/dog.png,dog\n",
            encoding="utf-8",
        )
        Image.new("RGB", (2, 1), color=(255, 0, 0)).save(
            root / "images" / "cat.png"
        )
        Image.new("RGB", (1, 2), color=(0, 255, 0)).save(
            root / "images" / "dog.png"
        )

    def test_next_returns_unique_path_tensor_and_advances_context(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            self.create_dataset(root)
            (root / "classes.txt").write_text(
                "cat\ndog\npet\n",
                encoding="utf-8",
            )
            loader = LocalStoreImageClassificationDataLoader("sample", str(root))

            image_id, image_tensor = loader.next()

            self.assertEqual(image_id, "images/cat.png")
            self.assertEqual(loader.current_index, 1)
            self.assertEqual(loader.labels_for(image_id), ("cat", "pet"))
            self.assertEqual(tuple(image_tensor.shape), (3, 1, 2))
            self.assertEqual(image_tensor.dtype, torch.float32)
            torch.testing.assert_close(
                image_tensor[:, 0, 0],
                torch.tensor([1.0, 0.0, 0.0]),
            )

            second_image_id, _ = loader.next()
            self.assertEqual(second_image_id, "images/dog.png")
            self.assertEqual(loader.current_index, 2)
            with self.assertRaises(StopIteration):
                loader.next()

    def test_unknown_annotation_class_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            self.create_dataset(root)
            with self.assertRaisesRegex(ValueError, "Unknown class"):
                LocalStoreImageClassificationDataLoader("sample", str(root))


if __name__ == "__main__":
    unittest.main()