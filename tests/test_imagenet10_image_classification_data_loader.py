from __future__ import annotations

import importlib.util
import sys
import tempfile
import types
import unittest
import zipfile
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMPL_DIR = PROJECT_ROOT / "task_benchmark" / "impl"
DATA_LOADER_DIR = IMPL_DIR / "data_loader"
IMAGE_CLASSIFICATION_DIR = DATA_LOADER_DIR / "image_classification"
SYNSETS = (
    "n01440764",
    "n01443537",
    "n01484850",
    "n01491361",
    "n01494475",
    "n01496331",
    "n01498041",
    "n01514668",
    "n01514859",
    "n01518878",
)


def load_imagenet10_data_loader():
    package_paths = {
        "task_benchmark": PROJECT_ROOT / "task_benchmark",
        "task_benchmark.impl": IMPL_DIR,
        "task_benchmark.impl.data_loader": DATA_LOADER_DIR,
        "task_benchmark.impl.data_loader.image_classification": (
            IMAGE_CLASSIFICATION_DIR
        ),
    }
    package_modules = {}
    for package_name, package_path in package_paths.items():
        package_module = types.ModuleType(package_name)
        package_module.__path__ = [str(package_path)]
        package_modules[package_name] = package_module

    base_module_name = "task_benchmark.impl.data_loader.data_loader"
    base_spec = importlib.util.spec_from_file_location(
        base_module_name,
        DATA_LOADER_DIR / "data_loader.py",
    )
    image_base_name = (
        "task_benchmark.impl.data_loader.image_classification.base_data_loader"
    )
    image_base_spec = importlib.util.spec_from_file_location(
        image_base_name,
        IMAGE_CLASSIFICATION_DIR / "base_data_loader.py",
    )
    loader_spec = importlib.util.spec_from_file_location(
        "imagenet10_image_loader_under_test",
        IMAGE_CLASSIFICATION_DIR / "imagenet10.py",
    )
    if any(spec is None or spec.loader is None for spec in (base_spec, image_base_spec, loader_spec)):
        raise RuntimeError("Could not load ImageNet10 DataLoader modules")

    base_module = importlib.util.module_from_spec(base_spec)
    image_base_module = importlib.util.module_from_spec(image_base_spec)
    loader_module = importlib.util.module_from_spec(loader_spec)
    with patch.dict(sys.modules, package_modules):
        sys.modules[base_module_name] = base_module
        base_spec.loader.exec_module(base_module)
        sys.modules[image_base_name] = image_base_module
        image_base_spec.loader.exec_module(image_base_module)
        loader_spec.loader.exec_module(loader_module)
    return loader_module


imagenet10_module = load_imagenet10_data_loader()
ImageNet10ImageClassificationDataLoader = (
    imagenet10_module.ImageNet10ImageClassificationDataLoader
)


class ImageNet10ImageClassificationDataLoaderTests(unittest.TestCase):
    def create_split(self, root: Path, split: str) -> None:
        for index, synset in enumerate(SYNSETS):
            class_root = root / split / synset
            class_root.mkdir(parents=True)
            Image.new("RGB", (2, 1), color=(index * 20, 0, 0)).save(
                class_root / "sample.png"
            )

    def create_archive(self) -> bytes:
        image = BytesIO()
        Image.new("RGB", (1, 1), color=(255, 0, 0)).save(image, format="PNG")
        archive_data = BytesIO()
        with zipfile.ZipFile(archive_data, "w") as archive:
            for split in ("train", "val"):
                for synset in SYNSETS:
                    archive.writestr(
                        f"{split}/{synset}/sample.png",
                        image.getvalue(),
                    )
        return archive_data.getvalue()

    def test_reads_validation_images_and_maps_synsets_to_yolo_labels(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            self.create_split(root, "val")
            loader = ImageNet10ImageClassificationDataLoader(
                str(root),
                download=False,
                max_samples=1,
            )

            image_id, image = next(loader)

            self.assertEqual(loader.dataset_id, "ultralytics/imagenet10")
            self.assertEqual(image_id, "val/n01440764/sample.png")
            self.assertEqual(loader.labels_for(image_id), ("tench",))
            self.assertEqual(tuple(image.shape), (3, 1, 2))
            self.assertEqual(image.dtype, torch.float32)
            torch.testing.assert_close(image[:, 0, 0], torch.tensor([0.0, 0.0, 0.0]))
            with self.assertRaises(StopIteration):
                next(loader)

    def test_downloads_and_caches_official_split_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "imagenet10"
            with patch.object(
                imagenet10_module.request,
                "urlopen",
                return_value=BytesIO(self.create_archive()),
            ) as urlopen:
                loader = ImageNet10ImageClassificationDataLoader(str(root))

            urlopen.assert_called_once_with(
                imagenet10_module._DATASET_URL,
                timeout=60.0,
            )
            self.assertTrue((root / "train").is_dir())
            self.assertEqual(len(loader._samples), 10)
            self.assertEqual(loader.labels_for("val/n01518878/sample.png"), ("ostrich",))


if __name__ == "__main__":
    unittest.main()
