from __future__ import annotations

import shutil
import stat
import tempfile
import zipfile
from io import BytesIO
from pathlib import Path, PurePosixPath
from urllib import request

import torch
from PIL import Image
from torch import Tensor

from task_benchmark.impl.data_loader.image_classification.base_data_loader import (
    ImageClassificationDataLoader,
)


_DATASET_URL = (
    "https://github.com/ultralytics/yolov5/releases/download/v1.0/imagenet10.zip"
)
_MAX_ARCHIVE_SIZE = 10 * 1024 * 1024
_SYNSET_LABELS = {
    "n01440764": "tench",
    "n01443537": "goldfish",
    "n01484850": "great white shark",
    "n01491361": "tiger shark",
    "n01494475": "hammerhead shark",
    "n01496331": "electric ray",
    "n01498041": "stingray",
    "n01514668": "cock",
    "n01514859": "hen",
    "n01518878": "ostrich",
}
_IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}


class ImageNet10ImageClassificationDataLoader(ImageClassificationDataLoader):
    """Load Ultralytics' 24-image ImageNet10 subset.

    The default ``val`` split contains 12 images from the first ten ImageNet
    classes. If the dataset is missing, it is downloaded to ``local_path``.
    Images are returned as RGB CHW float32 tensors in ``[0, 1]``; labels use
    the class names from Ultralytics' ImageNet configuration.

    ImageNet10 is intended for smoke tests, not accuracy benchmarking.
    """

    def __init__(
        self,
        local_path: str,
        *,
        split: str = "val",
        max_samples: int | None = None,
        download: bool = True,
    ) -> None:
        if not local_path:
            raise ValueError("local_path must not be empty")
        if split not in {"train", "val"}:
            raise ValueError("split must be 'train' or 'val'")
        if type(download) is not bool:
            raise TypeError("download must be a bool")

        super().__init__(
            dataset_id="ultralytics/imagenet10",
            local_path=local_path,
            params={"split": split},
            max_samples=max_samples,
        )
        self.root = Path(local_path).expanduser().resolve()
        self.split = split
        split_root = self.root / split
        if not split_root.is_dir():
            if not download:
                raise FileNotFoundError(
                    f"ImageNet10 split does not exist: {split_root}"
                )
            self._download(self.root)
            split_root = self.root / split
        if not split_root.is_dir():
            raise FileNotFoundError(f"ImageNet10 split does not exist: {split_root}")

        class_directories = {
            path.name for path in split_root.iterdir() if path.is_dir()
        }
        if class_directories != set(_SYNSET_LABELS):
            raise ValueError(
                f"Expected the ten ImageNet10 synset directories in {split_root}"
            )

        self.classes = tuple(_SYNSET_LABELS.values())
        self._samples: list[tuple[str, Path, str]] = []
        self._annotations: dict[str, tuple[str, ...]] = {}
        for synset, label in _SYNSET_LABELS.items():
            class_root = split_root / synset
            image_paths = sorted(
                path
                for path in class_root.iterdir()
                if path.is_file() and path.suffix.casefold() in _IMAGE_SUFFIXES
            )
            if not image_paths:
                raise ValueError(f"No images found for ImageNet10 class '{synset}'")
            for image_path in image_paths:
                image_id = image_path.relative_to(self.root).as_posix()
                self._samples.append((image_id, image_path, label))
                self._annotations[image_id] = (label,)
        self._index = 0

    @property
    def current_index(self) -> int:
        """Index of the next image to be returned."""
        return self._index

    def labels_for(self, image_id: str) -> tuple[str, ...]:
        """Return the ImageNet class name for a previously yielded image."""
        try:
            return self._annotations[image_id]
        except KeyError as exc:
            raise KeyError(f"No annotations found for image '{image_id}'") from exc

    def next(self) -> tuple[str, Tensor]:
        if self._index >= len(self._samples):
            raise StopIteration

        image_id, image_path, _ = self._samples[self._index]
        with Image.open(image_path) as source_image:
            image = source_image.convert("RGB")
            width, height = image.size
            pixels = torch.frombuffer(
                bytearray(image.tobytes()),
                dtype=torch.uint8,
            )
        image_tensor = (
            pixels.reshape(height, width, 3)
            .permute(2, 0, 1)
            .to(dtype=torch.float32)
            .div_(255.0)
        )
        self._index += 1
        return image_id, image_tensor

    @staticmethod
    def _download(root: Path) -> None:
        root.parent.mkdir(parents=True, exist_ok=True)
        if root.exists():
            if not root.is_dir():
                raise NotADirectoryError(f"Dataset path is not a directory: {root}")
            if any(root.iterdir()):
                raise FileNotFoundError(
                    f"ImageNet10 is incomplete at {root}; use an empty directory"
                )
            root.rmdir()

        with request.urlopen(_DATASET_URL, timeout=60.0) as response:
            archive_data = response.read(_MAX_ARCHIVE_SIZE + 1)
        if len(archive_data) > _MAX_ARCHIVE_SIZE:
            raise ValueError("ImageNet10 archive exceeds the 10 MB size limit")

        with tempfile.TemporaryDirectory(
            prefix="imagenet10-",
            dir=root.parent,
        ) as temporary_directory:
            extracted_root = Path(temporary_directory) / "dataset"
            extracted_root.mkdir()
            with zipfile.ZipFile(BytesIO(archive_data)) as archive:
                ImageNet10ImageClassificationDataLoader._extract_safely(
                    archive,
                    extracted_root,
                )
            if not all((extracted_root / split).is_dir() for split in ("train", "val")):
                raise ValueError("ImageNet10 archive must contain train and val splits")
            shutil.move(str(extracted_root), str(root))

    @staticmethod
    def _extract_safely(archive: zipfile.ZipFile, destination: Path) -> None:
        destination = destination.resolve()
        for member in archive.infolist():
            member_path = PurePosixPath(member.filename)
            if (
                member_path.is_absolute()
                or "\\" in member.filename
                or any(part in {"", ".", ".."} for part in member_path.parts)
            ):
                raise ValueError("ImageNet10 archive contains an unsafe path")
            if stat.S_ISLNK(member.external_attr >> 16):
                raise ValueError("ImageNet10 archive must not contain symbolic links")
            target = (destination / Path(*member_path.parts)).resolve()
            if not target.is_relative_to(destination):
                raise ValueError("ImageNet10 archive contains an unsafe path")
        archive.extractall(destination)
