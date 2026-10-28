# SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_LOADER_PATH = (
    PROJECT_ROOT
    / "task_benchmark"
    / "implementations"
    / "data_loader"
    / "data_loader.py"
)
MODULE_SPEC = importlib.util.spec_from_file_location(
    "data_loader_under_test",
    DATA_LOADER_PATH,
)
if MODULE_SPEC is None or MODULE_SPEC.loader is None:
    raise RuntimeError(f"Could not load DataLoader from {DATA_LOADER_PATH}")
DATA_LOADER_MODULE = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(DATA_LOADER_MODULE)
DataLoader = DATA_LOADER_MODULE.DataLoader


class ConcreteDataLoader(DataLoader):
    def __init__(self, dataset_id: str, local_path: str, params=None) -> None:
        super().__init__(dataset_id, local_path, params)
        self._consumed = False

    def next(self) -> tuple[str, object]:
        if self._consumed:
            raise StopIteration
        self._consumed = True
        return "sample", {"value": 1}


class DataLoaderTests(unittest.TestCase):
    def test_base_class_is_abstract(self) -> None:
        with self.assertRaises(TypeError):
            DataLoader("dataset", "./data")

    def test_concrete_loader_stores_configuration_and_returns_item(self) -> None:
        params = {"batch_size": 8, "shuffle": True}
        loader = ConcreteDataLoader("dataset", "./data", params=params)

        self.assertEqual(loader.dataset_id, "dataset")
        self.assertEqual(loader.local_path, "./data")
        self.assertIs(loader.params, params)
        self.assertEqual(loader.next(), ("sample", {"value": 1}))

    def test_base_iterator_protocol_delegates_to_next(self) -> None:
        loader = ConcreteDataLoader("dataset", "./data")

        self.assertIs(iter(loader), loader)
        self.assertEqual(next(loader), ("sample", {"value": 1}))
        with self.assertRaises(StopIteration):
            next(loader)


if __name__ == "__main__":
    unittest.main()