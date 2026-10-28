# SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INVOKER_BASE_PATH = (
    PROJECT_ROOT
    / "task_benchmark"
    / "implementations"
    / "invoker"
    / "base_invoker.py"
)
MODULE_SPEC = importlib.util.spec_from_file_location(
    "invoker_base_under_test",
    INVOKER_BASE_PATH,
)
if MODULE_SPEC is None or MODULE_SPEC.loader is None:
    raise RuntimeError(f"Could not load Invoker base class from {INVOKER_BASE_PATH}")
INVOKER_BASE_MODULE = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(INVOKER_BASE_MODULE)
Invoker = INVOKER_BASE_MODULE.Invoker


class ConcreteInvoker(Invoker[str, str]):
    def invoke(self, test_data: str) -> str:
        return test_data.upper()


class InvokerBaseTests(unittest.TestCase):
    def test_base_class_is_abstract(self) -> None:
        with self.assertRaises(TypeError):
            Invoker()

    def test_concrete_invoker_implements_generic_contract(self) -> None:
        invoker = ConcreteInvoker()

        self.assertEqual(invoker.invoke("image"), "IMAGE")


if __name__ == "__main__":
    unittest.main()