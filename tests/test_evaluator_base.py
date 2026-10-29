# SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVALUATOR_BASE_PATH = (
    PROJECT_ROOT
    / "task_benchmark"
    / "implementations"
    / "evaluator"
    / "base_evaluator.py"
)
MODULE_SPEC = importlib.util.spec_from_file_location(
    "evaluator_base_under_test",
    EVALUATOR_BASE_PATH,
)
if MODULE_SPEC is None or MODULE_SPEC.loader is None:
    raise RuntimeError(
        f"Could not load Evaluator base class from {EVALUATOR_BASE_PATH}"
    )
EVALUATOR_BASE_MODULE = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(EVALUATOR_BASE_MODULE)
Evaluator = EVALUATOR_BASE_MODULE.Evaluator


class ConstantEvaluator(Evaluator):
    def __init__(self, score: float) -> None:
        self.score = score

    def _evaluate(self, test_context: str, service_response: torch.Tensor) -> float:
        return self.score


class EvaluatorBaseTests(unittest.TestCase):
    def test_base_class_is_abstract(self) -> None:
        with self.assertRaises(TypeError):
            Evaluator()

    def test_evaluate_returns_validated_float_score(self) -> None:
        evaluator = ConstantEvaluator(0.75)

        score = evaluator.evaluate("sample-1", torch.tensor([0.1, 0.9]))

        self.assertEqual(score, 0.75)
        self.assertIsInstance(score, float)

    def test_evaluate_rejects_scores_outside_zero_to_one(self) -> None:
        for invalid_score in (-0.1, 1.1, float("nan"), float("inf")):
            with self.subTest(score=invalid_score):
                evaluator = ConstantEvaluator(invalid_score)
                with self.assertRaisesRegex(ValueError, "between 0 and 1"):
                    evaluator.evaluate("sample-1", torch.tensor([0.5]))


if __name__ == "__main__":
    unittest.main()