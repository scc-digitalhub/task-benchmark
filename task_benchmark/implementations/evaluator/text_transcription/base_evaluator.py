from __future__ import annotations

from task_benchmark.implementations.evaluator.base_evaluator import Evaluator


class TextTranscriptionEvaluator(Evaluator[str]):
    """Abstract evaluator of transcription text against a sample reference."""