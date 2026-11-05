from __future__ import annotations

from task_benchmark.impl.data_loader.text_transcription.base_data_loader import (
    TextTranscriptionDataLoader,
)

from .base_evaluator import TextTranscriptionEvaluator


class WordErrorRateEvaluator(TextTranscriptionEvaluator):
    """Return max(0, 1 - WER), ignoring case, punctuation and excess spaces."""

    def __init__(self, data_loader: TextTranscriptionDataLoader) -> None:
        try:
            import jiwer
        except ImportError as exc:
            raise ImportError("Install the project extra '.[text-transcription]'") from exc
        self.data_loader = data_loader
        self._jiwer = jiwer
        self._transform = jiwer.Compose([
            jiwer.ToLowerCase(), jiwer.RemovePunctuation(),
            jiwer.RemoveMultipleSpaces(), jiwer.Strip(),
            jiwer.ReduceToListOfListOfWords(),
        ])

    def word_error_rate(self, test_context: str, service_response: str) -> float:
        """Return raw WER, which may exceed 1 when there are many insertions."""
        if not isinstance(service_response, str):
            raise TypeError("service_response must be a string")
        reference = self.data_loader.transcript_for(test_context)
        return float(self._jiwer.process_words(
            reference, service_response,
            reference_transform=self._transform,
            hypothesis_transform=self._transform,
        ).wer)

    def _evaluate(self, test_context: str, service_response: str) -> float:
        return max(0.0, 1.0 - self.word_error_rate(test_context, service_response))