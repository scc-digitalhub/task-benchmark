from __future__ import annotations

from abc import abstractmethod

from task_benchmark.abstract.audio_sample import AudioSample
from task_benchmark.impl.data_loader.data_loader import DataLoader


class TextTranscriptionDataLoader(DataLoader):
    """Abstract source of audio samples and reference transcriptions."""

    @abstractmethod
    def next(self) -> tuple[str, AudioSample]:
        raise NotImplementedError

    @abstractmethod
    def transcript_for(self, sample_id: str) -> str:
        """Return the reference text for a previously yielded sample."""
        raise NotImplementedError