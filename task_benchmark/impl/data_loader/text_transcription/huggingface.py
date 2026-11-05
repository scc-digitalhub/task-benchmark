from __future__ import annotations

from collections.abc import Mapping
from io import BytesIO
from typing import Any

import torch

from task_benchmark.abstract.audio_sample import AudioSample

from .base_data_loader import TextTranscriptionDataLoader


class HuggingFaceTextTranscriptionDataLoader(TextTranscriptionDataLoader):
    """Load HF audio without TorchCodec and retain reference transcriptions."""

    def __init__(
        self,
        dataset_id: str,
        local_path: str,
        params: dict[str, Any] | None = None,
        *,
        max_samples: int | None = None,
    ) -> None:
        super().__init__(dataset_id, local_path, params, max_samples=max_samples)
        if not dataset_id:
            raise ValueError("dataset_id must not be empty")
        options = dict(self.params)
        self.split = options.pop("split", "validation")
        self.audio_column = options.pop("audio_column", "audio")
        self.text_column = options.pop("text_column", "text")
        self.id_column = options.pop("id_column", "id")
        cache_dir = options.pop("cache_dir", local_path or None)
        try:
            from datasets import Audio, load_dataset
            import soundfile
        except ImportError as exc:
            raise ImportError("Install the project extra '.[text-transcription]'") from exc
        self._soundfile = soundfile
        self.dataset = load_dataset(
            dataset_id, split=self.split, cache_dir=cache_dir, **options
        ).cast_column(self.audio_column, Audio(decode=False))
        self._rows = iter(self.dataset)
        self._index = 0
        self._transcripts: dict[str, str] = {}

    @property
    def current_index(self) -> int:
        return self._index

    def transcript_for(self, sample_id: str) -> str:
        return self._transcripts[sample_id]

    def next(self) -> tuple[str, AudioSample]:
        row = next(self._rows)
        text = row[self.text_column]
        if not isinstance(text, str):
            raise ValueError("Reference transcription must be a string")
        sample_id = row.get(self.id_column) if self.id_column is not None else None
        if sample_id is None:
            sample_id = f"{self.dataset_id}:{self.split}:{self._index}"
        sample_id = str(sample_id)
        if not sample_id or sample_id in self._transcripts:
            raise ValueError(f"Empty or duplicate sample identifier: '{sample_id}'")
        audio = row[self.audio_column]
        if not isinstance(audio, Mapping):
            raise ValueError("Audio must contain encoded bytes or a file path")
        if audio.get("bytes") is not None:
            source = BytesIO(audio["bytes"])
        elif audio.get("path"):
            source = audio["path"]
        else:
            raise ValueError("Audio must contain encoded bytes or a file path")
        waveform, sampling_rate = self._soundfile.read(
            source, dtype="float32", always_2d=True
        )
        waveform = torch.from_numpy(waveform.mean(axis=1).copy())
        if waveform.numel() == 0 or not torch.isfinite(waveform).all().item():
            raise ValueError("Audio must contain finite, non-empty samples")
        sample = AudioSample(waveform=waveform, sampling_rate=int(sampling_rate))
        self._transcripts[sample_id] = text
        self._index += 1
        return sample_id, sample