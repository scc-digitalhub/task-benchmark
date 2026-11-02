from __future__ import annotations

from math import gcd
from typing import Any

import torch

from task_benchmark.abstract.audio_sample import AudioSample

from .base_invoker import TextTranscriptionInvoker


class HuggingFaceTextTranscriptionInvoker(TextTranscriptionInvoker):
    """Transcribe short audio using an HF speech sequence-to-sequence model."""

    def __init__(
        self,
        model_name: str = "openai/whisper-tiny.en",
        device: str = "cpu",
        model_kwargs: dict[str, Any] | None = None,
        processor_kwargs: dict[str, Any] | None = None,
        generate_kwargs: dict[str, Any] | None = None,
    ) -> None:
        if not model_name:
            raise ValueError("model_name must not be empty")
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        self.device = torch.device(device)
        self.processor = AutoProcessor.from_pretrained(model_name, **dict(processor_kwargs or {}))
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name, **dict(model_kwargs or {}))
        self.model.to(self.device)
        self.model.eval()
        self.generate_kwargs = dict(generate_kwargs or {})
        self.sampling_rate = int(self.processor.feature_extractor.sampling_rate)
        self.max_duration = float(getattr(self.processor.feature_extractor, "chunk_length", 30))

    def invoke(self, test_data: AudioSample) -> str:
        if not isinstance(test_data, AudioSample):
            raise TypeError("test_data must be an AudioSample")
        waveform = test_data.waveform
        if type(test_data.sampling_rate) is not int or test_data.sampling_rate <= 0:
            raise ValueError("sampling_rate must be a positive integer in Hz")
        if (
            not isinstance(waveform, torch.Tensor) or waveform.ndim != 1
            or waveform.numel() == 0 or not waveform.is_floating_point()
            or not torch.isfinite(waveform).all().item()
        ):
            raise ValueError("waveform must be a non-empty, finite mono floating-point tensor")
        if waveform.numel() / test_data.sampling_rate > self.max_duration:
            raise ValueError(f"Audio exceeds {self.max_duration:g} seconds; split it into shorter clips")
        audio = waveform.detach().to(device="cpu", dtype=torch.float32).numpy()
        if test_data.sampling_rate != self.sampling_rate:
            from scipy.signal import resample_poly

            divisor = gcd(test_data.sampling_rate, self.sampling_rate)
            audio = resample_poly(
                audio, self.sampling_rate // divisor, test_data.sampling_rate // divisor
            )
        inputs = self.processor(
            audio, sampling_rate=self.sampling_rate,
            return_tensors="pt", return_attention_mask=True,
        )
        inputs = {name: value.to(self.device) for name, value in inputs.items()}
        with torch.inference_mode():
            tokens = self.model.generate(**inputs, **self.generate_kwargs)
        transcripts = self.processor.batch_decode(tokens, skip_special_tokens=True)
        if len(transcripts) != 1 or not isinstance(transcripts[0], str):
            raise RuntimeError("The model must return one transcription per audio sample")
        return transcripts[0].strip()