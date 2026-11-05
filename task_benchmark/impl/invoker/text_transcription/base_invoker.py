from __future__ import annotations

from task_benchmark.abstract.audio_sample import AudioSample
from task_benchmark.impl.invoker.base_invoker import Invoker


class TextTranscriptionInvoker(Invoker[AudioSample, str]):
    """Abstract audio-to-text model invocation interface."""