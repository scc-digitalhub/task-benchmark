from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor


@dataclass
class AudioSample:
    """Mono waveform and its sampling rate in Hz."""

    waveform: Tensor
    sampling_rate: int