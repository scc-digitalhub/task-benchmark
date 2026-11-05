# Text Transcription

This task follows the same DataLoader -> Invoker -> Evaluator pattern as image
classification, but consumes audio and returns transcription text.

## Installation and Example

From the repository root, using the project's Python environment:

```powershell
python -m pip install -e ".[text-transcription]"
python examples/text_transcription/huggingface/huggingface_text_transcription.py
```

The example defaults to CPU inference with `openai/whisper-tiny.en`, the
English-only Whisper model with approximately 39 million parameters. It uses
`hf-internal-testing/librispeech_asr_dummy`, configuration `clean`, split
`validation`: 73 short English clips, approximately 9 MB of dataset download.
Only five clips are processed by default. This is a functional demonstration,
not a statistically representative benchmark. Model weights require a separate
initial download and are cached for subsequent runs.

The extra installs Datasets, SoundFile, SciPy, and JiWER. Transformers and
PyTorch are already project dependencies. Audio decoding uses SoundFile with
`datasets.Audio(decode=False)` rather than TorchCodec or an FFmpeg subprocess.
On platforms without bundled libsndfile support, SoundFile may require that
system library. WAV and FLAC are suitable choices for this example.

## Contracts

- `AudioSample` in `task_benchmark.abstract.audio_sample` contains a mono
  floating-point `waveform: Tensor` and `sampling_rate: int` in Hz.
- `TextTranscriptionDataLoader` extends `DataLoader`, requiring
  `next() -> tuple[str, AudioSample]` and `transcript_for(sample_id: str) -> str`.
  The inherited iterator protocol supports `for sample_id, audio in loader`.
- `TextTranscriptionInvoker` extends `Invoker[AudioSample, str]`.
- `TextTranscriptionEvaluator` extends `Evaluator[str]`, retaining finite
  score validation in `[0, 1]`. `Evaluator` is now generic over response type;
  the image-classification evaluator remains `Evaluator[Tensor]`.

These bases are exported from their respective
`task_benchmark.impl.{data_loader,invoker,evaluator}.text_transcription`
packages. They do not register a new legacy `evaluate(task=...)` workflow.

## Hugging Face Implementations

`HuggingFaceTextTranscriptionDataLoader(dataset_id, local_path, params=None)`
uses `local_path` as the HF cache directory. Supported `params` include `split`
(default `validation`), `audio_column` (`audio`), `text_column` (`text`), and
`id_column` (`id`). Other options, such as `name`, `revision`, and `streaming`,
are forwarded to `datasets.load_dataset`. The example sets `name=clean`;
the generic loader does not assume a dataset configuration.

Decoded channels are averaged to mono. No waveform amplitude normalization is
applied. Reference text is stored only after a sample has been yielded.
Missing IDs are generated from dataset ID, split, and row index; empty or
duplicate IDs raise an error. Exhaustion propagates `StopIteration`.

`HuggingFaceTextTranscriptionInvoker` accepts `model_name`, `device` (`cpu` by
default), and optional `model_kwargs`, `processor_kwargs`, and `generate_kwargs`.
It loads `AutoProcessor` and `AutoModelForSpeechSeq2Seq`, switches to evaluation
mode, resamples audio to the processor's sample rate with SciPy when necessary,
and generates text under `torch.inference_mode()`. Special tokens are removed.
The initial implementation supports Whisper-style short-form transcription;
changing the model ID does not guarantee compatibility with arbitrary ASR
architectures such as CTC models.

Whisper expects clips up to 30 seconds. Longer inputs are rejected rather than
silently truncated. Split longer recordings before passing them to this
invoker; long-form chunking, diarization, and timestamps are not implemented.

```python
from task_benchmark.impl.data_loader.text_transcription import (
    HuggingFaceTextTranscriptionDataLoader,
)
from task_benchmark.impl.invoker.text_transcription import (
    HuggingFaceTextTranscriptionInvoker,
)
from task_benchmark.impl.evaluator.text_transcription import WordErrorRateEvaluator

loader = HuggingFaceTextTranscriptionDataLoader(
    "hf-internal-testing/librispeech_asr_dummy", ".cache/huggingface",
    params={"name": "clean", "split": "validation", "streaming": True},
)
invoker = HuggingFaceTextTranscriptionInvoker("openai/whisper-tiny.en", device="cpu")
evaluator = WordErrorRateEvaluator(loader)
sample_id, audio = next(loader)
transcript = invoker.invoke(audio)
score = evaluator.evaluate(sample_id, transcript)
wer = evaluator.word_error_rate(sample_id, transcript)
```

## Evaluation and Report

`WordErrorRateEvaluator` uses JiWER's edit-distance implementation. References
and predictions are lowercased, punctuation is removed, and whitespace is
normalized. This is a simple normalization policy, not Whisper's full English
normalizer; number spellings and other lexical differences still count as
errors. The evaluator depends on the abstract loader, not on Hugging Face.

Raw WER is `(substitutions + deletions + insertions) / reference_word_count`
and can exceed 1. `evaluate()` returns `max(0, 1 - WER)` to respect the generic
quality-score contract. A score of 1 means an exact match after normalization;
0 means WER is at least 1. An empty reference and prediction score 1; words
inserted into an empty reference score 0, following JiWER 4 semantics.

The example prints reference, prediction, WER, and quality for each clip, then
a JSON summary with `mean_sample_wer`, `mean_quality_score`, timings, and
per-sample details. Mean sample WER weights clips equally and is **not** corpus
WER, which weights by reference word count. Timings exclude model/dataset
construction and include preprocessing in each invocation. No confidence score
is fabricated from generated text.

## Configuration

| Environment Variable | Default |
| --- | --- |
| `HF_DATASET_ID` | `hf-internal-testing/librispeech_asr_dummy` |
| `HF_DATASET_CONFIG` | `clean` (set empty to omit) |
| `HF_DATASET_SPLIT` | `validation` |
| `HF_AUDIO_COLUMN` | `audio` |
| `HF_TEXT_COLUMN` | `text` |
| `HF_ID_COLUMN` | `id` |
| `HF_MODEL_ID` | `openai/whisper-tiny.en` |
| `HF_CACHE_DIR` | `.cache/huggingface` |
| `HF_MAX_SAMPLES` | `5` |
| `DEVICE` | `cpu` |

## Tests

```powershell
python -m unittest discover -s tests -p test_text_transcription.py -v
```

Tests decode generated FLAC bytes and exercise resampling and real JiWER
evaluation. HF loading and model generation are simulated, so tests require
the transcription extra but no network or model download.

## Docker and Open Inference V2

The [Whisper HTTP example](../examples/text_transcription/open_inference_simple/README.md)
runs the same model inside a CPU-only Docker container. Its default inference
endpoint is `http://localhost:9082/v2/models/whisper-tiny.en/infer`.

`OpenInferenceTextTranscriptionInvoker` implements `TextTranscriptionInvoker`
using numeric `audio` and `sampling_rate` inputs and a UTF-8 `BYTES` text output.
It can replace the local Hugging Face invoker without changing the loader or
`WordErrorRateEvaluator`. This adapter supports HTTP/HTTPS, not gRPC; the
numeric-tensor `OpenInferenceInvoker` remains unchanged.