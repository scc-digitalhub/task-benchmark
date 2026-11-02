# Whisper Open Inference V2 Example

This example runs `openai/whisper-tiny.en` on CPU inside Docker and exposes
Open Inference V2 HTTP endpoints. The default host port is **9082**, so it can
run alongside the image-classification server on port 8080.

## Start the Server

From the repository root:

```powershell
docker compose -f examples/text_transcription/open_inference_simple/docker-compose.yml up -d --build
docker compose -f examples/text_transcription/open_inference_simple/docker-compose.yml logs -f whisper-server
```

Wait for `Serving on http://0.0.0.0:9080/v2/models/whisper-tiny.en`.
The build installs CPU-only PyTorch and Transformers. On its first start the
server downloads the model and processor; a named Docker volume caches them.
Model loading must finish before the server becomes ready. The image uses a
non-root user. Dependency ranges and the base image tag are not immutable;
rebuilds may resolve newer packages and should be scanned before production use.

Endpoints exposed on localhost:

- `GET http://localhost:9082/v2/health/live`
- `GET http://localhost:9082/v2/health/ready`
- `GET http://localhost:9082/v2/models/whisper-tiny.en/ready`
- `GET http://localhost:9082/v2/models/whisper-tiny.en`
- `POST http://localhost:9082/v2/models/whisper-tiny.en/infer`

The server loads the model once and processes one request at a time. This is a
local demonstration, not a production inference server: no authentication,
TLS termination, batching, gRPC, or long-form chunking is implemented. Compose
binds the published port to `127.0.0.1` only.

## Run the Dataset Client

```powershell
python -m pip install -e ".[text-transcription]"
python examples/text_transcription/open_inference_simple/open_inference_hf.py
```

The client uses `HuggingFaceTextTranscriptionDataLoader` to stream five clips
from `hf-internal-testing/librispeech_asr_dummy` (`clean`, `validation`). It
passes waveform and sample rate through `OpenInferenceTextTranscriptionInvoker`,
then uses `WordErrorRateEvaluator` to compare the returned text with references.
No model weights are loaded on the client.

The report includes reference/predicted text, raw WER, quality score
`max(0, 1 - WER)`, and timings. `mean_sample_wer` is the mean of per-clip WER,
not corpus WER. HTTP latency includes serialization, transport, and server-side
preprocessing/generation. Dataset construction is excluded from loop elapsed time.
Case, punctuation, and excess whitespace are ignored by evaluation. See the
[transcription guide](../../../docs/text-transcription.md) for normalization
details and limitations.

## V2 Request and Response

The input is one mono waveform in `audio` (FP32, shape `[samples]`), plus
`sampling_rate` (INT64, shape `[1]`). Audio is decoded by the client, not sent
as base64. Rates from 1 to 192000 Hz are accepted and resampled on the server
to the processor's sample rate, normally 16 kHz. Clips exceeding 30 seconds
are rejected, not silently truncated. The request size limit is 16 MiB.

```json
{
  "id": "sample-1",
  "inputs": [
    {"name": "audio", "datatype": "FP32", "shape": [4], "data": [0.0, 0.1, -0.1, 0.0]},
    {"name": "sampling_rate", "datatype": "INT64", "shape": [1], "data": [16000]}
  ],
  "outputs": [{"name": "text"}]
}
```

This tiny waveform illustrates the wire format only; use actual speech for a
meaningful transcription. The response is:

```json
{
  "model_name": "whisper-tiny.en",
  "model_version": "1",
  "id": "sample-1",
  "outputs": [
    {"name": "text", "datatype": "BYTES", "shape": [1], "data": ["Recognized speech."]}
  ]
}
```

`BYTES` carries a UTF-8 string in JSON. The text adapter is separate from the
numeric `OpenInferenceInvoker`, because PyTorch tensors do not hold strings.
Validation errors return HTTP 400 with `{"error": "..."}`; oversized requests
return 413, unknown paths 404, and backend failures 500. Empty recognized text
is valid and is evaluated as a transcription, not a transport failure.

## Configuration

Server Compose variables:

- `WHISPER_PORT`: host port, default `9082`.
- `WHISPER_MODEL`: HF model ID or an available local model path, default
  `openai/whisper-tiny.en`. Only Whisper-compatible sequence-to-sequence models
  are supported. The endpoint alias remains `whisper-tiny.en`.

Client variables:

- `OPEN_INFERENCE_BASE_URL`: default `http://localhost:9082/v2/models/whisper-tiny.en`.
- `HF_MAX_SAMPLES`: default `5`.
- `HF_DATASET_ID`, `HF_DATASET_CONFIG`, `HF_DATASET_SPLIT`: default
  `hf-internal-testing/librispeech_asr_dummy`, `clean`, `validation`.
- `HF_AUDIO_COLUMN`, `HF_TEXT_COLUMN`, `HF_ID_COLUMN`: default `audio`, `text`, `id`.
- `HF_CACHE_DIR`: default `.cache/huggingface`.

When changing `WHISPER_PORT`, also update the client's `OPEN_INFERENCE_BASE_URL`.
Set `HF_DATASET_CONFIG` to an empty string for a dataset without configurations.

## Stop and Test

```powershell
docker compose -f examples/text_transcription/open_inference_simple/docker-compose.yml down
python -m unittest discover -s tests -p test_open_inference_text_transcription.py -v
```

Tests run a local HTTP server with a simulated Whisper backend, without
downloading model weights or contacting Hugging Face.