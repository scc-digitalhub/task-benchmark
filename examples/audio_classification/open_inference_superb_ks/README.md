# OpenInference SUPERB-KS example

This self-contained example evaluates OpenInference audio classification on the
SUPERB keyword-spotting test dataset (`anton-l/superb_demo`, configuration `ks`).

Start the local server from the repository root:

```bash
docker compose -f examples/audio_classification/open_inference_superb_ks/docker-compose.yml up --build
```

Wait for `Serving on http://0.0.0.0:8080` before running the client. The first
server start downloads model weights; later starts reuse the Docker volume cache.

Then run the benchmark in another terminal:

```bash
.venv/bin/python examples/audio_classification/open_inference_superb_ks/open_inference_superb_ks.py
```

The client downloads the dataset with `datasets`, converts each WAV sample to
raw float32 PCM, and sends audio bytes plus the sample rate to the server. The
server loads `superb/wav2vec2-base-superb-ks` and returns `label`/`score` results
through the OpenInference HTTP endpoint at `http://localhost:8080`.

Set `OPEN_INFERENCE_DATASET_SIZE`, `OPEN_INFERENCE_BATCH_SIZE`,
`OPEN_INFERENCE_MODEL`, or `OPEN_INFERENCE_BASE_URL` to override defaults.
