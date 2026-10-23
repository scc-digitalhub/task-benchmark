# SUPERB Keyword Spotting

`superb_ks.py` downloads the `anton-l/superb_demo` `ks` test split, converts its WAV samples to raw PCM float32, evaluates `superb/wav2vec2-base-superb-ks`, and saves `../report_superb_ks.json`.

From the repository root:

```sh
.venv/bin/python examples/audio_classification/superb-ks/superb_ks.py
```

Install the extra runtime dependencies first:

```sh
.venv/bin/pip install numpy scipy datasets
```
