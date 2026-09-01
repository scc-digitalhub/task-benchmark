# Audio Classification Examples

Run the self-contained `task-inference` example with generated silent PCM audio:

```sh
.venv/bin/python examples/audio_classification/simple_audio_classification.py
```

`superb-ks/superb_ks.py` evaluates `superb/wav2vec2-base-superb-ks` on the SUPERB keyword-spotting test dataset and writes `report_superb_ks.json`. It needs `numpy`, `scipy`, and `datasets` in addition to the project dependencies.

```sh
.venv/bin/python examples/audio_classification/superb-ks/superb_ks.py
```
