# Tiny-ImageNet Workflows

The local scripts download `akash2sharma/tiny-imagenet` through KaggleHub, evaluate validation images, and require `kagglehub` and `pandas` in addition to the project dependencies.

```sh
.venv/bin/pip install kagglehub pandas
.venv/bin/python examples/image_classification/imagenet-1000/tiny_imagenet_1000.py
.venv/bin/python examples/image_classification/imagenet-1000/tiny_imagenet_1000_multi_model.py
```

The single-model script evaluates the first 1,000 validation images and writes `report_cvt13.json`. The multi-model script evaluates both configured models and writes JSON reports and `model_comparison.csv` under `reports/`.

`tiny_imagenet_1000_digitalhub.py` is a DigitalHub runtime handler. Provide a Tiny-ImageNet dataset artifact and a `model_name` through the handler inputs; it logs `evaluation_report.json` as `evaluation_report`.
