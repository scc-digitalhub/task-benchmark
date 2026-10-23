from io import BytesIO
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from datasets import Audio, load_dataset
from scipy.io import wavfile

from task_benchmark import evaluate
from task_benchmark.tasks.audio_classification.task import AudioClassificationDataObject


MODEL = os.getenv("OPEN_INFERENCE_MODEL", "superb/wav2vec2-base-superb-ks")
BASE_URL = os.getenv("OPEN_INFERENCE_BASE_URL", "http://localhost:8080")
BATCH_SIZE = int(os.getenv("OPEN_INFERENCE_BATCH_SIZE", "16"))
DATASET_SIZE = int(os.getenv("OPEN_INFERENCE_DATASET_SIZE", "1000"))
REPORT_PATH = Path(__file__).parent / "report_open_inference_superb_ks.json"


def load_data_object(directory: Path) -> AudioClassificationDataObject:
    dataset = load_dataset("anton-l/superb_demo", "ks", split="test")
    label_names = dataset.features["label"].names
    sample_rate = dataset.features["audio"].sampling_rate
    dataset = dataset.cast_column("audio", Audio(decode=False))
    audio_paths, labels = [], []

    for index, sample in enumerate(dataset.select(range(min(DATASET_SIZE, len(dataset))))):
        _, audio = wavfile.read(BytesIO(sample["audio"]["bytes"]))
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32) / np.iinfo(audio.dtype).max
        path = directory / f"sample_{index}.raw"
        path.write_bytes(audio.tobytes())
        audio_paths.append(str(path))
        labels.append(label_names[sample["label"]])

    return AudioClassificationDataObject(audio_paths, labels, sample_rate)


if __name__ == "__main__":
    with TemporaryDirectory() as directory:
        data_object = load_data_object(Path(directory))
        report = evaluate(
            task="audio-classification",
            implementation="open-inference",
            data_object=data_object,
            model_name=MODEL,
            model_params={"base_url": BASE_URL},
            batch_size=BATCH_SIZE,
            report_path=REPORT_PATH,
        )
        print("Top-1 accuracy:", report["top1_accuracy"])
        print("Top-5 accuracy:", report["top5_accuracy"])
        print("Report saved to:", REPORT_PATH)
