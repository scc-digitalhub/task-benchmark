import struct
import numpy as np
from pathlib import Path
from tempfile import TemporaryDirectory
from io import BytesIO

from datasets import load_dataset, Audio

from task_benchmark import evaluate
from task_benchmark.tasks.audio_classification.task import AudioClassificationDataObject


MODEL_NAME = "superb/wav2vec2-base-superb-ks"
BATCH_SIZE = 16
DEVICE = "cpu"
REPORT_PATH = Path(__file__).parent / "report_superb_ks.json"


def wav_to_pcm_bytes(wav_bytes: bytes) -> tuple[bytes, int]:
    """Convert WAV bytes to raw PCM float32 bytes and sample rate."""
    import scipy.io.wavfile as wavfile
    
    wav_file = BytesIO(wav_bytes)
    sample_rate, data = wavfile.read(wav_file)

    if data.dtype != np.float32:
        max_val = np.iinfo(data.dtype).max
        data = data.astype(np.float32) / max_val

    return data.tobytes(), sample_rate


def build_data_object_from_dataset(tmp_path: Path) -> AudioClassificationDataObject:
    """Load SUPERB keyword spotting dataset and extract audio/label pairs."""
    ds = load_dataset("anton-l/superb_demo", "ks", split="test")
    ds_raw = ds.cast_column("audio", Audio(decode=False))
    
    label_names = ds.features["label"].names
    sample_rate = ds.features["audio"].sampling_rate
    
    audio_paths = []
    labels = []
    
    for idx, sample in enumerate(ds_raw):
        wav_bytes = sample["audio"]["bytes"]
        label_id = sample["label"]
        label_name = label_names[label_id]
        
        pcm_bytes, sr = wav_to_pcm_bytes(wav_bytes)
        audio_file = tmp_path / f"sample_{idx}.raw"
        audio_file.write_bytes(pcm_bytes)
        
        audio_paths.append(str(audio_file))
        labels.append(label_name)
    
    return AudioClassificationDataObject(
        audio_paths=audio_paths,
        labels=labels,
        sample_rate=sample_rate,
    )


if __name__ == "__main__":
    with TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        data_object = build_data_object_from_dataset(tmp_path)

        print(data_object)
        
        report = evaluate(
            task="audio-classification",
            implementation="task-inference",
            data_object=data_object,
            model_name=MODEL_NAME,
            device=DEVICE,
            batch_size=BATCH_SIZE,
            report_path=REPORT_PATH,
        )
        
        print("Top-1 accuracy:", report["top1_accuracy"])
        print("Top-5 accuracy:", report["top5_accuracy"])
        print("Report saved to:", REPORT_PATH)
