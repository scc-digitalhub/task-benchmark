import struct
from pathlib import Path
from tempfile import TemporaryDirectory

from task_benchmark.tasks.audio_classification import AudioClassificationDataObject

from task_benchmark.core import evaluate


SAMPLE_RATE = 16000
DURATION_SECONDS = 1

# Minimal raw PCM float32 audio (1 second of silence at 16 kHz)
SILENT_AUDIO = struct.pack(
    f"<{SAMPLE_RATE * DURATION_SECONDS}f",
    *([0.0] * (SAMPLE_RATE * DURATION_SECONDS)),
)


if __name__ == "__main__":
    with TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        inputs_dir = tmp_path / "audio"
        inputs_dir.mkdir(parents=True, exist_ok=True)

        (inputs_dir / "clip1.raw").write_bytes(SILENT_AUDIO)
        (inputs_dir / "clip2.raw").write_bytes(SILENT_AUDIO)

        data_object = AudioClassificationDataObject(
            audio_paths=[
                str(inputs_dir / "clip1.raw"),
                str(inputs_dir / "clip2.raw"),
            ],
            labels=["Speech", "Music"],
            sample_rate=SAMPLE_RATE,
        )

        report = evaluate(
            model_name="superb/wav2vec2-base-superb-ks",
            task="audio-classification",
            implementation="task-inference",
            data_object=data_object,
            device="cpu",
            batch_size=2,
        )

        print("Top-1 accuracy:", report["top1_accuracy"])
        print("Top-5 accuracy:", report["top5_accuracy"])
