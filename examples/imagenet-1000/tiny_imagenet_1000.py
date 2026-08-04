import csv
import kagglehub
from pathlib import Path

from task_benchmark import evaluate
from task_benchmark.tasks.image_classification.task import ImageClassificationDataObject

DATASET_CSV = Path(__file__).parent / "tiny-imagenet-200-slice-1000.csv"

# DOWNLOAD THE DATASET 
path = kagglehub.dataset_download("akash2sharma/tiny-imagenet")
IMAGES_ROOT = Path(path) / "tiny-imagenet-200" / "val" / "images"


MODEL_NAME = "microsoft/cvt-13-384"
BATCH_SIZE = 128
DEVICE = "cpu"
REPORT_PATH = Path(__file__).parent / "report_cvt13.json"


def load_data_object(csv_path: Path, images_root: Path) -> ImageClassificationDataObject:
    images_path: list[str] = []
    labels: list[str] = []

    with csv_path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            abs_path = images_root / row["filename"]
            images_path.append(str(abs_path))
            labels.append(row["label"])

    return ImageClassificationDataObject(
        images_path=images_path,
        labels=labels,
    )


if __name__ == "__main__":
    print(f"Dataset : {DATASET_CSV}")
    print(f"Images  : {IMAGES_ROOT}")
    print(f"Model   : {MODEL_NAME}")
    print(f"Device  : {DEVICE}")
    print(f"Batch   : {BATCH_SIZE}")

    data_object = load_data_object(DATASET_CSV, IMAGES_ROOT)

    report = evaluate(
        task="image-classification",
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
