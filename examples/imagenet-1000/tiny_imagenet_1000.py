import kagglehub
import pandas as pd
from pathlib import Path

from task_benchmark import evaluate
from task_benchmark.tasks.image_classification.task import ImageClassificationDataObject


# DOWNLOAD THE DATASET
path = kagglehub.dataset_download("akash2sharma/tiny-imagenet")
DATASET = Path(path) / "tiny-imagenet-200" / "val"

MODEL_NAME = "microsoft/cvt-13-384"
BATCH_SIZE = 128
DEVICE = "cpu"
REPORT_PATH = Path(__file__).parent / "report_cvt13.json"


# builds the pandas DataFrame from the tiny-imagenet validation set
def build_dataframe(val_root: Path) -> pd.DataFrame:
    words_file = val_root.parent / "words.txt"
    class_id_to_label = dict(
        line.strip().split("\t", 1)
        for line in words_file.open()
    )

    images_dir = val_root / "images"
    annotations = val_root / "val_annotations.txt"
    rows = []
    with annotations.open() as fh:
        for line in fh:
            filename, class_id, *_ = line.split("\t")
            rows.append({
                "image_path": str(images_dir / filename),
                "label": class_id_to_label[class_id],
            })
    return pd.DataFrame(rows).head(1000)

# loads the data object from the pandas DataFrame
def load_data_object(df: pd.DataFrame) -> ImageClassificationDataObject:
    return ImageClassificationDataObject(
        images_path=df["image_path"].tolist(),
        labels=df["label"].tolist(),
    )


if __name__ == "__main__":
    df = build_dataframe(DATASET)
    data_object = load_data_object(df)

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
